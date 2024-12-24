
import itertools
import multiprocessing
import os
import time
from multiprocessing import Array, Value, Manager
from typing import Any, Dict, List, Tuple, Union
from cirron import Collector
import numpy as np

from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS, _poly
from evalplus.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)


def compatible_eval_result(results: Dict) -> Dict:
    for task_results in results["eval"].values():
        if "files" in task_results and "nfiles" not in task_results:
            task_results["nfiles"] = len(task_results.pop("files"))
    return results


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def is_floats(x) -> bool:
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def unsafe_execute(
    dataset: str,
    entry_point: str,
    code: str,
    inputs,
    expected: List,
    time_limits,
    atol,
    fast_check,
    stat: Value, # type: ignore
    details: Array, # type: ignore  
    progress: Value, # type: ignore
    performance: Dict[str, Value], # type: ignore
):
    with create_tempdir():
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        maximum_memory_bytes = 4 * 1024 * 1024 * 1024
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)
        exec_globals = {}
        try:
            with swallow_io():
                collector = Collector()
                with collector:
                    exec(code, exec_globals)
                    fn = exec_globals[entry_point]
                    for i, inp in enumerate(inputs):
                        try:
                            with time_limit(time_limits[i]):
                                out = fn(*inp)

                            exp = expected[i]
                            exact_match = out == exp

                            if dataset == "mbpp":
                                if (
                                    "are_equivalent" == entry_point
                                ):
                                    exact_match = exact_match or True
                                elif "sum_div" == entry_point:
                                    exact_match = exact_match or out == 0
                                elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                                    if isinstance(out, bool):
                                        exact_match = out == exp
                                    else:
                                        exact_match = exp == (out is not None)

                            if dataset == "humaneval":
                                if "find_zero" == entry_point:
                                    assert _poly(*inp, out) <= atol

                            if atol == 0 and is_floats(exp):
                                atol = 1e-6
                            if not exact_match and atol != 0:
                                np.testing.assert_allclose(out, exp, atol=atol)
                            else:
                                assert exact_match
                        except BaseException:
                            if fast_check:
                                raise

                            details[i] = False
                            progress.value += 1
                            continue

                        details[i] = True
                        progress.value += 1
                performance['time_enabled_ns'].value = collector.counters.time_enabled_ns
                performance['instruction_count'].value = collector.counters.instruction_count
                performance['branch_misses'].value = collector.counters.branch_misses
                performance['page_faults'].value = collector.counters.page_faults
            stat.value = _SUCCESS


        except BaseException:
            stat.value = _FAILED
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(
    dataset: str,
    code: str,
    inputs: List[Any],
    entry_point: str,
    expected,
    atol,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
) -> Tuple[str, np.ndarray, Dict[str, int]]:
    
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = min(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", 60), sum(time_limits)) + 1
    if not fast_check:
        timeout += 1

    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(inputs))])

    with Manager() as manager:
        performance = {
            'time_enabled_ns': Value('L', 0),
            'instruction_count': Value('L', 0),
            'branch_misses': Value('L', 0),
            'page_faults': Value('L', 0)
        }


        p = multiprocessing.Process(
            target=unsafe_execute,
            args=(
                dataset,
                entry_point,
                code,
                inputs,
                expected,
                time_limits,
                atol,
                fast_check,
                stat,
                details,
                progress,
                performance,
            ),
        )
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.terminate()
            time.sleep(0.1)
        if p.is_alive():
            p.kill()
            time.sleep(0.1)

        status = _mapping[stat.value]
        details_array = np.array(details[:progress.value], dtype=bool)
        performance_dict = {k: v.value for k, v in performance.items()}


    if not status:
        status = TIMEOUT

    if status == PASS:
        if len(details_array) != len(inputs) or not all(details_array):
            status = FAIL

    return status, details_array, performance_dict


def evaluate_files(
    dataset: str,
    files: List[str],
    inputs: List,
    expected: List,
    entry_point: str,
    atol: float,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = 0.1,
    gt_time_limit_factor: float = 2.0,
) -> List[Tuple[str, List[bool], Dict[str, int]]]:
    ret = []
    files = sorted(files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    for file in files:
        code = open(file, "r").read()
        stat, det, perf = untrusted_check(
            dataset,
            code,
            inputs,
            entry_point,
            expected=expected,
            atol=atol,
            ref_time=ref_time,
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )
        perf_list = {k: v.tolist() for k, v in perf.items()}
        ret.append((stat, det.tolist(), perf_list))
    return ret


"""Purpose of this file: Sanitize the code produced by LLMs for the following reasons.
1. Vicuna generated code could miss one white space. We fix the white space to make Vicuna more capable.
2. {Our fault lol.} We find more EOFs tokens afterwards and truncate some messy code afterwards.
"""

import ast
import re
import traceback
from typing import List, Optional


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def remove_unindented_lines(code, protect_before, execeptions, trim_tails):
    lines = code.splitlines()
    cut_idx = []
    cut_enabled = False
    for i, line in enumerate(lines):
        if not cut_enabled and line.startswith(protect_before):
            cut_enabled = True
            continue
        if line.strip() == "":
            continue
        if any(line.startswith(e) for e in execeptions):
            continue

        lspace = len(line) - len(line.lstrip())
        if lspace == 0:
            cut_idx.append(i)

        if any(line.rstrip().startswith(t) for t in trim_tails):
            cut_idx.extend(list(range(i, len(lines))))
            break

    return "\n".join([line for i, line in enumerate(lines) if i not in cut_idx])


def to_four_space_indents(old_code):
    new_code = ""
    for line in old_code.splitlines():
        lspace = len(line) - len(line.lstrip())
        if lspace == 3:
            new_code += " "
        new_code += line + "\n"
    return new_code


def sanitize(
    old_code: str,
    entry_point: str,
    rm_prefix_lines: Optional[str] = None,
    eofs: List = None,
):
    new_code = old_code
    
    def format_single_line_function(code: str) -> str:
        lines = []
        for line in code.splitlines():
            if line.strip().startswith('def ') and ': return' in line:
                func_def, ret = line.split(': return')
                lines.append(func_def + ':')
                lines.append('    return' + ret)
            else:
                lines.append(line)
        return '\n'.join(lines)
    
    new_code = format_single_line_function(new_code)
    
    if rm_prefix_lines is not None:
        new_code = "\n".join(
            [
                line
                for line in old_code.splitlines()
                if not line.startswith(rm_prefix_lines)
            ]
        )

    new_code = "\n" + new_code
    def_left = "def " + entry_point

    new_code = new_code.replace("\n```python\n", "\n```\n")
    for chunk in new_code.split("\n```\n"):
        if def_left in chunk:
            new_code = chunk
            break

    chunks = [chunk for chunk in re.split(f"{def_left}\s*\(", new_code)]
    bodies = [chunk for chunk in chunks[1:] if "    return " in chunk.split("\ndef")[0]]
    def_left = def_left + "("
    new_code = def_left + def_left.join(bodies) if len(bodies) > 0 else ""
    new_code = to_four_space_indents(new_code)

    for eof in eofs or []:
        new_code = new_code.split(eof)[0]

    new_code = remove_unindented_lines(
        new_code,
        protect_before=def_left,
        execeptions=["def ", "import ", "from "],
        trim_tails=['"""', "if", "print"],
    )
    new_code = chunks[0] + new_code

    parts = new_code.split("\ndef ")
    includes = [parts[0]]
    for fn in new_code.split("\ndef ")[1:]:
        if (
            fn.strip().startswith(entry_point + " ")
            or fn.strip().startswith(entry_point + "(")
            or syntax_check("\ndef " + fn)
        ):
            includes.append(fn)
    new_code = "\ndef ".join(includes)
    return new_code.strip()
