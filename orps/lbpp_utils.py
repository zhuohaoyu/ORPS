import subprocess
import sys
import os
import base64
import traceback

from typing import Optional, List, Dict, Any

from cognitive_complexity.api import get_cognitive_complexity



import os
import json
import re
import ast
import tempfile
import radon.complexity as radon_cc

from hashlib import md5
from base64 import urlsafe_b64encode

from datasets import load_dataset

import traceback

lbpp_part1_task_ids = ['lbpp/0', 'lbpp/5', 'lbpp/10', 'lbpp/11', 'lbpp/12', 'lbpp/14', 'lbpp/17', 'lbpp/19', 'lbpp/22', 'lbpp/24', 'lbpp/25', 'lbpp/27', 'lbpp/28', 'lbpp/30', 'lbpp/32', 'lbpp/34', 'lbpp/35', 'lbpp/37', 'lbpp/38', 'lbpp/43', 'lbpp/45', 'lbpp/46', 'lbpp/48', 'lbpp/53', 'lbpp/54', 'lbpp/55', 'lbpp/56', 'lbpp/57', 'lbpp/58', 'lbpp/59', 'lbpp/63', 'lbpp/64', 'lbpp/66', 'lbpp/67', 'lbpp/69', 'lbpp/70', 'lbpp/72', 'lbpp/73', 'lbpp/74', 'lbpp/75', 'lbpp/76', 'lbpp/79', 'lbpp/80', 'lbpp/83', 'lbpp/84', 'lbpp/85', 'lbpp/90', 'lbpp/91', 'lbpp/95', 'lbpp/100', 'lbpp/106', 'lbpp/107', 'lbpp/109', 'lbpp/110', 'lbpp/111', 'lbpp/116', 'lbpp/119', 'lbpp/124', 'lbpp/127', 'lbpp/128', 'lbpp/130', 'lbpp/131', 'lbpp/132', 'lbpp/133', 'lbpp/135', 'lbpp/136', 'lbpp/137', 'lbpp/139', 'lbpp/142', 'lbpp/143', 'lbpp/144', 'lbpp/145', 'lbpp/146', 'lbpp/147', 'lbpp/153', 'lbpp/154', 'lbpp/155', 'lbpp/158', 'lbpp/159', 'lbpp/160', 'lbpp/161']

lbpp_part2_task_ids = ['lbpp/1', 'lbpp/2', 'lbpp/3', 'lbpp/4', 'lbpp/6', 'lbpp/7', 'lbpp/8', 'lbpp/9', 'lbpp/13', 'lbpp/15', 'lbpp/16', 'lbpp/18', 'lbpp/20', 'lbpp/21', 'lbpp/23', 'lbpp/26', 'lbpp/29', 'lbpp/31', 'lbpp/33', 'lbpp/36', 'lbpp/39', 'lbpp/40', 'lbpp/41', 'lbpp/42', 'lbpp/44', 'lbpp/47', 'lbpp/49', 'lbpp/50', 'lbpp/51', 'lbpp/52', 'lbpp/60', 'lbpp/61', 'lbpp/62', 'lbpp/65', 'lbpp/68', 'lbpp/71', 'lbpp/77', 'lbpp/78', 'lbpp/81', 'lbpp/82', 'lbpp/86', 'lbpp/87', 'lbpp/88', 'lbpp/89', 'lbpp/92', 'lbpp/93', 'lbpp/94', 'lbpp/96', 'lbpp/97', 'lbpp/98', 'lbpp/99', 'lbpp/101', 'lbpp/102', 'lbpp/103', 'lbpp/104', 'lbpp/105', 'lbpp/108', 'lbpp/112', 'lbpp/113', 'lbpp/114', 'lbpp/115', 'lbpp/117', 'lbpp/118', 'lbpp/120', 'lbpp/121', 'lbpp/122', 'lbpp/123', 'lbpp/125', 'lbpp/126', 'lbpp/129', 'lbpp/134', 'lbpp/138', 'lbpp/140', 'lbpp/141', 'lbpp/148', 'lbpp/149', 'lbpp/150', 'lbpp/151', 'lbpp/152', 'lbpp/156', 'lbpp/157']

SELECT_NODE_KEYS_ASCEND_SYMBOL = {"success_ratio":-1, "time_enabled_ns":1, "score":-1, "process_score":-1, "instruction_count":1 ,"branch_misses":1, "page_faults":1}


def get_analysis_string_from_dict(analysis: Dict[str, Any]) -> str:
    final_score = analysis.get('final_score', 0) / 2 * 100
    ret = f'## Overall Execution Score: {final_score:.2f}%\n'
    ret += '## Dynamic Execution Analysis\n'
    if 'dynamic' in analysis and len(analysis['dynamic']) > 0:
        pass_percent = int(analysis['dynamic'].get('success_rate', 0) * 100)
        avg_time = int(analysis['dynamic']['average_metrics'].get('time_enabled_ns', -1))
        avg_instruction_count = int(analysis['dynamic']['average_metrics'].get('instruction_count', -1))
        avg_branch_misses = int(analysis['dynamic']['average_metrics'].get('branch_misses', -1))
        avg_page_faults = int(analysis['dynamic']['average_metrics'].get('page_faults', -1))
        test_results = analysis['dynamic'].get('test_results', [])
    else:
        pass_percent = 0
        avg_time = -1
        avg_instruction_count = -1
        avg_branch_misses = -1
        avg_page_faults = -1
        test_results = []
    
    ret += f'Test case pass rate: {pass_percent}%\nAvg Time: {avg_time}ns\nAvg Instructions: {avg_instruction_count}\nAvg Branch Misses: {avg_branch_misses}\nAvg Page Faults: {avg_page_faults}\n'

    for test in test_results:
        pass_or_fail = 'PASS' if test['metrics']['success'] else 'FAIL'
        time = test['metrics'].get('time_enabled_ns', -1)
        instruction_count = test['metrics'].get('instruction_count', -1)
        branch_misses = test['metrics'].get('branch_misses', -1)
        page_faults = test['metrics'].get('page_faults', -1)
        error = test['metrics'].get('error', '')
        error = f', Error: {error}' if error else ''
        ret += f'\tTest #{test["test_case_index"]}: {pass_or_fail}, Time: {time}ns, Num Instructions: {instruction_count}, Branch Misses: {branch_misses}, Page Faults: {page_faults}{error}\n'

    if 'static' in analysis:
        static_code_length = analysis['static'].get('code_length', 0)
        static_ast_node_count = analysis['static'].get('ast_node_count', 0)
        static_cyclomatic_complexity = analysis['static'].get('cyclomatic_complexity', {})
        static_cognitive_complexity = analysis['static'].get('cognitive_complexity_by_function', {})
    else:
        static_code_length = 0
        static_ast_node_count = 0
        static_cyclomatic_complexity = {}
        static_cognitive_complexity = {}
        
    ret += '## Static Analysis\n'
    ret += f'Code Length: {static_code_length}, AST Node Count: {static_ast_node_count}\n'
    ret += f'Cyclomatic Complexity of each function:\n {static_cyclomatic_complexity}\n'
    ret += f'Cognitive Complexity of each function:\n {static_cognitive_complexity}\n'
    return ret


prompt_templates = {
    "verbose": "Provide a detailed, well-commented Python solution. Explain your approach in the docstring.",
    "efficient": "Implement the most time-efficient Python solution. Explain your approach in the docstring.",
    "concise": "Write the most concise Python code possible.",
    "memory_optimized": "Create a memory-efficient Python implementation. Explain your strategy in the docstring.",
    "recursive": "Implement a recursive Python solution. Explain the base case and recursive step in the docstring.",
    "iterative": "Provide an iterative Python implementation with efficient loop structures.",
    "one_liner": "Create a one-line Python solution if possible, or the most concise multi-line solution.",
    "advanced": "Develop an advanced Python solution using modern language features. Explain complex parts in comments.",
    "mathematical": "Implement a mathematically elegant Python solution. Include the mathematical reasoning in the docstring.",
    "pythonic": "Create the most Pythonic solution possible, utilizing idiomatic Python constructs.",
    "generic": "Design a generic Python solution adaptable for similar problems. Explain its flexibility in comments.",
    "robust": "Implement a robust Python solution with proper error handling and input validation.",
    "optimized_for_large_inputs": "Write a Python solution optimized for very large inputs. Explain your optimization strategy in comments.",
    "minimal_dependencies": "Create a Python solution using only built-in functions and libraries.",
    "scalable": "Develop a scalable Python solution for increasing problem sizes. Explain your scalability approach in comments.",
    "readable": "Write a highly readable and self-explanatory Python solution.",
    "novice_friendly": "Write a Python solution that a beginner could easily understand. Use simple constructs and explanatory comments.",
    "creative": "Implement a creative and unique Python solution. Explain your innovative approach in the docstring.",
    "object_oriented": "Design an object-oriented Python solution. Explain your class structure in comments.",
    "functional": "Develop a functional programming style solution in Python.",
}

metrics_description = '''Below are the meanings of the execution analysis results:
Static Metrics:
Code Length: Lines of code
AST Node Count: Number of Abstract Syntax Tree (AST) nodes of the code, higher means more complex code structure
Cyclomatic Complexity: Number of independent linear paths of the code, indicating its complexity and the difficulty of testing
Cognitive Complexity: Assesses how difficult the code is to understand, considering factors like nesting depth, recursion, and logical complexity.
Class Specific Metrics: Giving each member of the class separate metrics

Dynamic Metrics:
Success Rate: The proportion of test cases the code passes, measuring its functional correctness.
Test Results: Detailed outcomes for each test case, including whether it succeeded, execution time, instruction count, branch misses, page faults, and other performance indicators.

Overall Score: A quantitative score calculated according to everything above.'''


def static_analyze_code(code: str) -> Dict[str, Any]:
    if code is None:
        return {}

    try:
        code_without_comments = mini_clean_code(code)
    except Exception as e:
        raise Exception(f"Failed to sanitize code: {e}")
    
    try:
        ast_tree = ast.parse(code_without_comments)
        cc_result = radon_cc.cc_visit(code_without_comments)
    except Exception as e:
        raise Exception(f"Failed to parse code: {e}")
    
    try:
        function_defs = []
        
        def collect_functions(node, scope_path=""):
            if isinstance(node, ast.ClassDef):
                new_scope = f"{scope_path}.{node.name}" if scope_path else node.name
                for child in node.body:
                    collect_functions(child, new_scope)
                    
            elif isinstance(node, ast.FunctionDef):
                func_full_name = f"{scope_path}.{node.name}" if scope_path else node.name
                function_defs.append((func_full_name, node))
                for child in node.body:
                    collect_functions(child, func_full_name)
            
            elif hasattr(node, 'body'):
                for child in node.body:
                    collect_functions(child, scope_path)
        
        collect_functions(ast_tree)
        
        if not function_defs:
            raise Exception("No function definition found")
        
        cognitive_complexities = {
            name: get_cognitive_complexity(func)
            for name, func in function_defs
        }
        
        complexity_values = list(cognitive_complexities.values())
        complexity_stats = {
            "max": max(complexity_values),
            "sum": sum(complexity_values),
            "mean": sum(complexity_values) / len(complexity_values)
        }
        
        analysis = {
            "code_length": len(code_without_comments.split('\n')),
            "sanitized_code": code,
            "ast_node_count": len(list(ast.walk(ast_tree))),
            "cyclomatic_complexity": {
                func.name: func.complexity
                for func in cc_result
            },
            "cognitive_complexity": complexity_stats,
            "cognitive_complexity_by_function": cognitive_complexities
        }
        
        class_metrics = {}
        
        def analyze_class(node):
            if not isinstance(node, ast.ClassDef):
                return
                
            class_info = {
                "method_count": 0,
                "attribute_count": 0,
                "base_classes": len(node.bases),
                "method_cognitive_complexity": {},
                "nested_classes": {}
            }
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    class_info["method_count"] += 1
                    class_info["method_cognitive_complexity"][item.name] = get_cognitive_complexity(item)
                elif isinstance(item, ast.ClassDef):
                    class_info["nested_classes"][item.name] = analyze_class(item)
                elif isinstance(item, ast.Assign):
                    class_info["attribute_count"] += len(item.targets)
            
            class_metrics[node.name] = class_info
            return class_info
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                analyze_class(node)
        
        analysis["class_metrics"] = class_metrics
        
    except Exception as e:
        raise Exception(f"Failed to calculate cognitive complexity: {e}")

    return analysis

def mini_clean_code(code):
    code = re.sub(r'\n\s*\n', '\n', code)

    code = code.strip() 

    match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        pass

    parsed = ast.parse(code)

    for node in ast.walk(parsed):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
            if (len(node.body) > 0 and
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant)):
                node.body.pop(0)

    return ast.unparse(parsed)

def get_lbpp(split: str, dataset_path: str, if_select_lbpp: bool = False):
    try:
        ds = load_dataset(dataset_path)[split]
    except Exception as e:
        raise Exception(f"Failed to load dataset split: {split} at {dataset_path}, error: {e}")
    task_id_2_task={task['task_id']: task for task in ds}
    
    if if_select_lbpp:
        task_id_sample = list(task_id_2_task.keys())[0]
        assert task_id_sample.startswith('lbpp'), f"Task id should start with lbpp, but got {task_id_sample}"
        
        select_task_ids = lbpp_part2_task_ids
        task_id_2_task = {task_id: task for task_id, task in task_id_2_task.items() if task_id in select_task_ids}
    
    for task_id, task in task_id_2_task.items():
        if 'test_list' in task:
            if (task['test_list'].startswith('"') and task['test_list'].endswith('"')) or (task['test_list'].startswith("'") and task['test_list'].endswith("'")):
                task['test_list'] = task['test_list'][1:-1]
            try:
                test_cases = eval(task['test_list'])
            except Exception as e:
                print(f"Failed to eval test_list: {e}, task_id: {task_id}")
                continue

            task['test_list'] = test_cases
        
    return task_id_2_task

def dynamic_analyze_code_lbpp(
        code: str, 
        task_id: str, 
        task_id_2_task: Dict[str, Dict[str, Any]], 
        path_for_tmp_files: str,
        use_trace_back = False,
    ):
    if code is None:
        return {}
    
    code = mini_clean_code(code)
    
    task_data = task_id_2_task.get(task_id, None)
    if task_data is None:
        raise Exception(f"Task data not found for task_id: {task_id}")
        
    test_setup = task_data.get("test_setup", "")
    test_list = task_data.get("test_list", [])

    if not test_setup or not test_list:
        return {}
    
    test_setup = test_setup.replace("from code import", "from code114514 import")

    temp_dir_path = path_for_tmp_files
    if not os.path.exists(temp_dir_path):
        os.makedirs(temp_dir_path)

    with tempfile.TemporaryDirectory(dir=temp_dir_path,delete=True) as temp_dir:
        os.chmod(temp_dir, 0o777)
        code_path = os.path.join(temp_dir, "code114514.py")
        with open(code_path, "w") as f:
            f.write(code)
        os.chmod(code_path, 0o777)
        
        results = []
        
        for test_idx, test_case in enumerate(test_list):
            test_file = os.path.join(temp_dir, f"test_{test_idx}.py")

            inner_exec_string = f'''import sys\nsys.path.append('{temp_dir}')\n{test_setup}\n{test_case}'''
            inner_exec_string_encoded = base64.b64encode(inner_exec_string.encode('utf-8')).decode('utf-8')
            
            if use_trace_back:
                with open(test_file, "w") as f:
                    f.write(f"""
import sys
import base64
import traceback

from cirron import Collector
sys.path.append("{temp_dir}")


{test_setup}

def run_test():
    try:
        collector = Collector()
        code = base64.b64decode('{inner_exec_string_encoded}'.encode('utf-8')).decode('utf-8')
        globals_dict = dict(globals())
        with collector:
            exec(code, globals_dict)
        
        return {{
            "success": True,
            "time_enabled_ns": collector.counters.time_enabled_ns,
            "instruction_count": collector.counters.instruction_count,
            "branch_misses": collector.counters.branch_misses,
            "page_faults": collector.counters.page_faults,
            "error": None
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": traceback.format_exc(),
            "time_enabled_ns": 0,
            "instruction_count": 0,
            "branch_misses": 0,
            "page_faults": 0
        }}

if __name__ == "__main__":
    result = run_test()
    print("RESULT:", result)
""")
            else:
                with open(test_file, "w") as f:
                    f.write(f"""
import sys
import base64

from cirron import Collector
sys.path.append("{temp_dir}")


{test_setup}

def run_test():
    try:
        collector = Collector()
        code = base64.b64decode('{inner_exec_string_encoded}'.encode('utf-8')).decode('utf-8')
        globals_dict = dict(globals())
        with collector:
            exec(code, globals_dict)
        
        return {{
            "success": True,
            "time_enabled_ns": collector.counters.time_enabled_ns,
            "instruction_count": collector.counters.instruction_count,
            "branch_misses": collector.counters.branch_misses,
            "page_faults": collector.counters.page_faults,
            "error": None
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "time_enabled_ns": 0,
            "instruction_count": 0,
            "branch_misses": 0,
            "page_faults": 0
        }}

if __name__ == "__main__":
    result = run_test()
    print("RESULT:", result)
""")
            os.chmod(test_file, 0o777)

            try:
                result = subprocess.run(
                    [sys.executable, test_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                output = result.stdout.strip()
                test_result = eval(output.split("RESULT:", 1)[1].strip())
                
                results.append({
                    "test_case_index": test_idx,
                    "metrics": test_result
                })
                
            except subprocess.TimeoutExpired:
                results.append({
                    "test_case_index": test_idx,
                    "metrics": {
                        "success": False,
                        "error": "Timeout",
                        "time_enabled_ns": 0,
                        "instruction_count": 0,
                        "branch_misses": 0,
                        "page_faults": 0
                    }
                })
            except Exception as e:
                if use_trace_back:
                    error_str = traceback.format_exc()
                else:
                    error_str = str(e)
                results.append({
                    "test_case_index": test_idx,
                    "metrics": {
                        "success": False,
                        "error": error_str,
                        "time_enabled_ns": 0,
                        "instruction_count": 0,
                        "branch_misses": 0,
                        "page_faults": 0
                    }
                })
            
            if use_trace_back:  
                last_result = results[-1]
                if last_result['metrics']['error'] is not None:
                    if len(last_result['metrics']['error']) > 300:
                        last_result['metrics']['error'] = f'The error traceback is too long, here is last 500 charactors of error traceback: {last_result["metrics"]["error"][-300:]}'
        
        successful_tests = [r for r in results if r["metrics"]["success"]]
        success_rate = len(successful_tests) / len(results) if results else 0
        
        avg_metrics = {}
        if successful_tests:
            for metric in successful_tests[0]["metrics"].keys():
                if metric not in ["success", "error"]:
                    avg_metrics[metric] = sum(
                        r["metrics"][metric] for r in successful_tests
                    ) / len(successful_tests)
        
        return {
            "success_rate": success_rate,
            "test_results": results,
            "average_metrics": avg_metrics
        }

def dynamic_analyze_code_lbpp_alter(
        code: str, 
        task_id: str, 
        task_id_2_task: Dict[str, Dict[str, Any]], 
        path_for_tmp_files: str,
        use_trace_back = False,
    ):
    if code is None:
        return {}
    
    code = mini_clean_code(code)
    
    task_data = task_id_2_task.get(task_id, None)
    if task_data is None:
        raise Exception(f"Task data not found for task_id: {task_id}")
        
    test_list = task_data.get("test_list", [])

    if not test_list:
        return {}

    temp_dir_path = path_for_tmp_files
    if not os.path.exists(temp_dir_path):
        os.makedirs(temp_dir_path)

    with tempfile.TemporaryDirectory(dir=temp_dir_path) as temp_dir:
        os.chmod(temp_dir, 0o777)
        
        results = []
        
        for test_idx, test_case in enumerate(test_list):
            test_file = os.path.join(temp_dir, f"test_{test_idx}.py")
            
            indented_test_case = "\n".join(
                "            " + line if line.strip() else line 
                for line in test_case.split("\n")
            )
            
            if use_trace_back:
                with open(test_file, "w") as f:
                    f.write(f"""
import sys
import math
import traceback
from cirron import Collector

{code}

def run_test():
    try:
        collector = Collector()
        with collector:
            
{indented_test_case}
        
        return {{
            "success": True,
            "time_enabled_ns": collector.counters.time_enabled_ns,
            "instruction_count": collector.counters.instruction_count,
            "branch_misses": collector.counters.branch_misses,
            "page_faults": collector.counters.page_faults,
            "error": None
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": traceback.format_exc(),
            "time_enabled_ns": 0,
            "instruction_count": 0,
            "branch_misses": 0,
            "page_faults": 0
        }}

if __name__ == "__main__":
    result = run_test()
    print("RESULT:", result)
""")
            else:
                with open(test_file, "w") as f:
                    f.write(f"""
import sys
import math
from cirron import Collector

{code}

def run_test():
    try:
        collector = Collector()
        with collector:
            
{indented_test_case}
        
        return {{
            "success": True,
            "time_enabled_ns": collector.counters.time_enabled_ns,
            "instruction_count": collector.counters.instruction_count,
            "branch_misses": collector.counters.branch_misses,
            "page_faults": collector.counters.page_faults,
            "error": None
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "time_enabled_ns": 0,
            "instruction_count": 0,
            "branch_misses": 0,
            "page_faults": 0
        }}

if __name__ == "__main__":
    result = run_test()
    print("RESULT:", result)
""")
            os.chmod(test_file, 0o777)

            try:
                result = subprocess.run(
                    [sys.executable, test_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                output = result.stdout.strip()
                test_result = eval(output.split("RESULT:", 1)[1].strip())
                
                results.append({
                    "test_case_index": test_idx,
                    "metrics": test_result
                })
                
            except subprocess.TimeoutExpired:
                results.append({
                    "test_case_index": test_idx,
                    "metrics": {
                        "success": False,
                        "error": "Timeout",
                        "time_enabled_ns": 5e9,
                        "instruction_count": 0,
                        "branch_misses": 0,
                        "page_faults": 0
                    }
                })
            except Exception as e:
                if use_trace_back:
                    error_str = traceback.format_exc()
                else:
                    error_str = str(e)
                results.append({
                    "test_case_index": test_idx,
                    "metrics": {
                        "success": False,
                        "error": error_str,
                        "time_enabled_ns": 0,
                        "instruction_count": 0,
                        "branch_misses": 0,
                        "page_faults": 0
                    }
                })
            if use_trace_back:  
                last_result = results[-1]
                if last_result['metrics']['error'] is not None:
                    if len(last_result['metrics']['error']) > 300:
                        last_result['metrics']['error'] = f'The error traceback is too long, here is last 500 charactors of error traceback: {last_result["metrics"]["error"][-300:]}'
        
        successful_tests = [r for r in results if r["metrics"]["success"]]
        success_rate = len(successful_tests) / len(results) if results else 0
        
        avg_metrics = {}
        if successful_tests:
            for metric in successful_tests[0]["metrics"].keys():
                if metric not in ["success", "error"]:
                    avg_metrics[metric] = sum(
                        r["metrics"][metric] for r in successful_tests
                    ) / len(successful_tests)
        
        return {
            "success_rate": success_rate,
            "test_results": results,
            "average_metrics": avg_metrics
        }

def calculate_new_preference_score(result: Dict[str, Any]) -> float:
    static_analysis = result.get("static", {})
    code_length = static_analysis.get("code_length", 0)
    ast_node_count = static_analysis.get("ast_node_count", 0)
    
    cyclomatic_complexity_dict = static_analysis.get("cyclomatic_complexity", {})
    cyclomatic_complexity = (
        sum(cyclomatic_complexity_dict.values()) / len(cyclomatic_complexity_dict)
        if cyclomatic_complexity_dict else 0
    )
    
    cognitive_complexity = static_analysis.get("cognitive_complexity", {}).get("mean", 0)

    dynamic_analysis = result.get("dynamic", {})
    success_rate = dynamic_analysis.get("success_rate", 0.0)
    average_metrics = dynamic_analysis.get("average_metrics", {})
    time_enabled_ns = average_metrics.get("time_enabled_ns", 0)
    instruction_count = average_metrics.get("instruction_count", 0)
    branch_misses = average_metrics.get("branch_misses", 0)
    page_faults = average_metrics.get("page_faults", 0)

    try:
        static_score = (
            1 / (1 + ast_node_count / 100) +
            1 / (1 + cyclomatic_complexity) +
            1 / (1 + cognitive_complexity / 10) +
            1 / (1 + code_length / 50)
        ) / 4

        dynamic_score = (
            1 / (1 + time_enabled_ns / 1e7) +
            1 / (1 + instruction_count / 1e5) +
            1 / (1 + branch_misses / 1e3) +
            1 / (1 + page_faults / 100)
        ) / 4

        combined_score = 0.5 * static_score + 0.5 * dynamic_score

        final_score = success_rate * (1 + combined_score)

        return final_score

    except Exception as e:
        return 0.0


def is_valid_analysis(result: Dict[str, Any]) -> bool:
    static_analysis = result.get("static", {})
    static_required_keys = [
        "code_length", "ast_node_count",
        "cyclomatic_complexity", "cognitive_complexity", "cognitive_complexity_by_function",
        "class_metrics"
    ]
    if not static_analysis or not all(key in static_analysis for key in static_required_keys):
        return False
    if len(static_analysis.get("cyclomatic_complexity", {})) == 0:
        return False
    if len(static_analysis.get("cognitive_complexity_by_function", {})) == 0:
        return False

    dynamic_analysis = result.get("dynamic", {})
    dynamic_required_keys = [
        "success_rate", "test_results", "average_metrics"
    ]
    if not dynamic_analysis or not all(key in dynamic_analysis for key in dynamic_required_keys):
        return False
    if len(dynamic_analysis.get("test_results", [])) == 0:
        return False

    average_metrics = dynamic_analysis.get("average_metrics", {})
    average_metrics_required_keys = [
        "time_enabled_ns", "instruction_count", "branch_misses", "page_faults"
    ]
    if not all(key in average_metrics for key in average_metrics_required_keys):
        return False

    return True


class CodeNode:
    def __init__(self, current_code: str, current_programmer_output: str, current_analysis: str, current_criticism: str, current_score: float, current_process_score: int, parent: Optional["CodeNode"] = None, task_id: str = None):
        self.parent = parent
        self.children = []
        self.task_id = task_id
        self.history_codes = []
        self.current_code = current_code
        self.history_programmer_outputs = []
        self.current_programmer_output = current_programmer_output
        self.history_analyses = []
        self.current_analysis = current_analysis
        self.history_criticisms = []
        self.current_criticism = current_criticism
        self.history_scores = []
        self.current_score = current_score
        self.history_process_scores = []
        self.current_process_score = current_process_score
        self.exploration_truncated = False
        self.exploration_early_over = False
        self.exploration_over = False
        if parent:
            self.set_parent(parent)

    def set_parent(self, parent: "CodeNode"):
        self.parent = parent
        if self not in self.parent.children:
            self.parent.add_child(self)
        self.history_codes = self.parent.history_codes.copy()
        self.history_codes.append(self.parent.current_code)
        self.history_programmer_outputs = self.parent.history_programmer_outputs.copy()
        self.history_programmer_outputs.append(self.parent.current_programmer_output)
        self.history_analyses = self.parent.history_analyses.copy()
        self.history_analyses.append(self.parent.current_analysis)
        self.history_criticisms = self.parent.history_criticisms.copy()
        self.history_criticisms.append(self.parent.current_criticism)
        self.history_scores = self.parent.history_scores.copy()
        self.history_scores.append(self.parent.current_score)   
        self.history_process_scores = self.parent.history_process_scores.copy()
        self.history_process_scores.append(self.parent.current_process_score)

    def add_child(self, child: "CodeNode"):
        if child not in self.children:
            self.children.append(child)
        if self is not child.parent:
            child.set_parent(self)

    def hash(self):
        history_codes_str = json.dumps(self.history_codes)
        history_analyses_str = json.dumps(self.history_analyses)
        history_criticisms_str = json.dumps(self.history_criticisms)
        history_programmer_outputs_str = json.dumps(self.history_programmer_outputs)
        history_scores_str = json.dumps(self.history_scores)
        history_process_scores_str = json.dumps(self.history_process_scores)
        hashstr = f"$current_code${self.current_code}$current_programmer_output${self.current_programmer_output}$current_analysis${self.current_analysis}$current_criticism${self.current_criticism}$current_score${self.current_score}$current_process_score${self.current_process_score}$task_id${self.task_id}$history_codes${history_codes_str}$history_programmer_outputs${history_programmer_outputs_str}$history_analyses${history_analyses_str}$history_criticisms${history_criticisms_str}$history_scores${history_scores_str}$history_process_scores${history_process_scores_str}$exploration_truncated${self.exploration_truncated}$exploration_early_over${self.exploration_early_over}$exploration_over${self.exploration_over}"
        hash_digest = md5(hashstr.encode("utf-8")).digest()

        url_safe_hash = urlsafe_b64encode(hash_digest).rstrip(b"=").decode("utf-8")

        return url_safe_hash

    @staticmethod
    def parse_analysis_str(analysis_str: str):
        start_idx = analysis_str.find("{")
        end_idx = analysis_str.rfind("}") + 1
        if start_idx == -1 or end_idx == -1:
            return {}
        return json.loads(analysis_str[start_idx:end_idx])
    
    def calculate_uuid(self):
        self.uuid = self.hash()
        return self.uuid

    def to_dict(self):
        if not hasattr(self, 'dynamic_analysis_on_test_data') or self.dynamic_analysis_on_test_data is None:
            ret = {
                "uuid": self.calculate_uuid(),
                "task_id": self.task_id,
                "history_codes": self.history_codes,
                "current_code": self.current_code,
                "history_programmer_outputs": self.history_programmer_outputs,
                "current_programmer_output": self.current_programmer_output,
                "history_analyses": self.history_analyses,
                "current_analysis": self.current_analysis,
                "history_criticisms": self.history_criticisms,
                "current_criticism": self.current_criticism,
                "history_scores": self.history_scores,
                "current_score": self.current_score,
                "history_process_scores": self.history_process_scores,
                "current_process_score": self.current_process_score,
                "exploration_truncated": self.exploration_truncated,
                "exploration_early_over": self.exploration_early_over,
                "exploration_over": self.exploration_over,
                "parent_uuid": self.parent.uuid if self.parent else None,
                "children_uuids": [c.uuid for c in self.children]
            }
        else:
            ret = {
                "uuid": self.calculate_uuid(),
                "task_id": self.task_id,
                "history_codes": self.history_codes,
                "current_code": self.current_code,
                "history_programmer_outputs": self.history_programmer_outputs,
                "current_programmer_output": self.current_programmer_output,
                "history_analyses": self.history_analyses,
                "current_analysis": self.current_analysis,
                "history_criticisms": self.history_criticisms,
                "current_criticism": self.current_criticism,
                "history_scores": self.history_scores,
                "current_score": self.current_score,
                "history_process_scores": self.history_process_scores,
                "current_process_score": self.current_process_score,
                "exploration_truncated": self.exploration_truncated,
                "exploration_early_over": self.exploration_early_over,
                "exploration_over": self.exploration_over,
                "parent_uuid": self.parent.uuid if self.parent else None,
                "children_uuids": [c.uuid for c in self.children],
                "dynamic_analysis_on_test_data": self.dynamic_analysis_on_test_data
            }
        if hasattr(self, "final_code"):
            ret["final_code"] = self.final_code
        return ret

    def save(self, path: str):
        self.uuid = self.hash()
        safe_task_id = re.sub(r'[^\w\-_]', '_', self.task_id)   
        save_path = os.path.join(path, f"{safe_task_id}_{self.uuid}.json")
        if not os.path.exists(path):
            os.makedirs(path)
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    def to_d3_dict(self):
        return {
            "name": f"{self.current_score:.4f}",
            "uuid": self.uuid,
            "task_id": self.task_id,
            "code": self.current_programmer_output,
            "analysis": self.current_analysis,
            "criticism": self.current_criticism,
            "score": self.current_score,
            "exploration_truncated": self.exploration_truncated,
            "exploration_early_over": self.exploration_early_over,
            "exploration_over": self.exploration_over,
            "children": [child.to_d3_dict() for child in self.children]
        }

    def export_tree_visualization(self, output_path: str):
        def convert_to_echarts_format(node):
            def get_node_color(node):
                if node['exploration_truncated']:
                    return '#CCCCCC'
                elif node['exploration_early_over']:
                    return '#99CC66'
                elif node['exploration_over']:
                    return '#FFFF99'
                else:
                    return '#0099CC'

            return {
                "name": f"{node['score']:.2f}",
                "value": node['score'],
                "itemStyle": {
                    "borderColor": '#993333' if abs(node['score']) < 1e-6 else '#CCCC99',
                    "borderWidth": 2,
                    "color": get_node_color(node)
                },
                "children": [convert_to_echarts_format(child) for child in node['children']],
                "code": node['code'],
                "analysis": node['analysis'],
                "criticism": node['criticism'],
                "task_id": node['task_id'],
                "exploration_truncated": node['exploration_truncated'],
                "exploration_early_over": node['exploration_early_over'],
                "exploration_over": node['exploration_over']
            }

        tree_data = convert_to_echarts_format(self.to_d3_dict())
        
        html_content = '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Code Tree Visualization</title>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
            <style>
                #tree-container {
                    width: 100vw;
                    height: 100vh;
                }
            </style>
        </head>
        <body>
            <div id="tree-container"></div>
            <script>
                const treeData = ''' + json.dumps(tree_data) + '''
                
                const chart = echarts.init(document.getElementById('tree-container'));
                
                const option = {
                    tooltip: {
                        trigger: 'item',
                        formatter: function(params) {
                            const data = params.data;
                            return `
                                <div style="max-width: 300px; white-space: pre-wrap;">
                                    <b>Exploration Truncated:</b> ${data.exploration_truncated}<br>
                                    <b>Exploration Early Over:</b> ${data.exploration_early_over}<br>
                                    <b>Exploration Over:</b> ${data.exploration_over}
                                </div>
                            `;
                        }
                    },
                    series: [{
                        type: 'tree',
                        data: [treeData],
                        top: '5%',
                        bottom: '5%',
                        left: '10%',
                        right: '10%',
                        symbol: 'circle',  
                        symbolSize: 10,
                        orient: 'vertical',
                        layout: 'orthogonal',
                        roam: true,  
                        expandAndCollapse: true,
                        initialTreeDepth: -1, 
                        label: {
                            position: 'top',
                            rotate: 0,
                            verticalAlign: 'middle',
                            align: 'center',
                            fontSize: 12
                        },
                        leaves: {
                            label: {
                                position: 'top',
                                rotate: 0
                            }
                        },
                        animationDurationUpdate: 750,
                        emphasis: {
                            focus: 'descendant'
                        }
                    }]
                };
                
                chart.setOption(option);
                
               
                window.addEventListener('resize', function() {
                    chart.resize();
                });

                
                chart.on('dblclick', function(params) {
                    if (params.data) {
                        const data = params.data;
                        const newWindow = window.open('', '_blank');
                        newWindow.document.write(`
                            <html>
                            <head>
                                <title>Node Details</title>
                                <style>
                                    body { font-family: Arial, sans-serif; padding: 20px; }
                                    pre { white-space: pre-wrap; word-wrap: break-word; }
                                    .status { margin: 10px 0; padding: 10px; background: #f5f5f5; }
                                </style>
                            </head>
                            <body>
                                <h2>Task ID: ${data.task_id}</h2>
                                <h3>Score: ${data.value.toFixed(2)}</h3>
                                <div class="status">
                                    <h4>Exploration Status:</h4>
                                    <ul>
                                        <li>Exploration Truncated: ${data.exploration_truncated}</li>
                                        <li>Exploration Early Over: ${data.exploration_early_over}</li>
                                        <li>Exploration Over: ${data.exploration_over}</li>
                                    </ul>
                                </div>
                                <h4>Code:</h4>
                                <pre>${data.code}</pre>
                                <h4>Analysis:</h4>
                                <pre>${data.analysis}</pre>
                                ${data.criticism ? `<h4>Criticism:</h4><pre>${data.criticism}</pre>` : ''}
                            </body>
                            </html>
                        `);
                    }
                });
            </script>
        </body>
        </html>
        '''
        
        os.makedirs(output_path, exist_ok=True)
        safe_task_id = re.sub(r'[^\w\-_\. ]', '_', self.task_id)
        save_path = os.path.join(output_path, f"{safe_task_id}_{self.uuid}.html")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

