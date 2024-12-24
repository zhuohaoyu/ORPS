from abc import ABC, abstractmethod
from shortuuid import uuid
from typing import Optional, List, Dict, Union
from random import Random
import json, logging
from hashlib import md5
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from tqdm import tqdm


from orps.prompts import MULTIPLE_CHOICE_CHOICES_TMPL, PromptPostprocessor


class PermutationGenerator:
    def __init__(self, seed) -> None:
        self.random = Random(seed)

    def generate_permutation(self, n, keep_ans=None):
        perm = [_ for _ in range(n)]
        self.random.shuffle(perm)
        if keep_ans is not None and perm[keep_ans] != keep_ans:
            pos = perm.index(keep_ans)
            perm[pos], perm[keep_ans] = perm[keep_ans], perm[pos]
        return perm


class Instruction(ABC):
    # a base class for a single instruction

    def __init__(
        self,
        input: str,
        output: Optional[str | List[str]] = None,  # Optional: Model's output
        reference: Optional[
            str | List[str]
        ] = None,  # Optional: A single reference text, or a list of reference texts
        parent_uuid: Optional[
            str
        ] = None,  # Optional: (For augmented data) uuid of the original instruction instance
        generation_config: Optional[Dict] = None,
        extra: Optional[dict] = None,
    ):
        self.input = input
        self.output = output
        self.reference = reference
        self.history = []

        self.parent_uuid = parent_uuid
        self.generation_config = generation_config
        self.extra = extra

        self.uuid = self.hash()
        self.prompt = None  # 'prompt' is reserved for processed text directly fed into inference backends, do not use it for anything else

    def __hash__(self) -> int:
        reference_text = (
            ";".join(self.reference)
            if isinstance(self.reference, list)
            else str(self.reference)
        )
        hashstr = (
            "$instruction$"
            + str(self.input)
            + "$reference$"
            + reference_text
            + "$ex$"
            + json.dumps(self.extra)
        )
        hashstr = hashstr.encode("utf-8")
        hash_digest = md5(hashstr).digest()
        hash_value = int.from_bytes(hash_digest, byteorder="big")

        return hash_value

    def hash(self):
        reference_text = (
            ";".join(self.reference)
            if isinstance(self.reference, list)
            else str(self.reference)
        )
        hashstr = (
            "$instruction$"
            + str(self.input)
            + "$reference$"
            + reference_text
            + "$ex$"
            + json.dumps(self.extra)
        )

        hashstr = hashstr.encode("utf-8")
        hash_digest = md5(hashstr).digest()

        url_safe_hash = urlsafe_b64encode(hash_digest).rstrip(b"=").decode("utf-8")
        return url_safe_hash

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class InstructionDataset(ABC):
    # a base class for instruction datasets
    def __init__(self, **kwargs):
        super().__init__()

        self.tokenizer_name_or_path = kwargs.get("tokenizer_name_or_path", None)
        self.system_prompt = kwargs.get("system_prompt", None)
        self.apply_chat_template = kwargs.get("apply_chat_template", True)

        self.instructions = []
        self.fewshot_examples = []
        self.name_or_path = ""

    def select_fewshot_examples(self, fewshot_dataset, num_shots, seed=1):
        permutation_generator = PermutationGenerator(seed)
        perm = permutation_generator.generate_permutation(
            len(fewshot_dataset), keep_ans=None
        )
        return [fewshot_dataset[i] for i in perm[:num_shots]]

    def generate_prompt_text(self):
        self.post_processor = PromptPostprocessor(
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            system_prompt=self.system_prompt,
        )
        for inst in self.instructions:
            # Note if a instruction is already assigned a prompt, we will not overwrite it. This allows to specify a custom prompt conveniently.
            if self.apply_chat_template:
                inst.prompt = (
                    inst.prompt
                    or self.post_processor.get_full_prompt_from_conversation(
                        conversation=inst, fewshot_examples=self.fewshot_examples
                    )
                )
            else:
                inst.prompt = inst.input

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        return self.instructions[idx]

    def __iter__(self):
        return iter(self.instructions)

    def hash(self):
        problem_hashes = sorted([instruction.uuid for instruction in self.instructions])
        fewshot_hashes = [instruction.uuid for instruction in self.fewshot_examples]
        hashstr = "$".join(problem_hashes)
        hashstr += "#".join(fewshot_hashes)
        if self.tokenizer_name_or_path:
            hashstr += "$tokenizer$" + self.tokenizer_name_or_path
        if self.system_prompt:
            hashstr += "$sysprompt$" + self.system_prompt
        hashstr = hashstr.encode("utf-8")
        hash_digest = md5(hashstr).digest()

        url_safe_hash = urlsafe_b64encode(hash_digest).rstrip(b"=").decode("utf-8")
        return url_safe_hash

    def select_first_n(self, n):
        self.instructions = self.instructions[:n]

    def extend(self, other):
        self.instructions.extend(other.instructions)


if __name__ == "__main__":
    dic = {"a": 1, "b": 2}
    inst = Instruction(
        "Tell me how to ride a bike.",
        reference="You should pedal and balance.",
        extra=dic,
    )
    print(inst.__dict__)
    print(inst.hash())
    print(hash(inst))
