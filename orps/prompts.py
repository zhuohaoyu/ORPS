from transformers import AutoTokenizer
from typing import List, Dict, Optional
import logging, json
from hashlib import md5
from base64 import urlsafe_b64encode

MULTIPLE_CHOICE_CHOICES_TMPL = {
    "default": """{choice_key}. {choice_text}\n""",
    "numbers": """{choice_key}). {choice_text}\n""",
    "brace": """({choice_key}) {choice_text}\n""",
    "bracket": """[{choice_key}] {choice_text}\n""",
    "qa": """{choice_text}\n""",
    "reasoning": """{choice_key}. {choice_text}\n""",
}

MULTIPLE_CHOICE_PROMPT_TMPL = {
    "default": """### Question: {problem}\n\n### Choices: {choices}\n\n### Answer:""",
    "test_1": """### Question: {problem}\n\n### Choices:\n{choices}\n### Answer:\n\n""",
    "qa": """Question: {problem}\nAnswer:""",
    "reasoning": """### {problem}\n\n### Choices: {choices}\n\n### Answer:\n\n""",
}


SYSTEM_PROMPTS = {
    "default": "You are a helpful, respectful and honest assistant. {prompt}",
    "llama2": """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant that answers multiple choice problem accurately. You should be very strict on format and answer only the choice without any explanation. Only output in format `### Answer: {{answer_key}}`, this is very important.\n<</SYS>> \n{prompt}""",
}

def apply_system_prompt(prompt, system_prompt_type="default"):
    if system_prompt_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid system_prompt_type: {system_prompt_type}")
    return SYSTEM_PROMPTS[system_prompt_type].format(prompt=prompt)


def apply_multiple_choice_prompt(
    problem: str, choices: str, tmpl_name: str = "default"
):
    if tmpl_name not in MULTIPLE_CHOICE_PROMPT_TMPL:
        raise ValueError(f"Invalid tmpl_name: {tmpl_name}")
    tmpl = MULTIPLE_CHOICE_PROMPT_TMPL[tmpl_name]
    return tmpl.format(problem=problem, choices=choices)


class PromptPostprocessor:
    def __init__(
        self,
        tokenizer_name_or_path: str = None,
        multiple_choice_template_name: str = "default",
        system_prompt: Optional[str] = None,
        remove_system_prompt: bool = False,
        add_generation_prompt: bool = False,
        convert_system_prompt_to_user_prompt: bool = False,
    ):
        self.logger = logging.getLogger(__name__)

        if tokenizer_name_or_path:
            self.logger.warning(f"Loading AutoTokenizer from: {tokenizer_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, verbose=False, trust_remote_code=True
            )
        else:
            self.logger.warning(
                f"No tokenizer_name_or_path provided, using only `content` field in conversation."
            )
            self.tokenizer = None

        self.multiple_choice_template_name = multiple_choice_template_name
        self.multiple_choice_template = MULTIPLE_CHOICE_PROMPT_TMPL[
            multiple_choice_template_name
        ]
        self.system_prompt = system_prompt
        self.remove_system_prompt = remove_system_prompt
        self.add_generation_prompt = add_generation_prompt
        self.convert_system_prompt_to_user_prompt = convert_system_prompt_to_user_prompt

    def get_full_prompt_from_conversation(
        self, conversation: List[Dict[str, str]], sep="\n\n", rm_last_sep=False
    ) -> str:
        if self.tokenizer is not None:
            if self.remove_system_prompt and conversation[0]["role"] == "system":
                txt = self.tokenizer.apply_chat_template(
                    conversation[1:],
                    tokenize=False,
                    padding=False,
                    truncation=False,
                    add_generation_prompt=self.add_generation_prompt,
                )
            elif self.convert_system_prompt_to_user_prompt and conversation[0]["role"] == "system":
                if len(conversation) > 1 and conversation[1]["role"] == "user":
                    conversation[1]["content"] = conversation[0]["content"] + '\n\n' + conversation[1]["content"]
                conversation = conversation[1:]
                txt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    padding=False,
                    truncation=False,
                    add_generation_prompt=self.add_generation_prompt,
                )
            else:
                txt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    padding=False,
                    truncation=False,
                    add_generation_prompt=self.add_generation_prompt,
                )
        else:
            txt = ""
            for conv in conversation:
                txt += f"{conv['content']}{sep}"
            if rm_last_sep:
                txt = txt[: -len(sep)]
        return txt

    def create_conversation_from_mcp(self, problem, fewshot_examples) -> List[Dict]:
        conversation = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )

        if len(fewshot_examples) > 0:
            for eg in fewshot_examples:
                question_txt = apply_multiple_choice_prompt(
                    eg.problem,
                    eg.generate_choices_text(self.multiple_choice_template_name),
                    self.multiple_choice_template_name,
                )
                question_ans = eg.generate_output_text(
                    self.multiple_choice_template_name
                )
                conversation.append({"role": "user", "content": question_txt})
                conversation.append({"role": "assistant", "content": question_ans})

        question_txt = apply_multiple_choice_prompt(
            problem.problem,
            problem.generate_choices_text(self.multiple_choice_template_name),
            self.multiple_choice_template_name,
        )
        conversation.append({"role": "user", "content": question_txt})
        return conversation

    def create_conversation_from_qa(self, problem, fewshot_examples) -> List[Dict]:
        conversation = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )

        if len(fewshot_examples) > 0:
            for eg in fewshot_examples:
                question_txt = apply_multiple_choice_prompt(
                    eg.problem,
                    eg.generate_choices_text(self.multiple_choice_template_name),
                    self.multiple_choice_template_name,
                )
                question_ans = eg.generate_output_text(
                    self.multiple_choice_template_name
                )
                conversation.append({"role": "user", "content": question_txt})
                conversation.append({"role": "assistant", "content": question_ans})

        question_txt = apply_multiple_choice_prompt(
            problem.problem,
            problem.generate_choices_text(self.multiple_choice_template_name),
            self.multiple_choice_template_name,
        )
        conversation.append({"role": "user", "content": question_txt})
        return conversation

    def get_full_prompt_from_problem(self, problem, fewshot_examples) -> str:
        if len(problem.choices) > 1:  # multiple choice
            conversation = self.create_conversation_from_mcp(problem, fewshot_examples)
            return self.get_full_prompt_from_conversation(
                conversation, rm_last_sep=False
            )
        else:  # cloze question
            conversation = self.create_conversation_from_qa(problem, fewshot_examples)
            return self.get_full_prompt_from_conversation(
                conversation, rm_last_sep=True
            )
