from typing import Optional, Union, List, Dict
from orps.prompts import TriloguePrompter
from orps.utils import parse_json


class Conversation:
    def __init__(
        self, uuid: str, messages: List[Dict] = [], random_seed: Optional[int] = 0
    ) -> None:
        self.uuid = uuid
        self.messages = messages
        self.random_seed = random_seed
        self.prompt = None
        self.stop_interaction = False

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"content": content, "role": role})


KIEVAL_EVALUATOR_METRICS = [
    "accuracy",
    "logic",
    "relevance",
    "coherence",
    "conciseness",
]


class Triologue(Conversation):
    def __init__(
        self,
        uuid: str,
        init_messages: Dict,
        random_seed: Optional[int] = 0,
    ):
        super().__init__(uuid, messages=[], random_seed=random_seed)

        self.all_messages = []  # no system prompt
        self.current_party = None

        self.role_messages = init_messages

        self.evaluation_results = []
        self.aggregated_result = None

    def parse_evaluation_result(self, content: str):
        if self.stop_interaction:
            return None
        j = parse_json(content)
        for key in KIEVAL_EVALUATOR_METRICS:
            if key not in j:
                j[key] = {"comment": "", "score": None}
            if "comment" not in j[key]:
                j[key]["comment"] = ""
            if "score" not in j[key]:
                j[key]["score"] = None
        if "comment" not in j:
            j["comment"] = ""
        if "overall_score" not in j:
            j["overall_score"] = None
        if "stop_conversation" not in j:
            j["stop_conversation"] = False

        if j["stop_conversation"]:
            self.stop_interaction = True

        self.evaluation_results.append(j)

        return j

    def aggregate_result(self):
        if len(self.evaluation_results) == 0:
            return None
        if self.aggregated_result is not None:
            return self.aggregated_result
        j = {}
        for key in KIEVAL_EVALUATOR_METRICS:
            values = [e.get(key, None) for e in self.evaluation_results]
            values = [v.get("score", None) for v in values if v is not None]
            values = [v for v in values if v is not None]
            values_normalized = []
            for v in values:
                if isinstance(v, str):
                    try:
                        v = float(v)
                        if v >= 0 and v <= 5:
                            values_normalized.append(v)
                    except:
                        continue
                elif v >= 0 and v <= 5:
                    values_normalized.append(v)
            values = values_normalized
            if len(values) == 0:
                j[key] = None
            else:
                try:
                    j[key] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                    }
                except:
                    print(values)
                    raise ValueError("Error when aggregating results.")
        values = [e.get("overall_score", None) for e in self.evaluation_results]
        values = [v for v in values if v is not None]
        if len(values) == 0:
            j["overall_score"] = None
        else:
            j["overall_score"] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
        self.aggregated_result = j
        return j

    def set_party(self, party: str) -> None:
        if party not in ["candidate", "interactor", "evaluator"]:
            raise ValueError(
                "Party role not recognized. Valid roles are 'candidate', 'interactor', 'evaluator'."
            )
        self.current_party = party
        self.messages = self.role_messages[party]

    def add_message(self, party: str, content: str, prompter: TriloguePrompter) -> None:
        self.all_messages.append({"content": content, "role": party})
        if party == "candidate":
            self.role_messages["candidate"].append(
                {"role": "assistant", "content": content}
            )
            self.role_messages["interactor"].append(
                {"role": "user", "content": content}
            )

            evaluator_user_messages = self.all_messages[
                -2:
            ]  # last two messages are from interactor and candidate
            assert (
                evaluator_user_messages[0]["role"] == "interactor"
            ), f"Last message is not from interactor: {evaluator_user_messages[0]}"
            evaluator_user_prompt = prompter.apply_evaluator_user_prompt(
                interactor_content=evaluator_user_messages[0]["content"],
                candidate_content=evaluator_user_messages[1]["content"],
            )
            self.role_messages["evaluator"].append(
                {"role": "user", "content": evaluator_user_prompt}
            )

        elif party == "interactor":
            self.role_messages["interactor"].append(
                {"role": "assistant", "content": content}
            )
            self.role_messages["candidate"].append({"role": "user", "content": content})

        elif party == "evaluator":
            self.role_messages["evaluator"].append(
                {"role": "assistant", "content": content}
            )
