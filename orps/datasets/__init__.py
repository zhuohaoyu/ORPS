from orps.datasets.multiple_choice import MultipleChoiceDataset
import logging

# register new datasets here
from orps.datasets.arc import ARCChallengeDataset
from orps.datasets.mmlu import MMLUDataset
from orps.datasets.truthful_qa import TruthfulQADataset
from orps.datasets.hellaswag import HellaSwagDataset
from orps.datasets.self_instruct import SelfInstructDataset
from orps.datasets.ceval import CEvalDataset
from orps.datasets.gsm8k import GSM8KDataset
from orps.datasets.medmcqa import MedMCQADataset
from orps.datasets.hotpotqa import HotpotQADataset
from orps.datasets.reclor import ReclorDataset

TYPE_TO_DATASET = {
    "arc_challenge": ARCChallengeDataset,
    "mmlu": MMLUDataset,
    "truthful_qa": TruthfulQADataset,
    "hellaswag": HellaSwagDataset,
    "ceval": CEvalDataset,
    "self_instruct": SelfInstructDataset,
    "gsm8k": GSM8KDataset,
    "medmcqa": MedMCQADataset,
    "hotpot_qa": HotpotQADataset,
    "reclor": ReclorDataset,
}


def load_eval_dataset_class(type):
    assert type in TYPE_TO_DATASET
    dset_class = TYPE_TO_DATASET[type]

    return dset_class


def load_eval_dataset_group(dataset_kwargs):
    dset_instance = None
    for kwargs in dataset_kwargs:
        if dset_instance is None:
            dset_instance = load_eval_dataset(kwargs["type"], kwargs["dataset_kwargs"])
        else:
            dset_instance.extend(
                load_eval_dataset(kwargs["type"], kwargs["dataset_kwargs"])
            )
    return dset_instance


def load_eval_dataset(type, dataset_kwargs):
    if type == "group":
        return load_eval_dataset_group(dataset_kwargs)

    dset_class = load_eval_dataset_class(type)

    logging.getLogger(__name__).info(f"Loading dataset with kwargs: {dataset_kwargs}")

    dset_instance = dset_class(**dataset_kwargs)

    if "first_n" in dataset_kwargs:
        dset_instance.select_first_n(dataset_kwargs["first_n"])

    return dset_instance
