from orps.steps.base_step import BaseStep
from orps.steps.code_process_supervision_lbpp import CodeProcessSupervisionLbpp
from orps.steps.code_reflexsion_lbpp import CodeReflexsionLbpp
from orps.steps.code_best_of_n_lbpp import CodeBestOfNLbpp
from orps.steps.code_CoT_lbpp import CodeCoT
from orps.steps.code_process_supervision_lbpp_noA import CodeProcessSupervisionLbpp_noA
from orps.steps.code_process_supervision_lbpp_sft import CodeProcessSupervisionLbppSft
from orps.steps.code_process_supervision_lbpp_p2od2 import CodeProcessSupervisionLbppP2Od2
from orps.steps.code_process_supervision_lbpp_lineSelf import CodeProcessSupervisionLbpp_lineSelf
from orps.steps.code_process_supervision_lbpp_lineGPT import CodeProcessSupervisionLbpp_lineGPT
from orps.steps.code_process_supervision_lbpp_lineGPTall import CodeProcessSupervisionLbpp_lineGPTall
TYPE_TO_STEP = {
    "code_process_supervision_lbpp": CodeProcessSupervisionLbpp,
    "code_reflexsion_lbpp": CodeReflexsionLbpp,
    "code_best_of_n_lbpp": CodeBestOfNLbpp,
    "code_direct_prompt_lbpp": CodeCoT,
    "code_process_supervision_lbpp_noA": CodeProcessSupervisionLbpp_noA,
    "code_process_supervision_lbpp_sft": CodeProcessSupervisionLbppSft,
    "code_process_supervision_lbpp_p2od2": CodeProcessSupervisionLbppP2Od2,
    "code_process_supervision_lbpp_lineSelf": CodeProcessSupervisionLbpp_lineSelf,
    "code_process_supervision_lbpp_lineGPT": CodeProcessSupervisionLbpp_lineGPT,
    "code_process_supervision_lbpp_lineGPTall": CodeProcessSupervisionLbpp_lineGPTall,
}


def load_step_class(step_type):
    assert step_type in TYPE_TO_STEP
    step_class = TYPE_TO_STEP[step_type]
    return step_class


def load_step(step_type, step_config):
    step_class = load_step_class(step_type)
    step_instance = step_class(**step_config)
    return step_instance
