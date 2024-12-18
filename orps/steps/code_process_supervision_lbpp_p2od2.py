import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Optional, List, Dict, Any
from orps.models import load_inference_function
from orps.steps.base_step import BaseStep
from orps.core.context import Context
from orps.utils import calculate_inference_endpoint_hash, get_model_nicename
from orps.datasets.instructions import Instruction, InstructionDataset
from orps.prompts import PromptPostprocessor
from cognitive_complexity.api import get_cognitive_complexity

from orps.evalplus_utils import untrusted_check, sanitize

from tqdm import tqdm

import traceback
import logging
import os
import json
import codecs
import jsonlines
import re
import ast
import tempfile
import radon.complexity as radon_cc
from cirron import Collector
import random
from collections import defaultdict
import itertools
from random import choice
from huggingface_hub import InferenceClient

from hashlib import md5
from base64 import urlsafe_b64encode

from datasets import load_dataset

from orps.lbpp_utils import get_lbpp, mini_clean_code, static_analyze_code, dynamic_analyze_code_lbpp, calculate_new_preference_score, is_valid_analysis, dynamic_analyze_code_lbpp_alter,get_analysis_string_from_dict
from orps.lbpp_utils import CodeNode
from orps.lbpp_utils import prompt_templates, metrics_description, SELECT_NODE_KEYS_ASCEND_SYMBOL

import pdb

PROGRAMMER_SYSTEM_PROMPT = """Always follow the format below for each round:

# === BEGIN SOLUTION ===
```python
[Your code]
```
# === END SOLUTION ===

Output strictly as shown above."""






PROGRAMMER_USER_PROMPT_FIRST_ROUND = """Here's your coding task:
{base_prompt}

"""

PROGRAMMER_USER_PROMPT_LATER_ROUNDS = """
{analysis}

{criticism}

"""





class CodeProcessSupervisionLbppP2Od2(BaseStep):  

    def __init__(
            self, 
            inference_config: Dict,
            output_path: Optional[str],
            step_name: str,
            dataset_type: str,
            dataset_path: str,
            run_dataset_split: str,
            test_dataset_split: str,
            if_select_lbpp: bool = False,
            data_group_tuple: tuple[int, int] = (162, 5),
            num_samples: int = 20,
            num_rounds: int = 5,
            topk: int = 3,
            save_analysis: bool = True,
            inference_max_retries: int = 5,
            select_best_node_method: str = "success_ratio-time_enabled_ns",
            save_tree_visualization: bool = True,
            save_tree_visualization_only_tree_exist: bool = True,
            **kwargs
        ):
        super().__init__(step_type="code_process_supervision_lbpp", step_name=step_name, description="Code Process Supervision Lbpp")
        self.logger = logging.getLogger(__name__)
        self.num_samples: int = num_samples
        self.num_rounds: int = num_rounds
        self.topk: int = topk
        self.inference_config = inference_config
        self.data_group_count: int = data_group_tuple[0]
        self.data_group_idx: int = data_group_tuple[1]
        self.dataset_type: str = dataset_type
        self.dataset_path: str = dataset_path
        self.if_select_lbpp: bool = if_select_lbpp
        self.run_dataset_split: str = run_dataset_split
        self.test_dataset_split: str = test_dataset_split
        self.output_path: Optional[str] = output_path
        self.step_name: str = step_name
        self.save_analysis: bool = save_analysis
        self.inference_max_retries: int = inference_max_retries
        self.select_best_node_keys: list[str] = select_best_node_method.split("-")
        self.save_tree_visualization: bool = save_tree_visualization
        self.save_tree_visualization_only_tree_exist: bool = save_tree_visualization_only_tree_exist

        try:
            self.select_node_keys_ascend_symbol_list = [SELECT_NODE_KEYS_ASCEND_SYMBOL[key] for key in self.select_best_node_keys]
            self.select_node_keys_ascend_symbol_list.append(SELECT_NODE_KEYS_ASCEND_SYMBOL["score"])
        except KeyError as e:
            self.logger.error(f"Invalid select_best_node_keys: {select_best_node_method}, {e}")
            raise e
        
        prompt_postprocessor_config = self.inference_config.get("prompt_postprocessor_config", None)
        if prompt_postprocessor_config and "tokenizer_name_or_path" in prompt_postprocessor_config:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(prompt_postprocessor_config["tokenizer_name_or_path"])
        else:
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Please provide tokenizer_name_or_path in prompt_postprocessor_config")
        return len(self.tokenizer.encode(text))

    def _calculate_lbpp_group(self):
        group_index = self.data_group_idx
        groups_count = self.data_group_count
        if group_index < 0 or group_index >= groups_count:
            raise ValueError(f"Invalid group index: {group_index}, should be in range [0, {groups_count - 1}]")
        if groups_count <= 0 or groups_count > len(self.lbpp_data):
            raise ValueError(f"Invalid groups count: {groups_count}, should be in range [1, {len(self.lbpp_data)}]")
        
        task_ids = [(int(task_id.split('/')[-1]), task_id) for task_id in self.lbpp_data.keys()]
        sorted_task_ids = sorted(task_ids, key=lambda x: x[0])
        
        base_length = len(sorted_task_ids) // groups_count
        remainder = len(sorted_task_ids) % groups_count

        start_idx = group_index * base_length + min(group_index, remainder)
        end_idx = start_idx + base_length + (1 if group_index < remainder else 0)

        self.group_start_idx = start_idx
        self.group_end_idx = end_idx

        return {task_id: self.lbpp_data[task_id] for _, task_id in sorted_task_ids[start_idx:end_idx]}

    def init_model(self, type: str, inference_kwargs: Dict, prompt_postprocessor_config: Dict = None, **kwargs):
        self.inference_function = load_inference_function(type)

        self.inference_kwargs = inference_kwargs
        self.prompt_postprocessor = None if type == "openai" else PromptPostprocessor(**prompt_postprocessor_config)
    
    def _load_cached_analysis(self, task_id: str, uuid: str, path: str) -> Optional[Dict]:
        safe_task_id = re.sub(r'[^\w\-_\. ]', '_', task_id)
        cache_file_path = os.path.join(path, "analysis", f"{safe_task_id}_{uuid}.json")
        
        if os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, 'r') as f:
                    cached_data = json.load(f)
                if is_valid_analysis(cached_data.get("quality_metrics", {})):
                    self.logger.debug(f"Cache hit for task_id {task_id}, uuid {uuid}")
                    return cached_data
                else:
                    self.logger.debug(f"Cache hit for task_id {task_id}, uuid {uuid}, but not valid")
                    return cached_data
            except Exception as e:
                pass
        
        return None

    def _save_analysis(self, task_id: str, uuid: str, analyzed_code: Dict, path: str):
        safe_task_id = re.sub(r'[^\w\-_\. ]', '_', task_id)
        
        analysis_dir = os.path.join(path, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        file_path = os.path.join(analysis_dir, f"{safe_task_id}_{uuid}.json")
        
        with open(file_path, 'w') as f:
            json.dump(analyzed_code, f, indent=2)

    def preprocess(self, context: Context):
        self.logger.info("Preprocessing Code Process Supervision Lbpp") 
        self.lbpp_data = get_lbpp(self.run_dataset_split, self.dataset_path, if_select_lbpp=self.if_select_lbpp)
        self.lbpp_test_data = get_lbpp(self.test_dataset_split, self.dataset_path, if_select_lbpp=self.if_select_lbpp)
        
        self.origin_instruction_dataset = InstructionDataset()
        self.origin_critic_instruction_dataset = InstructionDataset()

        self.instruction_datasets_by_round = [{"origin_instruction_dataset": self.origin_instruction_dataset, "origin_critic_instruction_dataset": self.origin_critic_instruction_dataset}]
        
        self.round_nodes = [[]]

        self.lbpp_data_group = self._calculate_lbpp_group()
        
        for task_id, task_data in self.lbpp_data_group.items():

            prompt = task_data['instruction']
            test_setup = task_data['test_setup'].replace("from code import", "from YourCode import") if task_data['test_setup'] is not None else ""
            test_list = task_data['test_list']
            
            base_prompt = f"{prompt}\n\nI will place the code you implemented in a file named `YourCode.py`. Then, I will use another file, `B`, to test your code. This file will include the following prefix \n{test_setup}\n to reference your code, and then `B` will run the test cases below. Please implement the function to satisfy these test cases:\n"
            for index, test in enumerate(test_list):
                base_prompt += f"# Test case {index+1}. {test}\n# ----\n"
            
            template_keys = list(prompt_templates.keys())
            if self.num_samples <= len(template_keys):
                selected_templates = template_keys[:self.num_samples]
            else:
                selected_templates = template_keys * (self.num_samples // len(template_keys)) + template_keys[:self.num_samples % len(template_keys)]
            
            for template_key in selected_templates:
                new_prompt = f"{base_prompt}\nDo not include these test cases in your code.\n{prompt_templates[template_key]}\n"

                inst = Instruction(
                    input=new_prompt, 
                    extra={
                        "task_id": task_id,
                        "original_prompt": task_data['instruction'],
                        "test_list": test_list
                    }
                )

                self.origin_instruction_dataset.instructions.append(inst)

        self.init_model(**self.inference_config)

        unique_instructions = {inst.uuid: inst for inst in self.origin_instruction_dataset.instructions}
        self.origin_instruction_dataset.instructions = list(unique_instructions.values())

    def generate_origin_node_from_lbpp(self):
        self.logger.info("Generating origin code from lbpp")
        assert len(self.origin_instruction_dataset.instructions) > 0, "Origin instruction dataset should not be empty"
        for inst in self.origin_instruction_dataset.instructions:
            msgs = [
                {"role": "system", "content": f"{PROGRAMMER_SYSTEM_PROMPT}"},
                {"role": "user", "content": PROGRAMMER_USER_PROMPT_FIRST_ROUND.format(base_prompt=inst.input)},
            ]
            if self.prompt_postprocessor:
                inst.prompt = self.prompt_postprocessor.get_full_prompt_from_conversation(msgs)
            else:
                inst.messages = msgs
                inst.prompt = inst.input
        
        current_output_path = os.path.join(self.output_path, f"{self.step_name}")
        current_output_path = os.path.join(current_output_path, f"lbpp_data_group_{self.data_group_idx}_of_{self.data_group_count}")
        current_output_path = os.path.join(current_output_path, f"round_0")
        current_inference_output_path = os.path.join(current_output_path, "origin_code")
        inference_kwargs = {**self.inference_kwargs, "output_path": current_inference_output_path}
        self.logger.info(f"Inferring origin code from lbpp, inference_kwargs: {inference_kwargs}\noutput_path: {current_inference_output_path}")
        try:
            self.inference_function(self.origin_instruction_dataset, **inference_kwargs)
        except Exception as e:
            self.logger.error(f"Failed to infer origin code from lbpp: {e}")
            raise Exception(f"Failed to infer origin code from lbpp: {e}")
        with jsonlines.open(os.path.join(current_inference_output_path, "all_responses.jsonl")) as f:
            for line in tqdm(f):
                uuid = line["request"]["uuid"]
                generated_text = line["response"]["generated_text"].strip()
                for inst in self.origin_instruction_dataset.instructions:
                    if inst.uuid == uuid:
                        inst.output = generated_text
                        break   

        self.logger.info("Analyzing and scoring origin codes")
        
        temp_dir = os.path.join(current_inference_output_path, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        for inst in tqdm(self.origin_instruction_dataset.instructions, desc="Analyzing and scoring origin codes"):
            cached_analysis = self._load_cached_analysis(inst.extra['task_id'], inst.uuid, current_inference_output_path)
            if cached_analysis:
                analysis = cached_analysis
                inst.extra['analysis'] = analysis
                continue
            else:
                try:
                    static_result = static_analyze_code(inst.output)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze code \n{inst.output}\nwith static analysis, error: {e}")
                    static_result = {}
                try:
                    if self.dataset_type == "lbpp":
                        dynamic_result = dynamic_analyze_code_lbpp(inst.output, inst.extra['task_id'], self.lbpp_data_group, temp_dir)
                    elif self.dataset_type == "lbpp_alter":
                        dynamic_result = dynamic_analyze_code_lbpp_alter(inst.output, inst.extra['task_id'], self.lbpp_data_group, temp_dir)
                    else:
                        raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                except Exception as e:
                    self.logger.warning(f"Failed to analyze code \n{inst.output}\nwith dynamic analysis, error: {e}")
                    dynamic_result = {}
                analysis = {
                    "static": static_result,
                    "dynamic": dynamic_result
                }
                
            if analysis and is_valid_analysis(analysis):
                score = calculate_new_preference_score(analysis)
                analysis_with_score = {
                    "task_id": inst.extra['task_id'],
                    "uuid": inst.uuid,
                    "code": analysis["static"]["sanitized_code"].strip(),
                    "prompt": inst.prompt,
                    "quality_metrics": analysis,
                    "model_output": inst.output,
                    "score": score
                }
                inst.extra['analysis'] = analysis_with_score
                if self.save_analysis:
                    self._save_analysis(inst.extra['task_id'], inst.uuid, analysis_with_score, current_inference_output_path)
            else:
                failed_analysis = {
                    "task_id": inst.extra['task_id'],
                    "uuid": inst.uuid,
                    "code": None,
                    "prompt": inst.prompt,
                    "quality_metrics": analysis,
                    "model_output": inst.output,
                    "analysis_failed": True,
                    "score": 0
                }
                inst.extra['analysis'] = failed_analysis  
                if self.save_analysis:
                    self._save_analysis(inst.extra['task_id'], inst.uuid, failed_analysis, current_inference_output_path)
        
        for inst in self.origin_instruction_dataset.instructions:
            if len(inst.extra['analysis']['quality_metrics'].get('static', {})) == 0:
                self.logger.warning(f"Analysis.static is None, remove instruction {inst.uuid}, instruction output: {inst.output}")
                self.origin_instruction_dataset.instructions.remove(inst)
                
            
            





            



        for origin_inst in self.origin_instruction_dataset.instructions:
            origin_inst.extra['criticism'] = ""


        self.logger.info("Building first round nodes")
        for inst in self.origin_instruction_dataset.instructions:   
            if not inst.extra['analysis'].get('analysis_failed', False):
                code = inst.extra['analysis']['code']
                simple_analysis = inst.extra['analysis']['quality_metrics']
                if simple_analysis.get('static', None):
                    if simple_analysis['static'].get('sanitized_code', None):
                        del simple_analysis['static']['sanitized_code'] 
                simple_analysis['final_score'] = inst.extra['analysis']['score']
                parse_simple_analysis = json.dumps(simple_analysis, ensure_ascii=False)
                simple_analysis_str = json.dumps(simple_analysis, ensure_ascii=False)
                score = inst.extra['analysis']['score']
            else:
                code = inst.output
                failed_analysis = inst.extra['analysis']['quality_metrics']
                if failed_analysis.get('static', None):
                    if failed_analysis['static'].get('sanitized_code', None):
                        del failed_analysis['static']['sanitized_code']
                simple_analysis_str = json.dumps(failed_analysis, ensure_ascii=False)
                parse_simple_analysis = "Analysis failed! Sorry for that. I cannot provide any analysis for this code." + json.dumps(failed_analysis, ensure_ascii=False)
                simple_analysis_str = f"Analysis failed! Sorry for that. I cannot provide any analysis for this code.\n{simple_analysis_str}"
                score = 0.0

            criticism = inst.extra['criticism']

            process_score = 0.0

            node = CodeNode(code, inst.output, simple_analysis_str, criticism, score, current_process_score=process_score, parent=None, task_id=inst.extra['task_id'])

            if "<|good|>" in criticism.lower():
                node.exploration_early_over = True

            self.round_nodes[0].append(node)
        task_id_to_nodes = defaultdict(list)
        for node in self.round_nodes[0]:
            if node.exploration_early_over:
                continue
            analysis = CodeNode.parse_analysis_str(node.current_analysis)
            try:
                node.current_success_rate = analysis['dynamic']['success_rate']
            except:
                node.current_success_rate = 0.0
            task_id_to_nodes[node.task_id].append(node)
        for task_id, nodes in task_id_to_nodes.items():
            nodes.sort(key=lambda x: (x.current_success_rate, x.current_score), reverse=True)
            for node in nodes[self.topk:]:
                node.exploration_truncated = True

        for node in self.round_nodes[0]:
            node.calculate_uuid()

        self.logger.info(f"Saving nodes of round {0}")
        save_path = os.path.join(current_output_path, "nodes")
        for node in self.round_nodes[0]:
            node.save(save_path)
                
    def single_round_search(self, round_idx: int):
        self.logger.info(f"Starting single round search of round {round_idx}")
        self.instruction_datasets_by_round.append({"critic_instructions": InstructionDataset(), "programmer_instructions": InstructionDataset()})
        self.round_nodes.append([])
        assert len(self.instruction_datasets_by_round) == len(self.round_nodes) == round_idx + 1, "The length of instruction_datasets_by_round and round_nodes should be equal to round_idx + 1"
        current_critic_instruction_dataset = self.instruction_datasets_by_round[round_idx]["critic_instructions"]
        current_programmer_instruction_dataset = self.instruction_datasets_by_round[round_idx]["programmer_instructions"]
        current_node_list = self.round_nodes[round_idx]
        last_node_list = self.round_nodes[round_idx - 1]
        current_output_path = os.path.join(self.output_path, f"{self.step_name}")
        current_output_path = os.path.join(current_output_path, f"lbpp_data_group_{self.data_group_idx}_of_{self.data_group_count}")
        current_output_path = os.path.join(current_output_path, f"round_{round_idx}")

        all_over = True
        for node in last_node_list:
            if node.exploration_truncated or node.exploration_early_over:
                continue
            else:
                all_over = False
        if all_over:
            self.logger.info("All nodes are early over. Early stopping.")
            return {'all_over': True, 'round_idx': round_idx}
        
        template_keys = list(prompt_templates.keys())
        if self.num_samples <= len(template_keys):
            selected_templates = template_keys[:self.num_samples]
        else:
            selected_templates = template_keys * (self.num_samples // len(template_keys)) + template_keys[:self.num_samples % len(template_keys)]

        self.logger.info(f"Generating programmer instructions of round {round_idx}")
        for node in tqdm(last_node_list, desc=f"Generating programmer instructions of round {round_idx}"):
            if node.exploration_truncated or node.exploration_early_over:
                continue
            task_id = node.task_id
            task_data = self.lbpp_data_group.get(task_id)
            if not task_data:
                self.logger.error(f"Task data not found for task_id {task_id}, skip this node, node_uuid: {node.uuid}")
                continue
            prompt = task_data['instruction']
            test_setup = task_data['test_setup'].replace("from code import", "from YourCode import") if task_data['test_setup'] is not None else ""
            test_list = task_data['test_list']
            
            base_prompt = f"{prompt}\n\nI will place the code you implemented in a file named `YourCode.py`. Then, I will use another file, `B`, to test your code. This file will include the following prefix \n{test_setup}\n to reference your code, and then `B` will run the test cases below. Please implement the function to satisfy these test cases:\n"
            for index, test in enumerate(test_list):
                base_prompt += f"# Test case {index+1}. {test}\n# ----\n"
            msgs = [
                {"role": "system", "content": PROGRAMMER_SYSTEM_PROMPT},
                {"role": "user", "content": PROGRAMMER_USER_PROMPT_FIRST_ROUND.format(base_prompt=base_prompt)}
            ]

            for programmer_output, analysis, critique in zip(node.history_programmer_outputs, node.history_analyses, node.history_criticisms):
                parse_analysis = json.dumps(CodeNode.parse_analysis_str(analysis), ensure_ascii=False)
                msgs.append({"role": "assistant", "content": f"\n{programmer_output}"})
                msgs.append({"role": "user", "content": PROGRAMMER_USER_PROMPT_LATER_ROUNDS.format(
                    analysis=parse_analysis,
                    criticism=critique
                )})
            
            programmer_output = node.current_programmer_output
            analysis = node.current_analysis
            criticism = node.current_criticism

            parse_analysis = json.dumps(CodeNode.parse_analysis_str(analysis), ensure_ascii=False)
            msgs.append({"role": "assistant", "content": f"\n{programmer_output}"})
            new_msg_prefix = PROGRAMMER_USER_PROMPT_LATER_ROUNDS.format(
                analysis=parse_analysis,
                criticism=criticism
            )
            for template_key in selected_templates:
                new_msg_with_prompt_template = {"role": "user", "content": f"{new_msg_prefix}\n{prompt_templates[template_key]}"}
                final_msgs = msgs + [new_msg_with_prompt_template]

                final_msgs_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in final_msgs])

                programmer_inst = Instruction(
                    input=final_msgs_str,
                    extra={
                        "task_id": task_id,
                        "msgs": final_msgs,
                        "node_uuid": node.uuid,
                    }
                )

                if self.prompt_postprocessor:
                    programmer_inst.prompt = self.prompt_postprocessor.get_full_prompt_from_conversation(final_msgs)
                else:
                    programmer_inst.messages = final_msgs
                    programmer_inst.prompt = final_msgs_str

                programmer_inst.extra['derived_from_node_uuid'] = node.uuid
                
                current_programmer_instruction_dataset.instructions.append(programmer_inst)

        current_programmer_output_path = os.path.join(current_output_path, "programmer_instructions")
        inference_kwargs = {**self.inference_kwargs, "output_path": current_programmer_output_path}
        self.logger.info(f"Inferring programmer instructions of round {round_idx}, inference_kwargs: {inference_kwargs}\noutput_path: {current_programmer_output_path}")
        try:    
            self.inference_function(current_programmer_instruction_dataset, **inference_kwargs)
        except Exception as e:
            self.logger.error(f"Failed to infer programmer instructions of round {round_idx}, error: {e}")
            raise Exception(f"Failed to infer programmer instructions of round {round_idx}, error: {e}")

        with jsonlines.open(os.path.join(current_programmer_output_path, "all_responses.jsonl")) as f:
            for line in f:
                uuid = line["request"]["uuid"]
                generated_text = line["response"]["generated_text"].strip()
                for inst in current_programmer_instruction_dataset.instructions:
                    if inst.uuid == uuid:
                        inst.output = generated_text
                        break
        
        self.logger.info(f"Analyzing and scoring generated codes of round {round_idx}")

        temp_dir = os.path.join(current_programmer_output_path, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        for inst in tqdm(current_programmer_instruction_dataset.instructions, desc=f"Analyzing and scoring generated codes of round {round_idx}"):
            cached_analysis = self._load_cached_analysis(inst.extra['task_id'], inst.uuid, current_programmer_output_path)
            if cached_analysis:
                analysis = cached_analysis
                inst.extra['analysis'] = analysis
                continue
            else:
                try:
                    static_result = static_analyze_code(inst.output)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze code \n{inst.output}\nwith static analysis, error: {e}")
                    static_result = {}
                try:
                    if self.dataset_type == "lbpp":
                        dynamic_result = dynamic_analyze_code_lbpp(inst.output, inst.extra['task_id'], self.lbpp_data_group, temp_dir)
                    elif self.dataset_type == "lbpp_alter":
                        dynamic_result = dynamic_analyze_code_lbpp_alter(inst.output, inst.extra['task_id'], self.lbpp_data_group, temp_dir)
                    else:
                        raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                except Exception as e:
                    self.logger.warning(f"Failed to analyze code \n{inst.output}\nwith dynamic analysis, error: {e}")
                    dynamic_result = {}
                analysis = {
                    "static": static_result,
                    "dynamic": dynamic_result
                }
            
            if analysis and is_valid_analysis(analysis):
                score = calculate_new_preference_score(analysis)
                analysis_with_score = {
                    "task_id": inst.extra['task_id'],
                    "uuid": inst.uuid,
                    "code": analysis["static"]["sanitized_code"].strip(),
                    "prompt": inst.prompt,
                    "quality_metrics": analysis,
                    "model_output": inst.output,
                    "score": score
                }
                inst.extra['analysis'] = analysis_with_score
                if self.save_analysis:
                    self._save_analysis(inst.extra['task_id'], inst.uuid, analysis_with_score, current_programmer_output_path)
            else:
                failed_analysis = {
                    "task_id": inst.extra['task_id'],
                    "uuid": inst.uuid,
                    "code": None,
                    "prompt": inst.prompt,
                    "quality_metrics": analysis,
                    "model_output": inst.output,
                    "analysis_failed": True,
                    "score": 0
                }
                inst.extra['analysis'] = failed_analysis
                if self.save_analysis:
                    self._save_analysis(inst.extra['task_id'], inst.uuid, failed_analysis, current_programmer_output_path)

        for inst in current_programmer_instruction_dataset.instructions:
            if len(inst.extra['analysis']['quality_metrics'].get('static', {})) == 0:
                current_programmer_instruction_dataset.instructions.remove(inst)
                self.logger.warning(f"Analysis.static is None, remove instruction {inst.uuid}, instruction output: {inst.output}")


            

                

            
            



            
        

        
            
        for programmer_inst in current_programmer_instruction_dataset.instructions:
            programmer_inst.extra['criticism'] = ""
        
        self.logger.info("Building new nodes")
        for inst in current_programmer_instruction_dataset.instructions:
            for node in last_node_list:
                if inst.extra['node_uuid'] == node.uuid:
                    parent_node = node
                    break
            
            if not inst.extra['analysis'].get('analysis_failed', False):
                code = inst.extra['analysis']['code']
                simple_analysis = inst.extra['analysis']['quality_metrics']
                if simple_analysis.get('static', None):
                    if simple_analysis['static'].get('sanitized_code', None):
                        del simple_analysis['static']['sanitized_code']
                simple_analysis['final_score'] = inst.extra['analysis']['score']
                simple_analysis_str = json.dumps(simple_analysis, ensure_ascii=False)
                score = inst.extra['analysis']['score']
            else:
                code = inst.output
                failed_analysis = inst.extra['analysis']['quality_metrics']
                if failed_analysis.get('static', None):
                    if failed_analysis['static'].get('sanitized_code', None):
                        del failed_analysis['static']['sanitized_code']
                simple_analysis_str = json.dumps(failed_analysis, ensure_ascii=False)
                simple_analysis_str = f"Analysis failed! Sorry for that. I cannot provide any analysis for this code.\n{simple_analysis_str}"
                score = 0.0
            
            criticism = inst.extra['criticism']
            
            process_score = 0.0
                

            node = CodeNode(code, inst.output, simple_analysis_str, criticism, score, current_process_score=process_score, parent=parent_node, task_id=inst.extra['task_id'])
            
            if "<|good|>" in criticism.lower():
                node.exploration_early_over = True

            

            current_node_list.append(node)

        task_id_to_nodes = defaultdict(list)
        for node in current_node_list:
            if node.exploration_early_over:
                continue
            try:
                node.current_success_rate = node.current_analysis['dynamic']['success_rate']
            except:
                node.current_success_rate = 0.0
            task_id_to_nodes[node.task_id].append(node)
        for task_id, nodes in task_id_to_nodes.items():
            nodes.sort(key=lambda x: (x.current_success_rate, x.current_score), reverse=True)
            for node in nodes[self.topk:]:
                node.exploration_truncated = True

        if round_idx == self.num_rounds:
            for node in current_node_list:
                if not node.exploration_early_over and not node.exploration_truncated:
                    node.exploration_over = True

        for node in current_node_list:
            node.calculate_uuid()

        self.logger.info(f"Saving nodes of round {round_idx}")
        save_path = os.path.join(current_output_path, "nodes")
        for node in current_node_list:
            node.save(save_path)

        return {'num_nodes': len(current_node_list), 'current_node_list': current_node_list}

    def run(self, context: Context):
        if (self.step_type, self.step_name) not in context.results: 
            context.results[(self.step_type, self.step_name)] = {}
        context.results[(self.step_type, self.step_name)]['all_nodes_early_over'] = False
        self.generate_origin_node_from_lbpp()
        for round_idx in range(1, self.num_rounds + 1):
            result = self.single_round_search(round_idx)
            if result.get('all_over', False):
                self.logger.info(f"All nodes are early over at round {round_idx}, stop the search")
                context.results[(self.step_type, self.step_name)]['all_nodes_early_over'] = True
                break
        
    def postprocess(self, context: Context):
        self.logger.info("Saving early over nodes and over topk nodes")
        save_path = os.path.join(self.output_path, f"{self.step_name}")
        save_path = os.path.join(save_path, f"lbpp_data_group_{self.data_group_idx}_of_{self.data_group_count}")
        save_early_over_path = os.path.join(save_path, f"early_over_nodes")
        save_over_topk_path = os.path.join(save_path, f"over_top{self.topk}_nodes")
        early_over_nodes = []
        over_topk_nodes = []
        for nodes in self.round_nodes:
            for node in nodes:
                if node.exploration_early_over:
                    node.save(save_early_over_path)
                    early_over_nodes.append(node)
        if context.results[(self.step_type, self.step_name)].get('all_nodes_early_over'):
            self.logger.info("All nodes are early over, skip saving over topk nodes")
        else:
            task_id_to_nodes = defaultdict(list)
            for node in self.round_nodes[-1]:
                if node.exploration_over:
                    task_id_to_nodes[node.task_id].append(node)
            for task_id, nodes in task_id_to_nodes.items():
                for node in nodes:
                    try:
                        node.current_success_rate = node.current_analysis['dynamic']['success_rate']
                    except:
                        node.current_success_rate = 0.0
                nodes.sort(key=lambda x: (x.current_success_rate, x.current_score), reverse=True)
                for node in nodes[:min(self.topk, len(nodes))]:
                    node.save(save_over_topk_path)
                    over_topk_nodes.append(node)
        
        self.logger.info(f"Saved {len(early_over_nodes)} early over nodes and {len(over_topk_nodes)} over topk nodes")

        self.logger.info("Calculating the best node for each task")
        
        
        task_id_2_nodes = {}
        info_dict = {
            "score": 0,
            "process_score": 0, 
            "success_ratio": 0.0, 
            "time_enabled_ns": float('inf'), 
            "instruction_count": float('inf'), 
            "branch_misses": float('inf'), 
            "page_faults": float('inf')
        }
        node_uuid_2_info_dict = {}
        
        node_uuid_2_select_info = {}
        all_best_nodes = []
        for nodes in self.round_nodes:
            for node in nodes:
                if node.task_id not in task_id_2_nodes:
                    task_id_2_nodes[node.task_id] = []
                task_id_2_nodes[node.task_id].append(node)
                node_uuid_2_info_dict[node.uuid] = info_dict.copy()
                node_uuid_2_select_info[node.uuid] = []
                node_uuid_2_info_dict[node.uuid]['process_score'] = node.current_process_score
                node_uuid_2_info_dict[node.uuid]['score'] = node.current_score
                current_analysis_str = node.current_analysis
                current_analysis = CodeNode.parse_analysis_str(current_analysis_str)
                try:
                    node_uuid_2_info_dict[node.uuid]['time_enabled_ns'] = current_analysis['dynamic']['average_metrics']['time_enabled_ns']
                except Exception as e:
                    node_uuid_2_info_dict[node.uuid]['time_enabled_ns'] = float('inf')
                try:
                    node_uuid_2_info_dict[node.uuid]['success_ratio'] = current_analysis['dynamic']['success_rate']
                except Exception as e:
                    node_uuid_2_info_dict[node.uuid]['success_ratio'] = 0.0
                try:
                    node_uuid_2_info_dict[node.uuid]['instruction_count'] = current_analysis['dynamic']['average_metrics']['instruction_count']
                except Exception as e:
                    node_uuid_2_info_dict[node.uuid]['instruction_count'] = float('inf')
                try:
                    node_uuid_2_info_dict[node.uuid]['branch_misses'] = current_analysis['dynamic']['average_metrics']['branch_misses']
                except Exception as e:
                    node_uuid_2_info_dict[node.uuid]['branch_misses'] = float('inf')
                try:
                    node_uuid_2_info_dict[node.uuid]['page_faults'] = current_analysis['dynamic']['average_metrics']['page_faults']
                except Exception as e:
                    node_uuid_2_info_dict[node.uuid]['page_faults'] = float('inf')
        
        for search_key in self.select_best_node_keys:
            for node_uuid, node_select_info_list in node_uuid_2_select_info.items():
                node_select_info_list.append(node_uuid_2_info_dict[node_uuid][search_key])
        for node_uuid, node_select_info_list in node_uuid_2_select_info.items():
            node_select_info_list.append(node_uuid_2_info_dict[node_uuid]['score'])


        task_id_2_node_infoes = {}
        task_id_2_best_node = {}
        for task_id, nodes in task_id_2_nodes.items():
            node_infoes_list = [(node_uuid_2_select_info[node.uuid], node) for node in nodes]
            node_infoes_list.sort(key=lambda x: tuple(direction * val for direction, val in zip(self.select_node_keys_ascend_symbol_list, x[0])))
            task_id_2_node_infoes[task_id] = node_infoes_list
            task_id_2_best_node[task_id] = task_id_2_node_infoes[task_id][0][1]
            all_best_nodes.append(task_id_2_best_node[task_id])

        select_method_str = "-".join(self.select_best_node_keys)
        save_best_nodes_path = os.path.join(save_path, f"best_nodes-{select_method_str}")
        os.makedirs(save_best_nodes_path, exist_ok=True)
        self.logger.info(f"Found {len(all_best_nodes)} best nodes for all tasks, start analyzing them with test data, save the results to {save_best_nodes_path}")
        temp_dir = os.path.join(save_best_nodes_path, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        for node in tqdm(all_best_nodes, desc="Analyzing best nodes with test data"):
            if self.dataset_type == "lbpp":
                try:
                    dynamic_analysis = dynamic_analyze_code_lbpp(node.current_code, node.task_id, self.lbpp_test_data, temp_dir)
                except Exception as e:
                    self.logger.error(f"Failed to analyze node {node.uuid} with test data. task_id: {node.task_id}, error: {e}")
                    dynamic_analysis = {}
            elif self.dataset_type == "lbpp_alter":
                try:
                    dynamic_analysis = dynamic_analyze_code_lbpp_alter(node.current_code, node.task_id, self.lbpp_test_data, temp_dir)
                except Exception as e:
                    self.logger.error(f"Failed to analyze node {node.uuid} with test data. task_id: {node.task_id}, error: {e}")
                    dynamic_analysis = {}
            else:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")
            node.dynamic_analysis_on_test_data = json.dumps(dynamic_analysis, ensure_ascii=False)
            node.save(save_best_nodes_path)


        for nodes in self.round_nodes:
            for node in nodes:
                true_count = sum([node.exploration_early_over, node.exploration_over, node.exploration_truncated])
                if true_count > 1:
                    self.logger.error(f"Node {node.uuid} has more than one of exploration_early_over, exploration_over, exploration_truncated, this is invalid")
        if self.save_tree_visualization:
            self.logger.info("Saving root nodes visualization")
            save_root_nodes_visualization_path = os.path.join(save_path, f"root_nodes_visualization")
            for node in self.round_nodes[0]:
                if self.save_tree_visualization_only_tree_exist and node.children:
                    node.export_tree_visualization(save_root_nodes_visualization_path)
                elif not self.save_tree_visualization_only_tree_exist:
                    node.export_tree_visualization(save_root_nodes_visualization_path)
            self.logger.info("Root nodes visualization saved")

if __name__ == "__main__":
    pass
