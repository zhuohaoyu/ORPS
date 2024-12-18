import argparse
from orps.lbpp_utils import get_lbpp, static_analyze_code, dynamic_analyze_code_lbpp, dynamic_analyze_code_lbpp_alter
from tqdm import tqdm
import os, json

TMP_FILE_PATH = "./tmp"

def calculate_and_save_dataset_standard_runtime_metrics(dataset_path: str, dataset_type: str, save_file_path: str):
    ds = get_lbpp(split="test", dataset_path=dataset_path)
    
    if dataset_type == "lbpp":
        dynamic_analyze_code = dynamic_analyze_code_lbpp
    elif dataset_type == "lbpp_alter":
        dynamic_analyze_code = dynamic_analyze_code_lbpp_alter
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
    
    task_id_2_metrics = {}
    metrics_dict = {'code_length': None, 'ast_node_count': None, 'mean_cyclomatic_complexity': None, 'mean_cognitive_complexity': None, 'success_rate': None, 'mean_time_enabled_ns': None, 'mean_instruction_count': None, 'mean_branch_misses': None, 'mean_page_faults': None}

    for task_id , task in tqdm(ds.items(), desc=f"Calculating runtime metrics for data_path at {dataset_path}"):
        try:
            code = task["completion"]
            print(code)
            static_metrics = static_analyze_code(code)
            dynamic_metrics = dynamic_analyze_code(code, task_id, ds, path_for_tmp_files=TMP_FILE_PATH)
            task_id_2_metrics[task_id] = metrics_dict.copy()
            code_length = static_metrics['code_length']
            ast_node_count = static_metrics['ast_node_count']
            cyclomatic_complexity = static_metrics['cyclomatic_complexity']
            mean_cyclomatic_complexity = sum(cyclomatic_complexity.values()) / len(cyclomatic_complexity)
            cognitive_complexity = static_metrics['cognitive_complexity']
            mean_cognitive_complexity = sum(cognitive_complexity.values()) / len(cognitive_complexity)
            success_rate = dynamic_metrics['success_rate']
            mean_time_enabled_ns = dynamic_metrics['average_metrics']['time_enabled_ns']
            mean_instruction_count = dynamic_metrics['average_metrics']['instruction_count']
            mean_branch_misses = dynamic_metrics['average_metrics']['branch_misses']
            mean_page_faults = dynamic_metrics['average_metrics']['page_faults']
            task_id_2_metrics[task_id]['code_length'] = code_length
            task_id_2_metrics[task_id]['ast_node_count'] = ast_node_count
            task_id_2_metrics[task_id]['mean_cyclomatic_complexity'] = mean_cyclomatic_complexity
            task_id_2_metrics[task_id]['mean_cognitive_complexity'] = mean_cognitive_complexity
            task_id_2_metrics[task_id]['success_rate'] = success_rate
            task_id_2_metrics[task_id]['mean_time_enabled_ns'] = mean_time_enabled_ns
            task_id_2_metrics[task_id]['mean_instruction_count'] = mean_instruction_count
            task_id_2_metrics[task_id]['mean_branch_misses'] = mean_branch_misses
            task_id_2_metrics[task_id]['mean_page_faults'] = mean_page_faults
        except Exception as e:
            import pdb
            print(task_id)
            print(dynamic_metrics)
            continue
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, 'w') as f:
        json.dump(task_id_2_metrics, f, indent=4, ensure_ascii=False)
    
    return task_id_2_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate and save dataset standard runtime metrics')
    parser.add_argument('-d', '--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('-t', '--dataset_type', type=str, required=True, help='Type of the dataset')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output metrics')
    args = parser.parse_args()
    calculate_and_save_dataset_standard_runtime_metrics(args.dataset_path, args.dataset_type, args.output_path)

