from collections import OrderedDict, defaultdict
from orps.lbpp_utils import CodeNode, static_analyze_code
from typing import Dict, List
import os, json
import argparse
import numpy as np

def load_best_nodes(path, select_method) -> Dict[str, List[CodeNode]]:
    best_node_dir = f"best_nodes-{select_method}"
    best_nodes_path = os.path.join(path, best_node_dir)
    task_id_2_best_nodes = {}
    for node_uuid in os.listdir(best_nodes_path):
        node_path = os.path.join(best_nodes_path, node_uuid)
        best_nodes = []
        if not node_path.endswith(".json"):
            continue
        node_attributes = json.load(open(node_path, "r"))
        node = CodeNode(node_attributes["current_code"], node_attributes["current_programmer_output"], node_attributes["current_analysis"], node_attributes["current_criticism"], node_attributes["current_score"], node_attributes["current_process_score"], parent=None, task_id=node_attributes["task_id"])
        node.dynamic_analysis_on_test_data = node_attributes["dynamic_analysis_on_test_data"]
        node.history_codes = node_attributes["history_codes"]
        node.history_programmer_outputs = node_attributes["history_programmer_outputs"]
        node.history_analyses = node_attributes["history_analyses"]
        node.history_criticisms = node_attributes["history_criticisms"]
        node.history_scores = node_attributes["history_scores"]
        node.calculate_uuid()
        best_nodes.append(node)
        assert len(best_nodes) > 0, f"Task {node_uuid} has no best node"
        assert len(best_nodes) == 1, f"Task {node_uuid} has more than one best node"
        task_id_2_best_nodes[best_nodes[0].task_id] = best_nodes[0]
    return task_id_2_best_nodes

def merge_data_split(group_path, select_method):
    task_id_2_best_nodes = {}
    for data_split_path in os.listdir(group_path):
        task_id_2_best_nodes.update(load_best_nodes(os.path.join(group_path, data_split_path), select_method))
    return task_id_2_best_nodes

def calculate_success_rate(task_id_2_best_node: Dict[str, CodeNode]):
    success_ratio_dict = {}
    for task_id, best_node in task_id_2_best_node.items():
        dynamic_analysis_on_test_data = json.loads(best_node.dynamic_analysis_on_test_data)
        try:
            success_rate = dynamic_analysis_on_test_data['success_rate']
        except KeyError:
            success_rate = 0
        success_ratio_dict[task_id] = success_rate

    sorted_success_ratio_dict = OrderedDict(sorted(success_ratio_dict.items(), key=lambda x: x[0]))


    return sum(sorted_success_ratio_dict.values()) / len(sorted_success_ratio_dict)

def calculate_accuracy(task_id_2_best_node: Dict[str, CodeNode]):
    task_id_2_pass = {}
    for task_id, best_node in task_id_2_best_node.items():
        dynamic_analysis_on_test_data = json.loads(best_node.dynamic_analysis_on_test_data)
        try:
            success_rate = dynamic_analysis_on_test_data['success_rate']
        except KeyError:
            success_rate = 0

        if success_rate == 1.0:
            task_id_2_pass[task_id] = True
        else:
            task_id_2_pass[task_id] = False


    return sum(task_id_2_pass.values()) / len(task_id_2_pass)

def normalize_with_outlier_filter(value: float, standard: float, task_id: str, metric_name: str, threshold: float = 10) -> float:
    """Normalize a value against a standard with outlier filtering.
    
    Args:
        value: The value to normalize
        standard: The standard value to compare against
        task_id: Task ID for logging
        metric_name: Name of t
        he metric for logging
        threshold: Maximum allowed ratio (default 10.0)
    
    Returns:
        float: Normalized value, capped at threshold if exceeds it
    """
    if value < 0:
        return 1
    standard_adjusted = standard + (1 if standard == 0 else 0)
    ratio = value / standard_adjusted
    
    if ratio > threshold:
        print(f"Warning: Outlier detected in {task_id} for {metric_name}: {ratio:.2f}x (capped at {threshold:.1f}x)")
        return threshold
    return ratio

def calculate_average_metrics(task_id_2_best_node: Dict[str, CodeNode], standard_metrics_path: str, success_rate: float=1.0):
    standard_metrics = json.load(open(standard_metrics_path, "r"))
    task_id_2_standard_metrics = {}
    for task_id, metrics in standard_metrics.items():
        task_id_2_standard_metrics[task_id] = metrics

    metrics_dict = {
        'time_enabled_ns': defaultdict(float),
        'instruction_count': defaultdict(float),
        'branch_misses': defaultdict(float),
        'page_faults': defaultdict(float),
        'code_length': defaultdict(float),
        'ast_node_count': defaultdict(float),
        'cyclomatic_complexity': defaultdict(float),
        'cognitive_complexity': defaultdict(float),
    }
    
    valid_tasks = set()
    
    for task_id, best_node in task_id_2_best_node.items():
        is_valid = True
        
        best_node_info = json.loads(best_node.dynamic_analysis_on_test_data)
        if len(best_node_info) == 0:
            print(f"Task {task_id} has no dynamic analysis on test data")
            is_valid = False
            continue
            
        best_node_current_analysis = CodeNode.parse_analysis_str(best_node.current_analysis)
        best_node_static_analysis = best_node_current_analysis.get('static', {})
        try:
            best_node_static_analysis = static_analyze_code(best_node.current_code)
        except Exception as e:
            best_node_static_analysis = {}
        if len(best_node_static_analysis) == 0:
            print(f"Task {task_id} has no static analysis")
            is_valid = False
            continue

        standard_metrics = task_id_2_standard_metrics[task_id]
        
        if len(best_node_info.get('average_metrics', {})) == 0:
            print(f"Task {task_id} has no average metrics in dynamic analysis")
            is_valid = False
            continue
            
        required_fields = ['code_length', 'ast_node_count', 'cyclomatic_complexity', 'cognitive_complexity']
        if not all(field in best_node_static_analysis for field in required_fields):
            print(f"Task {task_id} missing some static analysis fields")
            is_valid = False
            continue

        try:
            metrics_dict['time_enabled_ns'][task_id] = normalize_with_outlier_filter(
                best_node_info['average_metrics']['time_enabled_ns'],
                standard_metrics['mean_time_enabled_ns'],
                task_id,
                'time_enabled_ns'
            )
            metrics_dict['instruction_count'][task_id] = normalize_with_outlier_filter(
                best_node_info['average_metrics']['instruction_count'],
                standard_metrics['mean_instruction_count'],
                task_id,
                'instruction_count'
            )
            metrics_dict['branch_misses'][task_id] = normalize_with_outlier_filter(
                best_node_info['average_metrics']['branch_misses'],
                standard_metrics['mean_branch_misses'],
                task_id,
                'branch_misses'
            )
            metrics_dict['page_faults'][task_id] = normalize_with_outlier_filter(
                best_node_info['average_metrics']['page_faults'],
                standard_metrics['mean_page_faults'],
                task_id,
                'page_faults'
            )
        except (KeyError, TypeError, ZeroDivisionError) as e:
            print(f"Task {task_id} has invalid dynamic metrics: {str(e)}")
            is_valid = False
            continue

        try:
            metrics_dict['code_length'][task_id] = normalize_with_outlier_filter(
                best_node_static_analysis['code_length'],
                standard_metrics['code_length'],
                task_id,
                'code_length'
            )
            metrics_dict['ast_node_count'][task_id] = normalize_with_outlier_filter(
                best_node_static_analysis['ast_node_count'],
                standard_metrics['ast_node_count'],
                task_id,
                'ast_node_count'
            )
            
            try:
                cyclo_complexity = (
                    sum(best_node_static_analysis['cyclomatic_complexity'].values()) / 
                    len(best_node_static_analysis['cyclomatic_complexity'])
                )
                metrics_dict['cyclomatic_complexity'][task_id] = normalize_with_outlier_filter(
                    cyclo_complexity,
                    standard_metrics['mean_cyclomatic_complexity'],
                    task_id,
                    'cyclomatic_complexity'
                )
            except (KeyError, ZeroDivisionError, AttributeError):
                print(f"Task {task_id} has invalid cyclomatic complexity format")
                is_valid = False
                continue
            
            try:
                metrics_dict['cognitive_complexity'][task_id] = normalize_with_outlier_filter(
                    best_node_static_analysis['cognitive_complexity']['mean'],
                    standard_metrics['mean_cognitive_complexity'],
                    task_id,
                    'cognitive_complexity'
                )
            except (KeyError, TypeError):
                print(f"Task {task_id} has invalid cognitive complexity format")
                is_valid = False
                continue

        except (KeyError, TypeError, ZeroDivisionError, AttributeError) as e:
            print(f"Task {task_id} has invalid static metrics: {str(e)}")
            is_valid = False
            continue

        if is_valid:
            valid_tasks.add(task_id)

    if not valid_tasks:
        print("Warning: No valid tasks found for computing metrics!")
        return tuple([-1] * 11)

    static_metrics = {
        'code_length': False,
        'ast_node_count': False,
        'cyclomatic_complexity': False,
        'cognitive_complexity': False
    }
    dynamic_metrics = {
        'time_enabled_ns': False,
        'instruction_count': False,
        'branch_misses': False,
        'page_faults': False
    }

    averages = {
        metric: np.average(list(values.values())) * (1.0 / success_rate) if metric in dynamic_metrics else np.average(list(values.values())) * (1.0 / success_rate)
        for metric, values in metrics_dict.items()
    }

    static_metrics = {
        'code_length': False,
        'ast_node_count': False,
        'cyclomatic_complexity': False,
        'cognitive_complexity': False
    }
    dynamic_metrics = {
        'time_enabled_ns': False,
        'instruction_count': False,
        'branch_misses': False,
        'page_faults': False
    }
    
    static_improvements = []
    for metric, higher_is_better in static_metrics.items():
        ratio = averages[metric]
        improvement = (ratio - 1) if higher_is_better else (1 - ratio)
        static_improvements.append(improvement)
    
    dynamic_improvements = []
    for metric, higher_is_better in dynamic_metrics.items():
        ratio = averages[metric]
        improvement = (ratio - 1) if higher_is_better else (1 - ratio)
        dynamic_improvements.append(improvement)
    
    avg_static_improvement = np.average(static_improvements)
    avg_dynamic_improvement = np.average(dynamic_improvements)
    
    non_empty_ratio = len(valid_tasks) / len(task_id_2_best_node)
    
    return (
        averages['time_enabled_ns'],
        averages['instruction_count'],
        averages['branch_misses'],
        averages['page_faults'],
        averages['code_length'],
        averages['ast_node_count'],
        averages['cyclomatic_complexity'],
        averages['cognitive_complexity'],
        non_empty_ratio,
        avg_static_improvement,
        avg_dynamic_improvement
    )

def save_results(metrics_dict, output_path, result_path):
    """Save results to a JSON file if result_path is provided."""
    if not result_path:
        return
    
    experiment_name = os.path.basename(output_path.rstrip('/'))
    
    existing_results = {}
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            existing_results = json.load(f)
    
    existing_results[experiment_name] = metrics_dict
    
    with open(result_path, 'w') as f:
        json.dump(existing_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate best node performance and save results')
    parser.add_argument(
        '-o', 
        '--output_path',
        type=str,
        default=None,
        help='Path to output tasks'
    )
    parser.add_argument(
        '-s',
        '--select_method',
        type=str,
        default='success_ratio-time_enabled_ns',
        help='Strategy for selecting best node'
    )
    parser.add_argument(
        '-d',
        '--standard_metrics_path',
        type=str,
        default=None,
        help='Path to standard metrics'
    )
    parser.add_argument(
        '-r',
        '--result_path',
        type=str,
        help='Path to save results JSON file (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    task_id_2_best_node = merge_data_split(args.output_path, args.select_method)
    success_rate = calculate_success_rate(task_id_2_best_node)
    accuracy = calculate_accuracy(task_id_2_best_node)
    (
        avg_time_enabled_ns,
        avg_instruction_count,
        avg_branch_misses,
        avg_page_faults,
        avg_code_length,
        avg_ast_node_count,
        avg_cyclomatic_complexity,
        avg_cognitive_complexity,
        non_empty_ratio,
        avg_static_improvement,
        avg_dynamic_improvement
    ) = calculate_average_metrics(task_id_2_best_node, args.standard_metrics_path, success_rate)
    
    metrics = {
        "pass_at_1": accuracy,
        "avg_accuracy": success_rate,
        "non_empty_solutions": non_empty_ratio,
        "avg_time": avg_time_enabled_ns,
        "avg_instructions": avg_instruction_count,
        "avg_branch_misses": avg_branch_misses,
        "avg_page_faults": avg_page_faults,
        "avg_code_length": avg_code_length,
        "avg_ast_node_count": avg_ast_node_count,
        "avg_cyclomatic_complexity": avg_cyclomatic_complexity,
        "avg_cognitive_complexity": avg_cognitive_complexity,
        "avg_static_improvement": avg_static_improvement,
        "avg_dynamic_improvement": avg_dynamic_improvement
    }
    
    if args.result_path:
        save_results(metrics, args.output_path, args.result_path)
    
    print(f"Success rate: {success_rate}, Accuracy: {accuracy}")
    print(f"Average Static Improvement: {avg_static_improvement:.1%}")
    print(f"Average Dynamic Improvement: {avg_dynamic_improvement:.1%}")
    print(f"(Positive values indicate improvement over standard solution)")
    
    print('Results for: ', args.output_path)
    print(f"| Pass@1 | Pass Cases % | Pass Compile % | Time | Instructions | Branch misses | "
          f"Page faults | Code length | AST nodes | Cyclomatic compl. | Cognitive compl. |")
    print(f"| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    print(f"| {accuracy * 100 :.2f} | {success_rate * 100:.2f} | {non_empty_ratio * 100:.2f} | {avg_time_enabled_ns * 100 :.2f} | "
          f"{avg_instruction_count * 100 :.2f} | {avg_branch_misses * 100 :.2f} | {avg_page_faults * 100:.2f} | {avg_code_length * 100:.2f} | "
          f"{avg_ast_node_count * 100:.2f} | {avg_cyclomatic_complexity * 100 :.2f} | {avg_cognitive_complexity * 100:.2f} |") 