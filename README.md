# Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation

This repository contains the implementation code for the paper "Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation" (ICML 2025).

## Environment Setup

### Setting up with Conda

1. First, ensure you have Conda installed. If not, please install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. Create and activate a conda environment:
```bash
# Create a new environment named orps
conda create -n orps python=3.12.7
# Activate the environment
conda activate orps
```

3. Install dependencies using requirements.txt:
```bash
pip install -r requirements.txt
```

Note: Make sure you have activated the conda environment before installing the dependencies.

## Model Deployment and Configuration

### Setting up the Language Model

We recommend using Hugging Face's Text Generation Inference (TGI) for model deployment and inference. Here are the steps to set up:

1. Deploy TGI by following the instructions at [Text Generation Inference](https://github.com/huggingface/text-generation-inference).

2. After deployment, you will get a base URL for your model. Configure this URL in the configuration file to run the code.

### Model Communication Modes

Our repository supports multiple communication modes with language models:

1. **Remote HF (Recommended)**
   - Uses TGI for efficient model serving
   - Provides better performance and resource management
   - Configure the base URL in the config file after TGI deployment

2. Alternative Modes
   - `local_hf`: For local Hugging Face model inference
   - `openai`: For using OpenAI's API

While we support multiple modes, we recommend using the `remote_hf` mode with TGI for optimal performance.

## Running Experiments

### Configuration Files

The code uses JSON configuration files to specify experiment parameters. Template configuration files can be found in the `./config/templates` directory.

To run an experiment:
```bash
python run.py -c path/to/your/config.json
```

### Parallel Execution with Data Splitting

To improve efficiency, our code supports running experiments on split datasets in parallel. This is controlled by the `data_group_tuple` parameter in the configuration file.

For example, to split your dataset into 3 parts, you would create three configuration files with:
```json
// Config 1: config_split_0.json
"data_group_tuple": [3, 0]
// Config 2: config_split_1.json
"data_group_tuple": [3, 1]
// Config 3: config_split_2.json
"data_group_tuple": [3, 2]
```

Where:
- First number (3) indicates the total number of splits
- Second number (0,1,2) indicates which split this configuration will process

#### Manual Parallel Execution with tmux

After creating the split configuration files, you can manually run them in parallel using tmux:

1. Create new tmux sessions for each split:
```bash
# Create and start session for split 0
tmux new-session -d -s split0 "python run.py -c config_split_0.json"

# Create and start session for split 1
tmux new-session -d -s split1 "python run.py -c config_split_1.json"

# Create and start session for split 2
tmux new-session -d -s split2 "python run.py -c config_split_2.json"
```

2. Monitor the sessions:
```bash
# List all running tmux sessions
tmux list-sessions

# Attach to a specific session
tmux attach-session -t split0  # For monitoring split 0
tmux attach-session -t split1  # For monitoring split 1
tmux attach-session -t split2  # For monitoring split 2
```

3. Useful tmux commands while monitoring:
- `Ctrl+b d`: Detach from current session
- `Ctrl+b s`: List and switch between sessions
- `tmux kill-session -t split0`: Kill a specific session
- `tmux kill-server`: Kill all sessions

### Automated Parallel Execution

To simplify the process of running multiple splits, we provide a `run_tmux.py` script that automates the creation and execution of split configurations using tmux sessions.

Usage:
```bash
python run_tmux.py \
    --num-partitions 8 \
    --base-config path/to/base/config.json \
    --output-config-dir path/to/output/configs \
    --python-path path/to/python \
    --output-log-dir path/to/logs
```

This will:
1. Create multiple configuration files with different data splits
2. Launch separate tmux sessions for each split
3. Run experiments in parallel

Parameters:
- `num-partitions`: Number of splits to create
- `base-config`: Path to your base configuration file
- `output-config-dir`: Directory to store generated config files
- `python-path`: Path to Python interpreter
- `output-log-dir`: Directory for log files

## Calculating Experiment Results

### Step 1: Calculate Standard Runtime Metrics

Before evaluating the results, you need to calculate the standard runtime metrics for your execution environment. This is crucial because runtime metrics are environment-dependent, and we need these baseline values for proper normalization.

Use `calculate_standard_runtime_metrics.py`:
```bash
python calculate_standard_runtime_metrics.py \
    --dataset_path path/to/your/dataset \
    --dataset_type lbpp \
    --output_path path/to/save/standard/metrics
```

Parameters:
- `dataset_path`: Path to your dataset
- `dataset_type`: Type of dataset. Use:
  - `lbpp` for LBPP dataset
  - `lbpp_alter` for MBPP or HumanEval datasets
- `output_path`: Where to save the standard metrics

### Step 2: Report Experiment Results

After running experiments, use `report_scores.py` to calculate the results. The results will be stored in the directory specified by `output_path/step_name` in your configuration file.

```bash
python report_scores.py \
    --output_path path/to/experiment/output \
    --select_method success_ratio-time_enabled_ns \
    --standard_metrics_path path/to/standard/metrics \
    --result_path path/to/save/results
```

Parameters:
- `output_path`: Path to experiment output (should match `output_path/step_name` from your config)
- `select_method`: Strategy for selecting best solutions. **Important**: This must match the `select_best_node_method` specified in your configuration file. For example, if your config uses `"select_best_node_method": "success_ratio-time_enabled_ns"`, you must use the same value here.
- `standard_metrics_path`: Path to standard metrics calculated in Step 1
- `result_path`: Where to save the final results

## Citation

If you find our work helpful, please consider citing:

```bibtex
@article{yu2024reasoning,
  title={Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation},
  author={Yu, Zhuohao and Gu, Weizheng and Wang, Yidong and Jiang, Xingru and Zeng, Zhengran and Wang, Jindong and Ye, Wei and Zhang, Shikun},
  journal={arXiv preprint arXiv:2412.15118},
  year={2024}
}
```
