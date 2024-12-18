import os
import json
import subprocess
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Create and run multiple tmux sessions for evaluation tasks')
    
    parser.add_argument('--num-partitions', '-n', type=int, default=8,
                        help='Total number of partitions (default: 8)')
    
    parser.add_argument('--base-config', '-c', type=str, 
                        required=True,
                        help='Path to base configuration file')
    
    parser.add_argument('--output-config-dir', '-o', type=str,
                        required=True,
                        help='Directory for generated configuration files')
    
    parser.add_argument('--python-path', '-p', type=str,
                        required=True,
                        help='Path to python interpreter')

    parser.add_argument('--output-log-dir', '-l', type=str,
                        help='Directory for generated log files')
    

    
    return parser.parse_args()

def create_config_files(args):
    """Create configuration files"""
    os.makedirs(args.output_config_dir, exist_ok=True)
    
    with open(args.base_config, 'r') as f:
        base_config = json.load(f)
    base_step_name = base_config['steps'][0]['step_name']
    base_output_name = args.base_config.split('/')[-1].split('.')[0]
    
    config_files = []
    for offset in range(args.num_partitions):
        base_config['steps'][0]['data_group_tuple'] = [args.num_partitions, offset]
        new_log_path = os.path.join(args.output_log_dir, f'{base_output_name}_{base_step_name}_{args.num_partitions}_{offset}.log')
        new_config_path = os.path.join(args.output_config_dir, f"{base_output_name}_{base_step_name}_{args.num_partitions}_{offset}.json")
        with open(new_config_path, 'w') as f:
            json.dump(base_config, f, indent=4)
        
        config_files.append({"config_path": new_config_path, "log_path": new_log_path})
    
    return config_files

def create_tmux_sessions(config_files, args):
    """Create tmux sessions and run commands"""
    abs_config_files = [{"config_path": str(Path(f["config_path"]).absolute()), 
                        "log_path": str(Path(f["log_path"]).absolute())} for f in config_files]

    with open(args.base_config, 'r') as f:
        base_config = json.load(f)
    base_step_name = base_config['steps'][0]['step_name']
    
    for i, config_file in enumerate(abs_config_files):
        session_name = f"{base_step_name}_{i}"
        
        subprocess.run([
            "tmux", "new-session",
            "-d",
            "-s", session_name,
            f"{args.python_path} run.py -c {config_file['config_path']} -l {config_file['log_path']}"
        ])
        
        print(f"Created tmux session: {session_name} with config: {config_file}")

def main():
    args = parse_args()
    
    print("Creating config files...")
    config_files = create_config_files(args)
    
    print("Creating tmux sessions...")
    create_tmux_sessions(config_files, args)
    
    print(f"\nCreated {len(config_files)} tmux sessions.")
    print("You can attach to a session using: tmux attach-session -t lbpp_<number>")
    print("List all sessions using: tmux list-sessions")

if __name__ == "__main__":
    main() 
