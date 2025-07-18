
# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import subprocess
import sys
from socket import socket
import argparse
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

def run_cmd(cmd):
    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            shell=True,
            universal_newlines=True  # Ensure output is in text mode
        )
        
        error_detected = False
        error_keywords = [
            "ERROR",
            "Error",
            "error"
            "Unrecognized model",
            "failed",
            "exception",
            "Traceback"
        ]
        
        # Read output in real-time and detect errors
        while True:
            line = p.stdout.readline()
            if not line:
                break
            logging.info(line.rstrip())  # Log normally
            
            # Check if any error keywords are present
            if any(keyword.lower() in line.lower() for keyword in error_keywords):
                error_detected = True
                logging.error(f"Detected error in output: {line.strip()}")
        
        # Wait for process to finish
        returncode = p.wait()
        
        # If errors were detected or return code is non-zero, return False
        if error_detected or returncode != 0:
            logging.error(f"Command failed (returncode={returncode}, errors detected)")
            return False
        
        return True  # Return True indicates success
        
    except Exception as e:
        logging.error(f"Unexpected error running command: {e}")
        return False

def process(job_type, config):
    if not os.path.isabs(config):
        config = os.path.join(script_dir, config)
    
    # Knowledge Distillation tasks
    if job_type in ['kd_black_box_train_only', 'kd_white_box_train_only']:
        cmd_train = [
            'accelerate', 'launch',
            '--config_file', os.path.join(parent_dir, 'configs/accelerate_config/muti_gpu.yaml'),
            os.path.join(script_dir, 'kd/train.py'),
            '--config', config
        ]
        cmd_train = ' '.join(cmd_train)
        logging.info(f"Running command: {cmd_train}")
        run_cmd(cmd_train)

    elif job_type in ['kd_black_box_train_only_multi', 'kd_white_box_train_only_multi']:
        cmd_train = [
            'accelerate', 'launch',
            '--config_file', os.path.join(parent_dir, 'configs/accelerate_config/muti_gpu.yaml'),
            os.path.join(script_dir, 'kd/multi_train.py'),
            '--config', config
        ]
        cmd_train = ' '.join(cmd_train)
        logging.info(f"Running command: {cmd_train}")
        run_cmd(cmd_train)

    elif job_type in ['kd_black_box_api', 'kd_black_box_local', 'kd_white_box']:
        cmd_infer = [
            'python', os.path.join(script_dir, 'kd/infer.py'),
            '--config', config
        ]
        cmd_infer = ' '.join(cmd_infer)
        logging.info(f"Running command: {cmd_infer}")
        infer_success = run_cmd(cmd_infer)
        if infer_success:
            cmd_train = [
                'accelerate', 'launch',
                '--config_file', os.path.join(parent_dir, 'configs/accelerate_config/muti_gpu.yaml'),
                os.path.join(script_dir, 'kd/train.py'),
                '--config', config
            ]
            cmd_train = ' '.join(cmd_train)
            logging.info(f"Running command: {cmd_train}")
            run_cmd(cmd_train)
        else:
            logging.error("Infer failed, skipping training")
    
    # Reinforcement Learning tasks
    elif job_type in ['rl_ppo', 'rl_grpo']:
        cmd = [
            'accelerate', 'launch',
            '--config_file', os.path.join(parent_dir, 'configs/accelerate_config/muti_gpu.yaml'),
            os.path.join(script_dir, f'rl/{job_type.split("_")[1]}.py'),
            '--config', config
        ]
        cmd = ' '.join(cmd)
        logging.info(f"Running command: {cmd}")
        run_cmd(cmd)
    
    elif job_type in ['rl_reward_api', 'rl_reward_local']:
        cmd = [
            'python',
            os.path.join(script_dir, 'rl/reward.py'),
            '--config', config
        ]
        cmd = ' '.join(cmd)
        logging.info(f"Running command: {cmd}")
        run_cmd(cmd)
    
    # Instruction Processing tasks
    elif job_type.startswith('instruction_'):
        task_type = job_type.replace('instruction_', '')
        cmd = [
            'python',
            os.path.join(script_dir, f'synthesis/synthesis_main.py'),
            '--config', config
        ]
        cmd = ' '.join(cmd)
        logging.info(f"Running command: {cmd}")
        run_cmd(cmd)
    
    # Chain of Thought tasks
    elif job_type.startswith('cot_'):
        task_type = job_type.replace('cot_', '')
        cmd = [
            'python',
            os.path.join(script_dir, f'synthesis/synthesis_main.py'),
            '--config', config
        ]
        cmd = ' '.join(cmd)
        logging.info(f"Running command: {cmd}")
        run_cmd(cmd)
    
    # Ranking and DPO tasks
    elif job_type.startswith('rank_'):
        task_type = job_type.replace('rank_', '')
        cmd = [
            'python',
            os.path.join(script_dir, f'rank/{task_type}.py'),
            '--config', config
        ]
        cmd = ' '.join(cmd)
        logging.info(f"Running command: {cmd}")
        run_cmd(cmd)
    
    else:
        logging.error(f"Unknown job type: {job_type}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config_path = args.config
    config = json.load(open(config_path))
    job_type = config["job_type"]
    process(job_type, config_path)  

if __name__ == '__main__':
    main()
