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

import json, jsonlines
import argparse
import torch
import logging
import os
from tqdm import tqdm
from openai import OpenAI
import json
import concurrent.futures
from typing import List, Dict, Any, Set
from infer_utils.prompts_config import PROMPTS
from infer_utils.graph import run_agent
from infer_utils.train_data_conversion import conversion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Run math tasks from a JSONL file')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_all = json.load(f)

    data_path = config_all["dataset"]["instruction_path"]
    config = config_all["inference"]
    config["labeled_path_raw"] = config_all["dataset"]["labeled_path_raw"]

    PROMPTS.load(config)

    GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
    SEARCH_URL = config["SEARCH_URL"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["SEARCH_URL"] = SEARCH_URL

    output_file_path_base = os.path.dirname(config_all["dataset"]["labeled_path_raw"])
    os.makedirs(output_file_path_base, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    def get_processed_ids(output_file_path: str) -> Set[str]:
        """Get a set of already processed task IDs from the log file"""
        processed_ids = set()
        
        if not os.path.exists(output_file_path):
            return processed_ids
        
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if "original_task_info" in log_entry and "id" in log_entry["original_task_info"]:
                            processed_ids.add(log_entry["original_task_info"]["id"])
                    except json.JSONDecodeError:

                        continue
        except Exception as e:
            logger.warning(f"Warning: Error reading log file: {e}")
        
        return processed_ids

    def process_single_task(task_data: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Process a single task and return True if successful"""
        try:
            logger.info(f"processing task: {task_data['id']}")
            run_agent(
                prompt=task_data['question'],
                config=config,  
                original_task_info={
                    "id": task_data['id'],
                    "question": task_data['question'],
                    "true_answer": task_data['true_answer']
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error processing task {task_data['id']}: {e}")
            return False

    def run_math_tasks_from_file(config: Dict[str, Any]):
        """Run math tasks from a JSONL file with concurrent processing"""
        processed_ids = get_processed_ids(config_all["dataset"]["labeled_path_raw"])
        tasks_to_process: List[Dict[str, Any]] = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    task_data = json.loads(line.strip())
                    
                    if task_data['id'] in processed_ids:
                        logger.info(f"skipping completed tasks: {task_data['id']}")
                        continue
                    
                    tasks_to_process.append(task_data)
                    
                    if config["processing"]["max_tasks"] and len(tasks_to_process) >= config["processing"]["max_tasks"]:
                        break
                        
                except json.JSONDecodeError:
                    logger.warning(f"Warning: Skipping invalid JSON line")
                except Exception as e:
                    logger.error(f"Error loading task: {e}")
        
        if not tasks_to_process:
            logger.info("No new tasks to process")
            return
        
        logger.info(f"start {len(tasks_to_process)} tasks, using {config['processing']['max_workers']} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["processing"]["max_workers"]) as executor:
            future_to_task = {
                executor.submit(process_single_task, task_data, config): task_data 
                for task_data in tasks_to_process
            }
            
            completed_tasks = 0
            failed_tasks = 0
            
            for future in concurrent.futures.as_completed(future_to_task):
                task_data = future_to_task[future]
                try:
                    success = future.result()
                    if success:
                        completed_tasks += 1
                    else:
                        failed_tasks += 1
                except Exception as e:
                    logger.error(f"Task {task_data['id']} generated an exception: {e}")
                    failed_tasks += 1
            
            logger.info(f"task complete : {completed_tasks} success, {failed_tasks} fail")

    run_math_tasks_from_file(config)

    if os.path.exists(config_all["dataset"]["labeled_path_raw"]):
        conversion(input_path = config_all["dataset"]["labeled_path_raw"], output_path = config_all["dataset"]["labeled_path"])

if __name__ == "__main__":
    main()
