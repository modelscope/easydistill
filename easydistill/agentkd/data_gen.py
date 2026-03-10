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
"""Entry point for virtual tool-use task synthesis (data generation).

Reads persona seeds from a JSONL file, synthesizes virtual tool-use tasks
(tools + policy + test cases) via LLM, and writes the results to a JSONL file.
"""

import json
import logging
import os
import argparse
import concurrent.futures
from typing import List, Dict, Any, Set

from infer_utils.graph.virtual_tools import run_data_gen_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_processed_ids(log_file_path: str) -> Set[str]:
    processed_ids: Set[str] = set()
    if not os.path.exists(log_file_path):
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        return processed_ids
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    if "id" in log_entry:
                        processed_ids.add(log_entry["id"])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Warning: Error reading log file: {e}")
    return processed_ids


def process_single_task(task_data: Dict[str, Any], config: Dict[str, Any]) -> bool:
    try:
        logger.info(f"Processing task: {task_data['id']}")
        run_data_gen_agent(
            seed_info={
                "id": task_data['id'],
                "background": task_data["persona"]
            },
            run_config=config
        )
        return True
    except Exception as e:
        logger.error(f"Error processing task {task_data['id']}: {e}")
        return False


def run_tasks_from_file(config: Dict[str, Any]):
    processed_ids = get_processed_ids(config["logging"]["task_file_path"])
    tasks_to_process: List[Dict[str, Any]] = []

    with open(config["paths"]["data_file"], 'r', encoding='utf-8') as f:
        for line in f:
            try:
                task_data = json.loads(line.strip())
                if task_data['id'] in processed_ids:
                    logger.info(f"Skipping processed task: {task_data['id']}")
                    continue
                tasks_to_process.append(task_data)
                if config["processing"]["max_tasks"] and len(tasks_to_process) >= config["processing"]["max_tasks"]:
                    break
            except json.JSONDecodeError:
                logger.warning("Warning: Skipping invalid JSON line")
            except Exception as e:
                logger.error(f"Error loading task: {e}")

    if not tasks_to_process:
        logger.info("No new tasks to process")
        return

    logger.info(f"Starting {len(tasks_to_process)} tasks with {config['processing']['max_workers']} workers")

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
        logger.info(f"Processing completed: {completed_tasks} successful, {failed_tasks} failed")


def main():
    parser = argparse.ArgumentParser(description='Synthesize virtual tool-use tasks from persona seeds')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_all = json.load(f)

    inference_config = config_all["inference"]
    run_tasks_from_file(inference_config)


if __name__ == "__main__":
    main()
