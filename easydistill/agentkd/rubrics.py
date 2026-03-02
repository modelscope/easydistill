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
"""Entry point for rubrics evaluation of solved virtual tool-use tasks.

Reads solution trajectories, evaluates them against rubrics via LLM,
writes per-task rubrics_output.json, and optionally produces filtered
training data from PASS solutions.
"""

import os
from dotenv import load_dotenv

load_dotenv()
import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from infer_utils.functions.call_llms import call_llm_api
from infer_utils.train_data_conversion import conversion_tool_use

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a professional AI system evaluation expert specializing in assessing the quality of tool-use task completion.

Your job is to evaluate whether a given agent solution trajectory satisfies the evaluation criteria for the task. You must be rigorous and precise — check every criterion mentioned in the evaluation description against the actual actions taken in the solution."""


def create_evaluation_prompt(
    task_description: str,
    agent_policy: str,
    test_policy: str,
    evaluation_criteria: str,
    tool_return_expected: dict,
    trajectory_text: str,
    file_name: str,
) -> str:
    normal_path_str = json.dumps(tool_return_expected.get("normal_path", []), indent=2, ensure_ascii=False)
    hack_path_str = json.dumps(tool_return_expected.get("hack_path", []), indent=2, ensure_ascii=False)

    return f"""# Task Description

{task_description}

---

# Agent Policy

{agent_policy}

---

# Test Policy (what this case is designed to test)

{test_policy}

---

# Expected Tool Call Workflow

## Normal Path (correct behavior)
{normal_path_str}

## Hack Path (incorrect behavior — should NOT happen)
{hack_path_str}

---

# Evaluation Criteria

{evaluation_criteria}

---

# Solution Trajectory: {file_name}

{trajectory_text}

---

# Your Evaluation Task

Evaluate the above solution trajectory against the evaluation criteria. Check:

1. **Tool Call Correctness**: Did the agent call the right tools in the right order with correct parameters? Compare against the normal_path.
2. **Policy Compliance**: Did the agent follow the agent policy? Did it avoid actions listed in the hack_path?
3. **Evaluation Criteria Satisfaction**: Does the solution meet every specific criterion described in the evaluation criteria section?
4. **User Interaction Quality**: Did the agent appropriately gather information from the user before acting? Did it ask for confirmation when required?

Please output your evaluation in the following format:

<reasoning>
[Your step-by-step analysis of how the solution does or does not meet each criterion]
</reasoning>

<verdict>
[PASS or FAIL]
</verdict>

<issues>
[If FAIL: list each specific criterion that was not met. If PASS: write "None"]
</issues>

Please begin your evaluation now."""


def format_trajectory(trajectory: List[Dict]) -> str:
    lines = []
    for i, message in enumerate(trajectory):
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            continue
        elif role == "user":
            if i == 1:
                lines.append(f"\n[User Request]\n{content}")
            elif "<tool_response>" in content:
                lines.append(f"\n[Tool Response]\n{content}")
            else:
                lines.append(f"\n[User Message]\n{content}")
        elif role == "assistant":
            if "<tool_call>" in content:
                reasoning = content.split("<tool_call>")[0].strip()
                tool_part = content[content.find("<tool_call>"):content.find("</tool_call>") + len("</tool_call>")]
                lines.append(f"\n[Assistant Step {i // 2}]\nReasoning: {reasoning}\nTool Call: {tool_part}")
            else:
                lines.append(f"\n[Assistant Response]\n{content}")

    return "\n".join(lines)


def load_solution_files(folder_path: str, top_k: int = 3) -> Dict[str, List[Dict]]:
    solution_files = {}
    folder = Path(folder_path)
    if not folder.exists():
        return solution_files

    json_files = sorted(folder.glob("solution*.json"))
    for file_path in json_files[:top_k]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                solution_files[file_path.name] = content
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error reading {file_path.name}: {e}")

    return solution_files


def parse_evaluation_response(response_content: str) -> Dict[str, str]:
    result = {"reasoning": "", "verdict": "", "issues": ""}

    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_content, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    verdict_match = re.search(r'<verdict>(.*?)</verdict>', response_content, re.DOTALL)
    if verdict_match:
        result["verdict"] = verdict_match.group(1).strip().upper()

    issues_match = re.search(r'<issues>(.*?)</issues>', response_content, re.DOTALL)
    if issues_match:
        result["issues"] = issues_match.group(1).strip()

    return result


def evaluate_task_solutions(
    folder_path: str,
    solution_top_k: int,
    api_base: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> Optional[Dict]:
    solution_files = load_solution_files(folder_path, solution_top_k)
    if not solution_files:
        logger.warning(f"No solution files found in {folder_path}")
        return None

    more_info_path = os.path.join(folder_path, "more_info.json")
    if not os.path.exists(more_info_path):
        logger.warning(f"more_info.json not found in {folder_path}")
        return None

    with open(more_info_path, 'r', encoding='utf-8') as f:
        more_info = json.load(f)

    task_description = more_info.get("task", "")
    agent_policy = more_info.get("agent_policy", "")
    test_policy = more_info.get("test_policy", "")
    evaluation_criteria = more_info.get("evaluation", "")
    tool_return_expected = more_info.get("tool_return_expected", {})

    if not evaluation_criteria:
        logger.warning(f"No evaluation criteria in {folder_path}")
        return None

    results = {}
    best_solution = None

    for file_name, trajectory in solution_files.items():
        trajectory_text = format_trajectory(trajectory)

        user_prompt = create_evaluation_prompt(
            task_description=task_description,
            agent_policy=agent_policy,
            test_policy=test_policy,
            evaluation_criteria=evaluation_criteria,
            tool_return_expected=tool_return_expected,
            trajectory_text=trajectory_text,
            file_name=file_name,
        )

        try:
            messages = call_llm_api(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response_content = messages[-1]["content"]
            parsed = parse_evaluation_response(response_content)
            results[file_name] = parsed

            if "PASS" in parsed["verdict"] and best_solution is None:
                best_solution = file_name
        except Exception as e:
            logger.error(f"Error evaluating {file_name}: {e}")
            results[file_name] = {"reasoning": "", "verdict": "ERROR", "issues": str(e)}

    pass_count = sum(1 for r in results.values() if "PASS" in r["verdict"])

    return {
        "evaluations": results,
        "best_solution": best_solution,
        "pass_count": pass_count,
        "total_count": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate teacher solutions against rubrics')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON configuration file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_all = json.load(f)

    config = config_all["inference"]

    rubrics_model = config["step_models"]["RubricsAgent"]
    model_name = rubrics_model["model_name"]
    model_api_config = config["api_configs"].get(model_name, config["api_configs"].get("default", {}))
    api_base = os.getenv(model_api_config.get("api_base", ""), model_api_config.get("api_base", ""))
    api_key = os.getenv(model_api_config.get("api_key_env", ""), model_api_config.get("api_key_env", ""))
    max_tokens = rubrics_model["max_tokens"]
    temperature = rubrics_model["temperature"]
    solution_top_k = rubrics_model.get("solution_top_k", 3)

    solution_path = config["paths"]["solution_path"]
    already_processed_path = config["logging"]["already_processed_path"]
    max_workers = config["processing"]["max_workers"]

    log_dir = os.path.dirname(already_processed_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    already_processed: Set[str] = set()
    if os.path.exists(already_processed_path):
        with open(already_processed_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    already_processed.add(json.loads(line.strip())["id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    tasks_to_process = [
        task_id for task_id in os.listdir(solution_path)
        if task_id not in already_processed
        and os.path.isdir(os.path.join(solution_path, task_id))
        and os.path.exists(os.path.join(solution_path, task_id, "more_info.json"))
        and any(
            f.startswith("solution") and f.endswith(".json")
            for f in os.listdir(os.path.join(solution_path, task_id))
        )
    ]

    logger.info(f"Tasks to process: {len(tasks_to_process)}, already processed: {len(already_processed)}")

    if not tasks_to_process:
        logger.info("No new tasks to evaluate")
    else:
        file_lock = Lock()

        def process_task(task_id: str):
            try:
                logger.info(f"Evaluating task: {task_id}")
                result = evaluate_task_solutions(
                    folder_path=os.path.join(solution_path, task_id),
                    solution_top_k=solution_top_k,
                    api_base=api_base,
                    api_key=api_key,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                if result is None:
                    return task_id, False, "No result returned"

                output_path = os.path.join(solution_path, task_id, "rubrics_output.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)

                with file_lock:
                    with open(already_processed_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"id": task_id}) + '\n')

                return task_id, True, None
            except Exception as e:
                logger.error(f"Error processing task {task_id}: {e}")
                return task_id, False, str(e)

        success_count = 0
        failure_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_task, task_id): task_id
                for task_id in tasks_to_process
            }

            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    tid, success, error = future.result()
                    if success:
                        success_count += 1
                        logger.info(f"Task {tid} done ({success_count}/{len(tasks_to_process)})")
                    else:
                        failure_count += 1
                        logger.warning(f"Task {tid} failed: {error}")
                except Exception as e:
                    failure_count += 1
                    logger.error(f"Task {task_id} exception: {e}")

        logger.info(f"Evaluation complete: {success_count} succeeded, {failure_count} failed out of {len(tasks_to_process)}")

    # Produce filtered training data if output_path is configured
    dataset_cfg = config_all.get("dataset", {})
    labeled_path = dataset_cfg.get("labeled_path")
    if labeled_path and os.path.isdir(solution_path):
        logger.info(f"Producing filtered training data -> {labeled_path}")
        conversion_tool_use(
            solve_path=solution_path,
            output_path=labeled_path,
        )
        logger.info("Training data conversion complete")


if __name__ == "__main__":
    main()
