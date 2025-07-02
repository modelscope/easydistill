
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

import json
import argparse
import logging
import os
import re
from tqdm import tqdm
from openai import OpenAI
from collections import Counter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


predefined_distribution = {
    'Math': 0.167,
    'Code Generation': 0.083,
    'Writing': 0.017,
    'Computer Science': 0.017,
    'Reasoning': 0.167,
    'Complex Format': 0.017,
    'Code Debug': 0.083,
    'Common-Sense': 0.017,
    'Counterfactual': 0.017,
    'Multilingual': 0.017,
    'Roleplay': 0.017,
    'Biology': 0.017,
    'Technology': 0.017,
    'Ethics': 0.017,
    'Sport': 0.017,
    'Law': 0.017,
    'Medicine': 0.017,
    'Literature': 0.017,
    'Entertainment': 0.017,
    'Art': 0.017,
    'Music': 0.017,
    'Toxicity': 0.017,
    'Economy': 0.017,
    'Physics': 0.017,
    'History': 0.017,
    'Chemistry': 0.017,
    'Philosophy': 0.017,
    'Health': 0.017,
    'Ecology': 0.017,
    'Grammar': 0.017,
    'Paraphrase': 0.017,
    'Others': 0.041
}

predefined_prompt = """
    You are a data annotation expert. Please classify the task type or domain of #Given Instruction. 
    The task type or domain should be in the list: [’Math’, ’Code Generation’, ’Writing’, ’Computer Science’, 
    ’Reasoning’, ’Complex Format’, ’Code Debug’, ’Common-Sense’, ’Counterfactual’, ’Multilingual’, ’Roleplay’,
    ’Biology’, ’Technology’, ’Ethics’, ’Sport’, ’Law’, ’Medicine’, ’Literature’, ’Entertainment’, ’Art’, ’Music’, 
    ’Toxicity’, ’Economy’, ’Physics’, ’History’, ’Chemistry’, ’Philosophy’,’Health’,’Ecology’,’Grammar’,’Paraphrase’, 
    ’Others’]. You should place your answer enclosed within <answer></answer> tags, such as <answer>Math</answer>. 
    Do not return anything else.
    #Given Instruction#:
"""

def extract_answer(content):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None


def read_json_fields(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logging.error("The file was not found.")
    except json.JSONDecodeError:
        logging.error("There was an error decoding the JSON file.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def write_data_to_json_file(data, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logging.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def classify_instruction(instruction, client, model, config):
    message = [
        {"role": "user", "content": predefined_prompt + "\n" + instruction}
    ]
    completion = client.chat.completions.create(
        messages = message,
        model = model,
        max_completion_tokens = config["inference"]["max_new_tokens"]
    )
    result = completion.choices[0].message.content.strip()
    result = extract_answer(result)
    if result is None or result not in predefined_distribution.keys():
        result = 'Others'
    return result

        
def generate_teacher_response_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"]
    )
    models = client.models.list()
    model = models.data[0].id
    logging.info(model)

    classified_data = []
    for sample in tqdm(data_list, desc="Call remote model and generating responses"):
        instruction = sample["instruction"]
        category = classify_instruction(item['instruction'], client, model)
        new_sample = sample.copy()
        new_sample['category'] = category
        classified_data.append(new_sample)

    # Count occurrences per category
    category_counts = Counter(item['category'] for item in classified_data)
    total_samples = len(classified_data)

    # Resample according to predefined distribution
    resampled_data = []
    for category, target_ratio in predefined_distribution.items():
        target_count = int(total_samples * target_ratio)
        category_samples = [item for item in classified_data if item['category'] == category]
        if len(category_samples) == 0:
            logging.warning("No instructions are provided for the category: " + category)
            continue
        if len(category_samples) > target_count:
            # Randomly sample the required number of instructions
            resampled_category_samples = random.sample(category_samples, target_count)
        else:
            # If not enough samples, repeat the existing ones
            resampled_category_samples = category_samples * (target_count // len(category_samples)) + random.sample(
                category_samples, target_count % len(category_samples))
        resampled_data.extend(resampled_category_samples)

    write_data_to_json_file(resampled_data, config["dataset"]["output_path"])


def sample_with_teacher_model(config):
    logging.info('Generating distillation data from the teacher model!')
    data_list = read_json_fields(config["dataset"]["input_path"])
    job_type = config["job_type"]
    assert job_type == "instruct_sample_api"
    generate_teacher_response_api(data_list, config)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    sample_with_teacher_model(config)


if __name__ == "__main__":
    main()