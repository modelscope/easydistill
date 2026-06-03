
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_json_field(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        output = []
        for item in data:
            output.append({"prompt": item["instruction"], "chosen": item["output"]})
        return output
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


def generate_student_response(data_list, config):
    # load student model
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        device_map="auto",
        trust_remote_code=True
    ) 
    outcomes = []
    for sample in tqdm(data_list, desc="Call remote model and generating responses"):
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        # for student model
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = student_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = student_tokenizer([text], return_tensors="pt").to(student_model.device)
        generated_ids = student_model.generate(
            **model_inputs,
            max_new_tokens=config["inference"]["max_new_tokens"]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        rejected = student_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        gen_data = {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}
        outcomes.append(gen_data)
    write_data_to_json_file(outcomes, config["dataset"]["labeled_path"])

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    data_list = read_json_field(config["dataset"]["instruction_path"])
    generate_student_response(data_list, config)


if __name__ == "__main__":
    main()