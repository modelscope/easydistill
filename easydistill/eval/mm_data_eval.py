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
import concurrent.futures
from threading import Lock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_cot_prompts(sample):
    prompt = f"""
    You are given a visual instruction task consisting of an image and a textual instruction about it, along with a provided response. Your job is to evaluate this task and output a JSON object with three fields:

    1. **Difficulty** (integer 1–5):  
       - 1 = Very easy (e.g., obvious object presence, simple color/shape recognition)  
       - 2 = Easy (e.g., basic counting of clearly visible items, simple spatial relations)  
       - 3 = Moderate (e.g., requires short reasoning, identifying common actions or attributes)  
       - 4 = Hard (e.g., multi-step reasoning, subtle visual cues, or uncommon concepts)  
       - 5 = Very hard (e.g., abstract reasoning, complex scene understanding, or ambiguous context)

    2. **Quality** (integer 1–5):  
       - 1 = Completely incorrect or irrelevant answer  
       - 2 = Mostly wrong with minor correct elements  
       - 3 = Partially correct but misses key details or contains errors  
       - 4 = Mostly correct with minor inaccuracies or omissions  
       - 5 = Fully accurate, precise, and complete answer

    3. **Labels** (list of lowercase strings):  
       Assign 3–6 concise, relevant, open-ended task tags describing the nature of the question (e.g., "counting", "color", "spatial", "action", "object", "attribute", "reasoning", "scene", "text", "math", etc.). Use only common, general-purpose tags.

    Output ONLY a valid JSON object in the following format:

    {{
      "Difficulty": 5,
      "Quality": 5,
      "Labels": ["math", "count", "reasoning", "object", "scene"]
    }}

    Do not include any additional text or explanation.

    ### Given Instruction and Response

    {sample}
    """
    return prompt

def extract_score(text):
    text=text.replace('```json', '').replace('```', '').strip()
    return json.loads(text)
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

import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read the image file in binary mode
        binary_data = image_file.read()
        # Encode the binary data to Base64
        base64_encoded = base64.b64encode(binary_data)
        # Convert bytes to string and return
        return base64_encoded.decode('utf-8')
    
def image2base64(image_path):
    base64_str = image_to_base64(image_path)
    # Determine MIME type from file extension
    mime_type = "image/jpeg"  # Default
    if image_path.lower().endswith(('.png', '.PNG')):
        mime_type = "image/png"
    elif image_path.lower().endswith(('.gif', '.GIF')):
        mime_type = "image/gif"
    return f"data:{mime_type};base64,{base64_str}"

def generate_teacher_response_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"]
    )
    models = client.models.list()
    model = os.getenv('JUDGE_MODEL',models.data[0].id)
    logging.info(model)
    
    def process_single_sample(sample_with_index):
        index, sample = sample_with_index
        images = []
        for item in sample:
            content = item.get('content')

            if isinstance(content, list):
                for content_item in content:
                    if content_item.get('type') == 'image':
                        image_path = content_item.get('image')
                        if image_path:
                            images.append(image_path)

        def generate_score(sample, model, config):
            try:
                D_prompt_template = build_cot_prompts(sample)
                base64_qwen=image2base64(images[0])
                message = [
                    {'role': 'user', 'content': [
                        {'type':'image_url','image_url':{"url": base64_qwen}},
                        {'type':'text','text':D_prompt_template}
                    ]}
                ]

                completion = client.chat.completions.create(
                    messages = message,
                    model = model,
                    max_completion_tokens = config["inference"]["max_new_tokens"]
                )

                result = completion.choices[0].message.content
                score = extract_score(result)
                return score
            except Exception as e:
                logging.error(f"Error in API call for sample {index}: {str(e)}")
                return None
        
        try:
            score = generate_score(sample, model, config)
            result = {
                'index': index,
                'messages': sample,
                "rating": score,
            }
        except Exception as e:
            logging.error(f"Error processing sample {index}: {str(e)}")
            result = {
                'index': index,
                'messages': sample,
            }
        return result
    samples_with_index = [(i, sample) for i, sample in enumerate(data_list)]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_to_sample = {executor.submit(process_single_sample, sample_with_index): sample_with_index 
                           for sample_with_index in samples_with_index}
        for future in tqdm(concurrent.futures.as_completed(future_to_sample), 
                          total=len(samples_with_index), 
                          desc="Call remote model and generating responses"):
            result = future.result()
            results.append(result)
    results.sort(key=lambda x: x['index'])
    outcomes = [{k: v for k, v in result.items() if k != 'index'} for result in results]

    write_data_to_json_file(outcomes, config["dataset"]["output_path"])



def infer_with_teacher_model(config):
    logging.info('Generating distillation data from the teacher model!')
    data_list = read_json_fields(config["dataset"]["input_path"])
    job_type =  config["job_type"]
    is_cot_model = "cot" in job_type
    generate_teacher_response_api(data_list, config, is_cot_model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    infer_with_teacher_model(config)


if __name__ == "__main__":
    main()