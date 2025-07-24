
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
import math
import argparse
import logging
from tqdm import tqdm
from openai import OpenAI
import torch
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_json_field(filename):
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

def load_tokenizer_and_vllm(config, eos_token=None):

    model_path = config["models"]["teacher"]
    logging.info(f"Loading processor & vLLM model from {model_path}")

    # 1. Use AutoProcessor, which integrates the tokenizer, image_processor, and video_processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 2. eos / pad token 处理（与官方示例保持一致，不再显式改 pad_token）
    if eos_token:
        eos_token_id = processor.tokenizer.convert_tokens_to_ids(eos_token)
        logging.info(f"eos_token {eos_token} from user input")
    elif hasattr(processor.tokenizer, "eos_token_id") and processor.tokenizer.eos_token_id is not None:
        eos_token_id = processor.tokenizer.eos_token_id
        eos_token = processor.tokenizer.convert_ids_to_tokens(eos_token_id)
        logging.info(f"Initial eos_token_id {eos_token_id} from tokenizer")
    else:
        raise ValueError("No available eos_token or eos_token_id.")

    # 3. 设置 tokenizer 的 eos 相关字段（pad_token 保持 None，由 vLLM 自动处理）
    try:
        processor.tokenizer.eos_token = eos_token
        processor.tokenizer.eos_token_id = eos_token_id
    except Exception as e:
        logging.warning(f"[WARNING] Cannot set eos_token: {e}")

    logging.info(
        f"processor.tokenizer eos_token: {processor.tokenizer.eos_token}, "
        f"eos_token_id: {processor.tokenizer.eos_token_id}"
    )

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 10, "video": 10},   # 可按需调整
        # 其余超参沿用原 config
        gpu_memory_utilization=config["inference"].get("gpu_memory_utilization", 0.9),
        max_model_len=config["inference"].get("max_model_len", 4096),
        enforce_eager=config["inference"].get("enforce_eager", False),
    )

    logging.info("Qwen2.5-VL vLLM model loaded successfully")
    #return processor, llm
    
    return processor, llm

def generate_teacher_response_batch(processor, llm, data_list, config, batch_size=32):

    outcomes = []
    sampling_params = SamplingParams(
        n = 1,
        top_k = 1,
        temperature=config["inference"]["temperature"],
        seed = config["inference"]["seed"],
        max_tokens = config["inference"]["max_new_tokens"],
    )
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    for batch in tqdm(batches, desc="Generating responses"):
        new_batch = []
        batch_outcomes = []
        for sample in batch:
            batch_outcomes.append(sample)
            prompt = processor.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(sample)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            sample_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            new_batch.append(sample_inputs)
        outputs = llm.generate(new_batch, sampling_params=sampling_params)
        for b in range(len(batch_outcomes)):
       
            generated_text = outputs[b].outputs[0].text
            out={
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": generated_text,
                    }
                ],
            }
            batch_outcomes[b].append(out)
        outcomes.extend(batch_outcomes)
    write_data_to_json_file(outcomes, config["dataset"]["labeled_path"])
    
def generate_teacher_logits_batch(processor, llm, data_list, config, batch_size=32):

    outcomes = []
    sampling_params = SamplingParams(
        n = 1,
        top_k = 1,
        temperature=config["inference"]["temperature"],
        seed = config["inference"]["seed"],
        skip_special_tokens=False,
        max_tokens = config["inference"]["max_new_tokens"],
        logprobs=config["inference"]["top_logits_num"],
    )
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    logits=[]
    for batch in tqdm(batches, desc="Generating responses"):
        new_batch = []
        batch_outcomes = []
        for sample in batch:
            batch_outcomes.append(sample)
            prompt = processor.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(sample)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            sample_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            new_batch.append(sample_inputs)
        outputs = llm.generate(new_batch, sampling_params=sampling_params)
        logits+=[output.outputs[0].logprobs for output in outputs]

        for b in range(len(batch_outcomes)):
       
            generated_text = outputs[b].outputs[0].text
            out={
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": generated_text,
                    }
                ],
            }
            batch_outcomes[b].append(out)
        outcomes.extend(batch_outcomes)

    for logit in logits:
        for pos in logit:
            for k,v in pos.items():
                pos[k]=math.exp(v.logprob)
    
    with jsonlines.open(config["dataset"]["logits_path"], mode='w') as writer:
        for row in logits:
            #for item in row:
            writer.write(row)
    
    write_data_to_json_file(outcomes, config["dataset"]["labeled_path"])

            
    

def generate_teacher_response_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"]
    )
    models = client.models.list()
    model = models.data[0].id
    logging.info(model)
    system_prompt = config["inference"]["system_prompt"]
    if system_prompt == "":
        system_prompt = "You are a helpful assistant."
    outcomes = []
    for text, image in tqdm(data_list, desc="Call remote model and generating responses"):
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        },
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
        completion = client.chat.completions.create(
            messages = messages,
            model = model,
            max_completion_tokens = config["inference"]["max_new_tokens"]
        )
        result = completion.choices[0].message.content
        outcomes.append({'instruction': text, 'image': image, 'output': result})
    write_data_to_json_file(outcomes, config["dataset"]["labeled_path"])


def infer_with_teacher_model(config):
    logging.info('Generating distillation data from the teacher model!')
    data_list = read_json_field(config["dataset"]["instruction_path"])
    
    try:
        job_type =  config["job_type"]
        if job_type == "mmkd_black_box_api":
            generate_teacher_response_api(data_list, config)
        elif job_type == "mmkd_black_box_local":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            generate_teacher_response_batch(tokenizer, llm, data_list, config)
        elif job_type == "mmkd_white_box":
            
            tokenizer, llm = load_tokenizer_and_vllm(config)
            generate_teacher_logits_batch(tokenizer, llm, data_list, config)
        else:
            logging.error(f"Invalid job type: {job_type}")
            raise ValueError(f"Invalid job type: {job_type}")
    except ValueError as e:
        logging.error(f"Training job terminated: {e}")
        return

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    infer_with_teacher_model(config)


if __name__ == "__main__":
    main()