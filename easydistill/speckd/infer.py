import os
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from infer_utils.transformToTrain import format_conversations

def main():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config_all = json.load(f)
    
    split_file = config_all["dataset"]["instruction_path"]
    output_dir = config_all["dataset"]["labeled_progress_dir"]
    output_file = config_all["dataset"]["labeled_path_raw"]
    checkpoint_file = os.path.join(output_dir, "inference_checkpoint.txt")

    model_path = config_all["models"]["teacher"]

    infer_config = config_all["inference"]
    batch_size = infer_config["batch_size"]

    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.truncation_side = 'left' 

    vllm_config = infer_config["vllm_config"]
    

    llm = LLM(
        model=model_path,
        tensor_parallel_size= vllm_config["tensor_parallel_size"],
        enable_expert_parallel= vllm_config["enable_expert_parallel"],
        gpu_memory_utilization= vllm_config["gpu_memory_utilization"],
        max_model_len= vllm_config["max_model_len"],
        trust_remote_code= vllm_config["trust_remote_code"]
    )

    max_len = llm.llm_engine.model_config.max_model_len

    sampling_config = infer_config["sampling_config"]
    sampling_params = SamplingParams(
        temperature= sampling_config["temperature"],
        top_p= sampling_config["top_p"],
        max_tokens= sampling_config["max_tokens"],
    )

    processed_lines = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as cf:
            content = cf.read().strip()
            if content:
                processed_lines = int(content)
        print(f"Resuming from line {processed_lines}.")

    print(f"Loading data from {split_file}...")
    all_records = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_records.append(json.loads(line))

    total_records = len(all_records)
    print(f"Total records in file: {total_records}")

    records_to_process = all_records[processed_lines:]
    print(f"Records to process: {len(records_to_process)} (skipped {processed_lines})")

    def build_prompt(problem_text):
        messages = [{"role": "user", "content": problem_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    with open(output_file, 'a', encoding='utf-8') as data_file:
        for i in tqdm(range(0, len(records_to_process), batch_size), desc="Processing"):
            batch_records = records_to_process[i:i+batch_size]
            if not batch_records:
                continue
            
            batch_prompts = [build_prompt(record['instruction']) for record in batch_records]

            truncated_batch_prompts = []
            for prompt_str in batch_prompts:
                token_ids = tokenizer.encode(
                    prompt_str,
                    add_special_tokens=False, 
                    truncation=True,
                    max_length=max_len
                )
                
                truncated_prompt = tokenizer.decode(token_ids)
                truncated_batch_prompts.append(truncated_prompt)
            
            outputs = llm.generate(truncated_batch_prompts, sampling_params)
            
            for record, output in zip(batch_records, outputs):
                answer = output.outputs[0].text.strip()
                new_record = {
                    "instruction": record['instruction'],
                    "response": answer,
                }
                data_file.write(json.dumps(new_record, ensure_ascii=False) + '\n')
            
            processed_lines += len(batch_records)
            data_file.flush()
            
            with open(checkpoint_file, 'w', encoding='utf-8') as cf:
                cf.write(str(processed_lines))

    print(f"Completed! Processed {processed_lines} records.")

    if os.path.exists(output_file):
        format_conversations(input_path = output_file, output_path = config_all["dataset"]["labeled_path"])


if __name__ == "__main__":
    main()
