import json
from tqdm import tqdm
import os

def format_conversations(input_file_path: str, output_file_path: str):
    """
    Reads a jsonl file, merges 'instruction' and 'response' columns into a new 'text' column,
    and saves it to a new jsonl file.

    Args:
        input_file_path (str): The path to the input jsonl file.
        output_file_path (str): The path to the output jsonl file.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found -> {input_file_path}")
        return

    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    template = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}"

    processed_lines = 0
    malformed_lines = 0

    try:
        print("Calculating total number of lines...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
        
        print(f"Start processing file: {input_file_path}")

        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, total=total_lines, desc="Converting format"):
                try:
                    original_data = json.loads(line)
                    
                    instruction = original_data["instruction"]
                    response = original_data["response"]
                    
                    if instruction and response:
                        formatted_text = template.format(
                            instruction=instruction,
                            response=response
                        )
                        
                        new_record = {"text": formatted_text}
                        
                        outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                        processed_lines += 1
                    else:
                        malformed_lines += 1

                except (json.JSONDecodeError, KeyError):
                    malformed_lines += 1
                    continue
                    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

    print("\nProcessing complete!")
    print(f"Successfully processed and wrote {processed_lines} lines.")
    if malformed_lines > 0:
        print(f"Skipped {malformed_lines} lines (malformed or missing 'instruction'/'response' fields).")
    print(f"Results saved to: {output_file_path}")


if __name__ == "__main__":
    # input_path = "input_path.jsonl"
    # output_path = "output_path.jsonl"
    
    # format_conversations(input_path, output_path)