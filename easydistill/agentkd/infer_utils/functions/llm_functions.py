# functions/functions.py
import re
import json
from typing import Optional
from openai import OpenAI
from ..prompts_config import PROMPTS

def call_llm_api(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
):
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
        
        models = client.models.list()
        if models.data and model_name in [model.id for model in models.data]:
            dynamic_model_id = model_name
        else:
            dynamic_model_id = models.data[0].id
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
        
        response = client.chat.completions.create(
            model=dynamic_model_id,
            messages=messages,
            temperature=temperature,
            extra_body={"max_completion_tokens": max_tokens}
        )
        response_content = response.choices[0].message.content

        return response_content
    except Exception as e:
        raise Exception(f"[{__file__}:{__import__('inspect').currentframe().f_lineno-15}] Failed to call LLM API: {e}") from e

def one_thought_code_step(
    cfg, idx, input_query,
    first_thought="None",
    previous_context="None",
    failed_experience="None"
):
    thought_code_query = PROMPTS.REACT_USER_PROMPT.format(
        query=input_query, idx=idx,
        first_thought=first_thought,
        failed_experience=failed_experience,
        previous_context=previous_context,
    )

    thought_code_content = call_llm_api(
        user_prompt=thought_code_query,
        system_prompt=PROMPTS.REACT_SYSTEM_PROMPT,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    if thought_code_content is None:
        raise Exception(f"Failed to generate thought and code for query: {input_query[:100]}...")

    thought_code_content = re.sub(
        r'\<think\>\n(.*?)\</think\>\n',
        '', thought_code_content, flags=re.DOTALL
    )

    # Extract Thought and Code. The thought is everything outside/before the code block.
    code_match = re.search(r'```python(.*?)```', thought_code_content, re.DOTALL)
    if code_match:
        code_content = code_match.group(1).strip()
        # The thought is the part of the content before the code block starts.
        thought_content = thought_code_content[:code_match.start()].strip()
    else:
        # If no code block is found, there's no code, and the whole content is the thought.
        code_content = ''
        thought_content = thought_code_content.strip()

    # Clean up keywords from the extracted thought
    thought_content = thought_content.replace('Code:', '').strip()

    return thought_content, code_content


def get_first_thought(cfg, input_query):
    """Generate the first thought prefix before the Thought-Code-Observation cycle."""
    first_query = PROMPTS.FIRST_THOUGHT_USER_PROMPT.format(query=input_query)

    initial_thought = call_llm_api(
        user_prompt=first_query,
        system_prompt=PROMPTS.FIRST_THOUGHT_SYSTEM_PROMPT,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    if initial_thought is None:
        raise Exception(f"[{__file__}:{__import__('inspect').currentframe().f_lineno-15}] Failed to generate first thought for query")
    
    # Remove the \<think\> tags from the response
    initial_thought = re.sub(r'\<think\>\n(.*?)\</think\>\n', '', initial_thought, flags=re.DOTALL)
    return initial_thought

def answer_evaluate_wo_repair(cfg, question, true_answer, generated_answer):
    system_prompt = """You are an expert judge evaluating whether two answers are equivalent.
        You will be given a predicted answer and a ground truth answer.
        Your task is to determine if they are semantically equivalent, ignoring minor differences in formatting, spacing, or notation.
        You must respond in JSON format with a 'equivalent' field that is either true or false."""
        
    prompt = f"""Question: {question}

    Predicted answer: {generated_answer}
        
    Ground truth answer: {true_answer}
        
    Are these answers semantically equivalent? Respond in JSON format with only the 'equivalent' field.

    Sometimes, predicted_answer may contain some irrelevant content, please ignore it, as long as predicted_answer contains the final answer, it is considered correct.
    Example: predicted_answer: {{The two sets are different because the sum of the remainders cannot equal the sum of the integers under the given constraints.}}.ground_truth: {{The two sets are different.}}. predicted_answer should be considered correct.

    Example response: {{"equivalent": true}}
    """
        
    evaluation_result_str = call_llm_api(
        user_prompt=prompt, 
        system_prompt=system_prompt, 
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    
    return evaluation_result_str

def answer_evaluate(cfg, question, true_answer, generated_answer, thought_code_cycle):
    """Evaluate the correctness of the final answer."""
    # Create the evaluation prompt
    evaluation_prompt = PROMPTS.JUDGE_ANSWER_PROMPT.format(
        question=question, true_answer=true_answer,
        generated_answer=generated_answer,
        thought_code_cycle=thought_code_cycle
    )
    
    # Call the LLM to evaluate the answer
    try:
        evaluation_result_str = call_llm_api(
            user_prompt=evaluation_prompt,
            system_prompt="You are a precise evaluator who analyzes reasoning processes and determines if answers are correct. Always respond with a valid JSON object and ensure all fields are properly filled.",
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model_name=cfg.model_name,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        
        if evaluation_result_str is None:
            raise Exception(f"[{__file__}:{__import__('inspect').currentframe().f_lineno-15}] Failed to evaluate answer for question: {question[:100]}...")
        
        # Use regex to extract fields from the response
        is_correct_match = re.search(r'"is_correct"\s*:\s*(true|false)', evaluation_result_str, re.IGNORECASE)
        error_analysis_match = re.search(r'"error_analysis"\s*:\s*"([^"]*)"', evaluation_result_str)
        correction_start_step_match = re.search(r'"correction_start_step"\s*:\s*(\d+|null)', evaluation_result_str, re.IGNORECASE)
        correction_suggestion_match = re.search(r'"correction_suggestion"\s*:\s*"([^"]*)"', evaluation_result_str)
        
        # Build a proper JSON object
        result_dict = {}
        
        # Handle is_correct field
        if is_correct_match:
            is_correct_val = is_correct_match.group(1).lower()
            result_dict["is_correct"] = True if is_correct_val == "true" else False
        else:
            result_dict["is_correct"] = False
            
        # Handle error_analysis field
        if error_analysis_match:
            # Unescape quotes and other characters
            error_analysis = error_analysis_match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            result_dict["error_analysis"] = error_analysis
        else:
            result_dict["error_analysis"] = None
            
        # Handle correction_start_step field
        if correction_start_step_match:
            correction_start_step_val = correction_start_step_match.group(1).lower()
            if correction_start_step_val == "null":
                result_dict["correction_start_step"] = None
            else:
                try:
                    result_dict["correction_start_step"] = int(correction_start_step_val)
                except ValueError:
                    result_dict["correction_start_step"] = None
        else:
            result_dict["correction_start_step"] = None
            
        # Handle correction_suggestion field
        if correction_suggestion_match:
            # Unescape quotes and other characters
            correction_suggestion = correction_suggestion_match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            result_dict["correction_suggestion"] = correction_suggestion
        else:
            result_dict["correction_suggestion"] = None
            
        # Serialize to proper JSON
        return json.dumps(result_dict, ensure_ascii=False)
        
    except Exception as e:
        raise Exception(f"[{__file__}:{__import__('inspect').currentframe().f_lineno-15}] Error during evaluation: {str(e)}") from e
