import re
import json
import threading
from typing import TypedDict, Annotated, List, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from infer_utils.functions.configuration import Configuration
from infer_utils.functions.llm_functions import (
    one_thought_code_step,
    get_first_thought, answer_evaluate_wo_repair
)
from infer_utils.functions.python_interpreter import python_interpreter
from infer_utils.functions.search_tools import quick_search

# Add a lock for thread-safe file writing
log_file_lock = threading.Lock()

# Extend the AgentState to include a log of all steps and iteration count
class AgentState(TypedDict):
    history: Annotated[List[str], lambda x, y: x + y]
    python_scope: Dict[str, Any]
    is_finished: bool
    steps_log: List[Dict[str, Any]]  # Add this for logging all steps
    original_task_info: Dict[str, Any]  # To store original task information
    iteration_count: int  # Track the number of iterations
    first_thought: str  # Store the initial thought
    evaluation_result: Dict[str, Any]  # Store the evaluation result
    max_iterations: int  # Store the max iterations from config

def create_step_config(base_config: RunnableConfig, step_name: str) -> RunnableConfig:
    """Create a new configuration for a specific step with its designated model"""
    cfg = Configuration.from_runnable_config(base_config)
    
    # If step_models is configured and this step has a specific configuration
    if cfg.step_models and step_name in cfg.step_models:
        step_model_config = cfg.step_models[step_name]
        
        # Create a new config with the specific model for this step
        step_config = base_config.copy() if base_config else {}
        if "configurable" not in step_config:
            step_config["configurable"] = {}
            
        # Apply the step-specific model configuration
        step_config["configurable"]["model_name"] = step_model_config["name"]
        if "temperature" in step_model_config:
            step_config["configurable"]["temperature"] = step_model_config["temperature"]
        if "max_tokens" in step_model_config:
            step_config["configurable"]["max_tokens"] = step_model_config["max_tokens"]
            
        return step_config
    
    # If no specific configuration for this step, return the base config
    return base_config

def first_thought_node(state: AgentState, config: RunnableConfig):
    """Generate the initial thought for the problem."""
    # Create step-specific configuration
    step_config = create_step_config(config, "first_thought")
    cfg = Configuration.from_runnable_config(step_config)
    
    original_query = state["history"][0]
    
    # Generate the first thought
    first_thought = get_first_thought(
        cfg, input_query=original_query,
    )
    
    # Add the first thought to history
    history_entry = f"Initial Thought: {first_thought}"
    
    return {
        "history": [history_entry],
        "original_task_info": state["original_task_info"],
        "first_thought": first_thought
    }

def reasoning_node(state: AgentState, config: RunnableConfig):
    """Think and decide on the next action."""
    # Create step-specific configuration
    step_config = create_step_config(config, "reasoning")
    cfg = Configuration.from_runnable_config(step_config)
    
    prompt_history = "\n".join(state["history"][2:])
    first_thought = state["first_thought"]
    original_query = state["history"][0]

    thought, code = one_thought_code_step(
        cfg, idx=state["iteration_count"],
        input_query=original_query,
        first_thought=first_thought,
        previous_context=prompt_history
    )

    # extract the web_search("...") query
    query_match = re.search(r'web_search\("(.*?)"\)', code, re.DOTALL)
    if query_match:
        try:
            result = {}
            web_search_query = query_match.group(1)
            result["output"] = quick_search(web_search_query)
        except Exception as e:
            result = {
                "output": f"Error: {str(e)}",
                "updated_scope": state["python_scope"]
            }
            
    else:
        result = python_interpreter(code, state["python_scope"])
    
    history_entry = f"{thought}\nCode:\n```python\n{code}\n```"
    observation = f"Observation: {result['output']}"

    is_finished = False
    if "final_answer_print" in code:
        is_finished = True
    
    # Log this step
    step_entry = {
        "thought": thought,
        "code": code,
        "observation": str(result['output']).strip(),
        "is_finished": is_finished
    }
    
    return {
        "history": [history_entry, observation], 
        "python_scope": result["updated_scope"],
        "is_finished": is_finished,
        "steps_log": state["steps_log"] + [step_entry],  # Add step to log
        "iteration_count": state["iteration_count"] + 1,  # Increment iteration count
    }

def evaluation_node(state: AgentState, config: RunnableConfig):
    """Evaluate the correctness of the final answer."""
    # Create step-specific configuration
    step_config = create_step_config(config, "evaluation")
    cfg = Configuration.from_runnable_config(step_config)
    
    # Extract necessary information
    original_task_info = state["original_task_info"]
    steps_log = state["steps_log"]
    
    # Get the final answer from the last step if available
    generated_answer = ""
    if steps_log and len(steps_log) > 0:
        last_step = steps_log[-1]
        if not last_step["is_finished"]:
            return {
                "evaluation_result": "Not finished"
            }
        
        if "observation" in last_step:
            generated_answer = last_step["observation"]
    
    question = original_task_info.get("question", "")
    true_answer = original_task_info.get("true_answer", "")

    evaluation_result_str = answer_evaluate_wo_repair(
        cfg=cfg, question=question,
        true_answer=true_answer,
        generated_answer=generated_answer,
    )
    return {
        "evaluation_result": evaluation_result_str
    }

def should_continue(state: AgentState):
    # Get max iterations from state (set during initialization)
    max_iterations = state.get("max_iterations", 10)  # Default to 10 if not set
    
    # Check if we've reached the maximum iterations
    if state.get("iteration_count", 0) >= max_iterations:
        return "evaluate"
    # Check if the task is finished
    elif state.get("is_finished"):
        return "evaluate"
    else:
        return "continue"

def should_end_after_evaluation(state: AgentState):
    return "end"

# Build the graph with the new first thought node and evaluation node
builder = StateGraph(AgentState, config_schema=RunnableConfig)
builder.add_node("first_thought", first_thought_node)
builder.add_node("reason_and_act", reasoning_node)
builder.add_node("evaluate_answer", evaluation_node)
builder.set_entry_point("first_thought")
builder.add_edge("first_thought", "reason_and_act")
builder.add_conditional_edges(
    "reason_and_act",
    should_continue,
    {"continue": "reason_and_act", "evaluate": "evaluate_answer"}
)
builder.add_edge("evaluate_answer", END)
graph = builder.compile()


def run_agent(prompt: str, config: dict = None, original_task_info: dict = None):
    run_config = {"configurable": config or {}}
    
    # Extract max_iterations from config
    max_iterations = 10  # Default value
    if config and "processing" in config and "max_reasoning_iterations" in config["processing"]:
        max_iterations = config["processing"]["max_reasoning_iterations"]
    
    initial_state = {
        "history": [prompt],
        "python_scope": {},
        "is_finished": False,
        "steps_log": [],  # Initialize empty steps log
        "original_task_info": original_task_info or {},  # Store original task info
        "iteration_count": 0,  # Initialize iteration count
        "first_thought": "",  # Initialize first thought
        "max_iterations": max_iterations  # Set max iterations from config
    }
    
    final_state = graph.invoke(initial_state, config=run_config)
    final_response = final_state['history'][-1]
    
    # Save all steps to the single JSONL file
    save_steps_log(
        final_state['steps_log'], 
        prompt, 
        final_state['original_task_info'],
        final_state.get('evaluation_result', {}),
        final_state.get('first_thought', ''),
        config["labeled_path_raw"]
    )
    
    return final_state

def save_steps_log(steps_log: List[Dict[str, Any]], original_prompt: str, 
                   original_task_info: dict = None, evaluation_result: dict = None,
                   first_thought: str = None,  
                   log_file_path: str = "agent_logs.jsonl"):
    """Save all steps to a single JSONL file (thread-safe)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_data = {
        "timestamp": timestamp,
        "steps": steps_log
    }
    
    # Include first thought if available
    if first_thought:
        log_data["first_thought"] = first_thought
        
    # Include original task information if available
    if original_task_info:
        log_data["original_task_info"] = original_task_info
        
    # Include evaluation result if available
    if evaluation_result:
        log_data["evaluation_result"] = evaluation_result
    
    # Thread-safe write to the JSONL file
    with log_file_lock:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
    

if __name__ == "__main__":
    # Example usage
    task = "Calculate a when a^2=2, then calculate a^2, please seperate it with two code blocks."
    from infer_utils.prompts_config import PROMPTS
    with open("configs/agentkd_local.json", "r") as f:
        config = json.load(f)["inference"]
    PROMPTS.load(config)
    run_agent(task, config)
