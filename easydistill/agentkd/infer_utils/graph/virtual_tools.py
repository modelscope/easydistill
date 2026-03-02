import os
import json
import threading
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from infer_utils.functions.configuration import Configuration
from infer_utils.functions.policy_task import generate_policy_test_case
from infer_utils.functions.tool_set_policy_gen import generate_tool_set_policy
from infer_utils.functions.refine_policy_task import generate_task_and_user_background

log_file_lock = threading.Lock()


class DataGenAgentState(TypedDict):
    seed_info: Dict[str, Any]
    breaked: bool

    initial_toolset_create: str
    initial_tools: str
    initial_task: str
    initial_policy: str

    policy_str: str
    test_cases: List[str]

    checked_tools: List[Dict[str, Any]]
    tasks_and_backgrounds: List[Dict[str, Any]]


_STEP_CONFIG_KEYS = ("model_name", "temperature", "max_tokens")


def create_step_config(
    base_config: RunnableConfig, step_name: str,
) -> RunnableConfig:
    step_model = base_config["configurable"]["step_models"][step_name]
    configurable = {k: v for k, v in step_model.items() if k in _STEP_CONFIG_KEYS}
    configurable["api_configs"] = base_config["configurable"]["api_configs"]
    return {"configurable": configurable}


def toolset_gen_node(state: DataGenAgentState, config: RunnableConfig):
    step_config = create_step_config(config, "ToolSetGenAgent")
    cfg = Configuration.from_runnable_config(step_config)

    original_bg = state["seed_info"]["background"]
    all_content, task, tools, policy = generate_tool_set_policy(cfg=cfg, background_info=original_bg)

    if task is None or tools is None or policy is None:
        return {
            "breaked": True,
            "initial_toolset_create": None,
            "initial_task": None,
            "initial_tools": None,
            "initial_policy": None
        }

    return {
        "initial_toolset_create": all_content,
        "initial_task": task,
        "initial_tools": tools,
        "initial_policy": policy
    }


def policy_task_node(state: DataGenAgentState, config: RunnableConfig):
    if state["breaked"]:
        return {
            "policy_str": None,
            "test_cases": None
        }
    step_config = create_step_config(config, "PolicyTaskAgent")
    cfg = Configuration.from_runnable_config(step_config)

    task_description = state["initial_task"]
    policy_tree = state["initial_policy"]
    tools = state["initial_tools"]
    try:
        tools = json.loads(tools)
    except Exception:
        print(f"Error: The tools is not a valid JSON string: {tools}")
        return {
            "breaked": True,
            "policy_str": None,
            "test_cases": None
        }
    all_content, policy, test_cases_task_bg_policy = generate_policy_test_case(cfg, task_description, policy_tree, tools)

    return {
        "checked_tools": tools,
        "policy_str": policy,
        "test_cases": test_cases_task_bg_policy
    }


def final_task_node(state: DataGenAgentState, config: RunnableConfig):
    if state["breaked"]:
        return {
            "tasks_and_backgrounds": None
        }
    step_config = create_step_config(config, "FinalTaskAgent")
    cfg = Configuration.from_runnable_config(step_config)

    test_cases = state["test_cases"]
    tools = state["checked_tools"]

    task_and_user_background = generate_task_and_user_background(cfg, tools, test_cases)

    if not task_and_user_background:
        return {
            "breaked": True,
            "tasks_and_backgrounds": None
        }

    return {
        "tasks_and_backgrounds": task_and_user_background
    }


builder = StateGraph(DataGenAgentState, config_schema=RunnableConfig)
builder.add_node("toolset_gen", toolset_gen_node)
builder.add_node("policy_task", policy_task_node)
builder.add_node("final_task", final_task_node)

builder.set_entry_point("toolset_gen")
builder.add_edge("toolset_gen", "policy_task")
builder.add_edge("policy_task", "final_task")
builder.add_edge("final_task", END)
data_gen_graph = builder.compile()


def run_data_gen_agent(seed_info: dict, run_config: dict = None):
    virtual_tool_use_task_path = run_config["logging"]["task_file_path"]

    log_dir = os.path.dirname(virtual_tool_use_task_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_config_inner = {"configurable": run_config or {}}
    run_config_inner["recursion_limit"] = 100

    initial_state = {
        "seed_info": seed_info,
        "breaked": False,
    }
    final_state = data_gen_graph.invoke(initial_state, config=run_config_inner)

    save_data = {
        "id": seed_info["id"],
        "checked_tools": final_state["checked_tools"],
        "policy": final_state["policy_str"],
        "tasks_and_backgrounds": final_state["tasks_and_backgrounds"],
    }

    if not final_state["breaked"]:
        with log_file_lock:
            with open(virtual_tool_use_task_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(save_data, ensure_ascii=False) + '\n')

    return final_state
