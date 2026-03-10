import re
import copy
from typing import List, Dict, Optional
from infer_utils.functions.call_llms import call_llm_messages


def solve_task_by_tools(cfg, solve_history):
    solve_history = copy.deepcopy(solve_history)
    messages = call_llm_messages(
        messages=solve_history,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    think_and_tool_call = messages[-1]["content"]
    tool_call_matches = re.findall(r"<tool_call>(.+?)</tool_call>", think_and_tool_call, re.DOTALL)
    if tool_call_matches:
        last_match = tool_call_matches[-1]
        tool_call = last_match.strip()
    else:
        tool_call = None

    return think_and_tool_call, tool_call
