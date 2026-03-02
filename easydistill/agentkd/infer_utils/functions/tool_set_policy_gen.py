import os
import re
import json
from infer_utils.functions.call_llms import call_llm_api

_EXAMPLE_JSON_PATH = os.path.join(os.path.dirname(__file__), "example.json")
with open(_EXAMPLE_JSON_PATH, "r") as f:
    tool_call_example = json.load(f)

tool_set_prompt = """You are an expert capable of creating complex tool-use tasks and designing executable strategic rules.

Task Objective:  
Based on the provided background information, design a complex but realistically feasible task that requires at least two different tools working together.  

You must also design several virtual tools and define their input/output schemas.  
Note: The completion of the tool-calling task will be judged by checking the state changes before and after tool execution. Therefore, ensure that the tools and tasks you design involve state‑modifying or otherwise objectively verifiable actions. (For example, booking a flight can be verified by checking database changes, but writing a research report cannot be objectively evaluated.)

Finally, produce a tree-based policy that describes the agent's allowed behaviors, prohibited actions, tool preconditions, and refusal rules across different scenarios.

Notes:  
This task is *not* about designing a "guaranteed executable workflow," but about designing a "policy tree with multiple scenario branches, constraints, and conditional logic."  
In some scenarios, the agent should be able to execute the task; in others, it must refuse, avoid tool invocation, or follow strict tool‑order requirements.

Complexity Requirements:

1. Before constructing the task, you must guarantee that the task can be evaluated by the state changes objectively, and require 3–5 different categories of tools working together.

2. Tools must be based on real-world technologies. They may be hypothetical virtual tools, but their functionality must be clear and limited in scope.
The types of tools should include writing to a database, querying information, deleting information, and modifying information, but should not include content‑generation APIs, because they make it difficult to evaluate whether the task has been completed.

3. Tools must include explicit input/output structures (field names, types, descriptions).

4. You must construct a *tree‑based policy* for the task, including:
   - allowed actions (operations allowed when certain conditions are met)
   - disallowed actions
   - tool preconditions (requirements before a tool can be used)
   - refusal-required conditions
   - transfer-required conditions (when escalation to a human is necessary)
   - cross-policy dependencies

Note that you should not impose any restrictions on the order of task completion process,  or which tools must be called.

5. The policy must be presented as a JSON tree containing multiple scenario branches, describing decision paths for the agent.

----------------------------------------
Reasoning Steps:

1. Background Analysis  
   - Infer possible user goals based on the background
   - List 2–3 feasible high-level task directions  
   - Choose the most complex one suited for multi-tool collaboration, and it can be evaluated by the state changes objectively.

2. Task Definition  
   - Describe the complex task: its goal, context, and key challenges

3. Tool List Design
Design 3–5 tools.  
The types of tools should include writing to a database, querying information, deleting information, and modifying information, but should not include content‑generation APIs, because they make it difficult to evaluate whether the task has been completed.
Each tool must include: name, description, parameters, outputs using this format:

{{
    "name": "tool_name",
    "description": "Brief description of the tool functionality.",
    "parameters": {{
        "type": "object",
        "properties": {{
            "parameter_name": {{
                "description": "Description of the parameter",
                "type": "string"
            }}
            // Additional input parameters can be added here
        }},
        "required": ["parameter_name"]
    }},
    "outputs": {{
        "type": "object",
        "properties": {{
            "output_field_name": {{
                "description": "Description of the output field",
                "type": "string"
            }}
            // Additional output fields can be added here
        }}
    }}
}}

IMPORTANT: The arguments of a tool should be less or equal to 3.

4. Tree-based Policy (Core Part)  
Produce a JSON policy tree containing:

Example tree based policy:

- "root_condition": top-level trigger condition  
- "allowed_actions"  
- "disallowed_actions"  
- "tool_preconditions"  
- "refusal_conditions"  
- "transfer_conditions"  
- "branches": a recursive JSON tree of scenario branches  
  Each branch must have:  
  - "condition"  
  - "next" (sub-branches)  
  - "outcome" (allowed / denied / ask for clarification)
- other actions that can make the task more complex

The policy does not need to cover every real-world scenario but must be sufficiently complex and multi-layered.

{{
  "root_condition": "User requests help with booking, modifying, canceling flights, or requesting refunds/compensation.",

  "allowed_actions": [
    "Ask for required identifiers (user id, reservation id).",
    "Collect required booking or modification details.",
    "Use booking/modify/cancel/refund tools after explicit user confirmation.",
    "Provide factual, policy-based information only.",
    "Transfer to human agent when the case is outside allowed actions."
  ],

  "disallowed_actions": [
    "Perform actions without explicit user confirmation before database updates.",
    "Provide subjective opinions or information not from tools or user input.",
    "Make simultaneous tool calls and user responses.",
    "Modify reservation passenger count.",
    "Modify basic economy flights.",
    "Add insurance after initial booking."
  ],

  "clarification_required": [
    "Missing user id.",
    "Missing reservation id (when modifying/canceling).",
    "Incomplete booking details (trip type, origin, destination, cabin, passengers).",
    "Missing cancellation reason.",
    "Missing payment method details."
  ],

  "tool_preconditions": {{
    "booking_tools": {{ "must_have": ["user id", "trip details", "passenger info", "payment method", "explicit confirmation"] }},
    "modify_tools": {{ "must_have": ["user id", "reservation id", "rule-compliant changes", "explicit confirmation"] }},
    "cancel_tools": {{ "must_have": ["user id", "reservation id", "valid cancellation condition", "explicit confirmation"] }},
    "refund_compensation_tools": {{ "must_have": ["verified facts", "eligible membership/ticket class"] }},
    "transfer_to_human_agents": {{ "must_have": ["task outside policy scope"] }}
  }},

  "refusal_conditions": [
    "User attempts action that violates fare rules (e.g., modify basic economy).",
    "User requests prohibited behavior (adding insurance post‑booking, removing bags, etc.).",
    "User requests subjective advice or information not allowed.",
    "User requests tool usage without required preconditions."
  ],

  "transfer_conditions": [
    "Flight already partially flown and user wants modification or cancellation.",
    "Cancellation request invalid but user insists.",
    "Passenger count modification requested.",
    "Policy conflict the agent cannot resolve."
  ],

  "branches": [

    {{
      "condition": "User requests flight booking",
      "action": "proceed",
      "next": [
        {{
          "condition": "Missing booking prerequisites (user id, trip details, passengers, payment)",
          "action": "clarify",
          "next": null
        }},
        {{
          "condition": "All prerequisites met",
          "action": "request_confirmation",
          "next": [
            {{
              "condition": "User confirms",
              "action": "tool_call: booking_tools",
              "next": null
            }},
            {{
              "condition": "User declines",
              "action": "abort",
              "next": null
            }}
          ]
        }}
      ]
    }},

    {{
      "condition": "User requests modification of an existing reservation",
      "action": "proceed",
      "next": [
        {{
          "condition": "Missing user id or reservation id",
          "action": "clarify",
          "next": null
        }},
        {{
          "condition": "User requests forbidden modification (basic economy flight change, passenger count change, etc.)",
          "action": "refuse",
          "next": null
        }},
        {{
          "condition": "Modification allowed by rules",
          "action": "request_confirmation",
          "next": [
            {{
              "condition": "User confirms",
              "action": "tool_call: modify_tools",
              "next": null
            }},
            {{
              "condition": "User declines",
              "action": "abort",
              "next": null
            }}
          ]
        }}
      ]
    }},

    {{
      "condition": "User requests flight cancellation",
      "action": "proceed",
      "next": [
        {{
          "condition": "Missing user id or reservation id",
          "action": "clarify",
          "next": null
        }},
        {{
          "condition": "Flight already flown",
          "action": "transfer",
          "next": null
        }},
        {{
          "condition": "Cancellation conditions allowed",
          "action": "request_confirmation",
          "next": [
            {{
              "condition": "User confirms",
              "action": "tool_call: cancel_tools",
              "next": null
            }},
            {{
              "condition": "User declines",
              "action": "abort",
              "next": null
            }}
          ]
        }},
        {{
          "condition": "Cancellation conditions not met",
          "action": "refuse",
          "next": null
        }}
      ]
    }},

    {{
      "condition": "User requests refunds or compensation",
      "action": "proceed",
      "next": [
        {{
          "condition": "User does not provide necessary facts",
          "action": "clarify",
          "next": null
        }},
        {{
          "condition": "User not eligible based on membership/ticket/insurance rules",
          "action": "refuse",
          "next": null
        }},
        {{
          "condition": "User eligible and facts confirmed",
          "action": "request_confirmation",
          "next": [
            {{
              "condition": "User confirms",
              "action": "tool_call: refund_compensation_tools",
              "next": null
            }},
            {{
              "condition": "User declines",
              "action": "abort",
              "next": null
            }}
          ]
        }}
      ]
    }},

    {{
      "condition": "User asks for something outside the policy scope",
      "action": "transfer",
      "next": null
    }}

  ]
}}

----------------------------------------
Example Task: {tool_call_example}
----------------------------------------

----------------------------------------
Background Information: {background_info}
----------------------------------------

----------------------------------------
Final Output Format (must follow strictly):

## Reasoning Step
<reasoning>step-by-step reasoning…</reasoning>

## 1. Task Description
<task>(task description)</task>

## 2. Tool List (JSON)
<tools>(tools JSON list), The arguments of a tool should be less or equal to 3.</tools>

## 3. Tree-based Policy (JSON)
<policy_tree>(policy tree JSON)</policy_tree>
"""


def generate_tool_set_policy(cfg, background_info):
    prompt = tool_set_prompt.format(background_info=background_info, tool_call_example=tool_call_example)
    messages = call_llm_api(
        user_prompt=prompt,
        system_prompt="",
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    all_content = messages[-1]["content"]

    all_content = re.sub(r"<reasoning>(.+?)</reasoning>", "", all_content, flags=re.DOTALL)

    tool_matches = re.findall(r"<tools>(.+?)</tools>", all_content, re.DOTALL)
    if tool_matches:
        last_match = tool_matches[-1]
        tools = last_match.strip()
    else:
        tools = None

    policy_matches = re.findall(r"<policy_tree>(.+?)</policy_tree>", all_content, re.DOTALL)
    if policy_matches:
        last_match = policy_matches[-1]
        policy = last_match.strip()
    else:
        policy = None

    task_matches = re.findall(r"<task>(.+?)</task>", all_content, re.DOTALL)
    if task_matches:
        last_match = task_matches[-1]
        task = last_match.strip()
    else:
        task = None

    return all_content, task, tools, policy
