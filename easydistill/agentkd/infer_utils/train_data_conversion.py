import json
import os
import random


def conversion_tool_use(solve_path, output_path, val_split=100):
    """Convert PASS-filtered virtual tool-use solutions into SFT training data.

    Scans ``solve_path`` for task directories that contain a
    ``rubrics_output.json`` with at least one PASS verdict. For the best
    passing solution, the ``solve_history`` (system / user / assistant messages)
    is converted into the ``conversations`` format expected by ``train.py``.

    Args:
        solve_path: Root directory containing per-task solution folders.
        output_path: JSON file path to write the training data.
        val_split: Number of samples to hold out for a validation split.
    """
    saves = []
    skipped_no_rubrics = 0
    skipped_no_pass = 0

    for task_id in sorted(os.listdir(solve_path)):
        task_dir = os.path.join(solve_path, task_id)
        if not os.path.isdir(task_dir):
            continue

        rubrics_path = os.path.join(task_dir, "rubrics_output.json")
        more_info_path = os.path.join(task_dir, "more_info.json")

        if not os.path.exists(rubrics_path) or not os.path.exists(more_info_path):
            skipped_no_rubrics += 1
            continue

        with open(rubrics_path, 'r', encoding='utf-8') as f:
            rubrics_output = json.load(f)

        if rubrics_output.get("pass_count", 0) == 0:
            skipped_no_pass += 1
            continue

        best_solution_name = rubrics_output.get("best_solution")
        if not best_solution_name:
            skipped_no_pass += 1
            continue

        best_solution_path = os.path.join(task_dir, best_solution_name)
        if not os.path.exists(best_solution_path):
            skipped_no_pass += 1
            continue

        with open(best_solution_path, 'r', encoding='utf-8') as f:
            solution = json.load(f)

        # Convert solve_history messages into conversations format
        conversations = []
        for msg in solution:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                conversations.append({"from": "system", "value": content})
            elif role == "user":
                conversations.append({"from": "human", "value": content})
            elif role == "assistant":
                conversations.append({"from": "gpt", "value": content})

        # Remove trailing human turn (e.g. ###STOP from mock user) so SFT ends with assistant response
        while conversations and conversations[-1].get("from") == "human":
            conversations.pop()

        if not conversations:
            continue

        saves.append({"conversations": conversations})

    print(f"Total PASS: {len(saves)}, skipped (no rubrics): {skipped_no_rubrics}, skipped (no PASS): {skipped_no_pass}")

    random.shuffle(saves)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if val_split > 0 and len(saves) > val_split:
        val_data = saves[:val_split]
        train_data = saves[val_split:]

        val_output = output_path.replace(".json", "_val.json")
        with open(val_output, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=4)
        print(f"Validation set: {len(val_data)} -> {val_output}")
    else:
        train_data = saves

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    print(f"Training set: {len(train_data)} -> {output_path}")