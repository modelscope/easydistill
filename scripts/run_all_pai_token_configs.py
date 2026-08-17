#!/usr/bin/env python3
# Copyright 2026 Alibaba Group Holding Limited. All Rights Reserved.
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

"""Run every PAI-Token YAML config end-to-end and report pass/fail status.

Usage:
    # Full run (uses all samples in each config's input file)
    python scripts/run_all_pai_token_configs.py

    # Smoke run: 1 sample, max_workers=1, temp outputs
    python scripts/run_all_pai_token_configs.py --smoke

Required environment variables:
    PAI_TOKEN_API_KEY
    PAI_TOKEN_BASE_URL (defaults to https://cn-beijing.pai-token.aliyuncs.com/v1)
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_BASE_URL = "https://cn-beijing.pai-token.aliyuncs.com/v1"


def find_pai_token_configs() -> List[Path]:
    """Return all *_pai_token.yaml configs sorted by path."""
    return sorted(CONFIG_DIR.rglob("*_pai_token.yaml"))


def _set_max_workers(obj: Any, value: int) -> None:
    """Recursively set max_workers to ``value`` wherever it appears."""
    if isinstance(obj, dict):
        if "max_workers" in obj:
            obj["max_workers"] = value
        for v in obj.values():
            _set_max_workers(v, value)
    elif isinstance(obj, list):
        for item in obj:
            _set_max_workers(item, value)


def _reduce_agent_scale(obj: Any) -> None:
    """Recursively reduce agent trajectory scale to avoid smoke-test timeouts."""
    if isinstance(obj, dict):
        for key in ("max_steps", "repeat_times", "max_tool_calls"):
            if key in obj and isinstance(obj[key], int) and obj[key] > 1:
                obj[key] = 1
        for v in obj.values():
            _reduce_agent_scale(v)
    elif isinstance(obj, list):
        for item in obj:
            _reduce_agent_scale(item)


def _rewrite_output_paths(obj: Any, prefix: str) -> None:
    """Recursively rewrite output_path values to temp files in outputs/."""
    if isinstance(obj, dict):
        if "output_path" in obj and isinstance(obj["output_path"], str):
            original = Path(obj["output_path"])
            obj["output_path"] = str(OUTPUT_DIR / f"{prefix}_{original.name}")
        for v in obj.values():
            _rewrite_output_paths(v, prefix)
    elif isinstance(obj, list):
        for item in obj:
            _rewrite_output_paths(item, prefix)


def _truncate_input(obj: Dict[str, Any], max_samples: int) -> None:
    """Truncate the top-level dataset input to ``max_samples`` lines."""
    dataset = obj.get("dataset")
    if not isinstance(dataset, dict):
        return
    input_path_str = dataset.get("input_path")
    if not isinstance(input_path_str, str):
        return
    input_path = REPO_ROOT / input_path_str
    if not input_path.exists():
        return
    lines = input_path.read_text().strip().split("\n")
    if len(lines) <= max_samples:
        return
    fd, tmp_input = tempfile.mkstemp(
        suffix=".jsonl", dir=OUTPUT_DIR, prefix=f"smoke_{input_path.stem}_"
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:max_samples]) + "\n")
    dataset["input_path"] = str(Path(tmp_input).relative_to(REPO_ROOT))


def prepare_smoke_config(original_path: Path, max_samples: int) -> Path:
    """Create a temporary smoke-test config with reduced scale."""
    with open(original_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prefix = f"smoke_{original_path.stem}"
    _set_max_workers(config, 1)
    _reduce_agent_scale(config)
    _rewrite_output_paths(config, prefix)
    _truncate_input(config, max_samples)

    fd, tmp_config = tempfile.mkstemp(suffix=".yaml", dir=OUTPUT_DIR, prefix=f"{prefix}_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return Path(tmp_config)


def run_config(config_path: Path, smoke: bool, max_samples: int) -> bool:
    """Execute a single config and return True on success."""
    temp_input_path: Optional[Path] = None
    if smoke:
        actual_path = prepare_smoke_config(config_path, max_samples)
        dataset = yaml.safe_load(actual_path.read_text(encoding="utf-8")).get("dataset")
        if isinstance(dataset, dict) and isinstance(dataset.get("input_path"), str):
            candidate = REPO_ROOT / dataset["input_path"]
            if candidate != config_path and candidate.name.startswith("smoke_"):
                temp_input_path = candidate
    else:
        actual_path = config_path

    rel_path = config_path.relative_to(REPO_ROOT)
    print(f"\n{'=' * 70}")
    print(f"Running: {rel_path}")
    if smoke:
        print(f"Using smoke config: {actual_path.relative_to(REPO_ROOT)}")
    print(f"{'=' * 70}")

    env = os.environ.copy()
    env.setdefault("PAI_TOKEN_BASE_URL", DEFAULT_BASE_URL)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "easydistill.cli", "--config", str(actual_path)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600 if smoke else 3600,
        )
        if proc.returncode == 0:
            print("SUCCESS")
            return True
        print("FAILED")
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return False
    finally:
        if smoke and actual_path.exists() and actual_path != config_path:
            actual_path.unlink(missing_ok=True)
        if temp_input_path and temp_input_path.exists():
            temp_input_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all PAI-Token configs end-to-end."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode: 1 sample per config, max_workers=1, temp outputs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help=(
            "Number of input samples to use per config in smoke mode (default: 3). "
            "Set to at least 3 for instruction_expansion configs, which need in-context examples."
        ),
    )
    args = parser.parse_args()

    configs = find_pai_token_configs()
    print(f"Found {len(configs)} PAI-Token config(s).")

    if not os.environ.get("PAI_TOKEN_API_KEY"):
        print(
            "WARNING: PAI_TOKEN_API_KEY is not set. Set it before running this script.",
            file=sys.stderr,
        )

    results: Dict[Path, bool] = {}
    for config in configs:
        results[config] = run_config(config, args.smoke, args.max_samples)

    passed = sum(results.values())
    total = len(results)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for config, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {config.relative_to(REPO_ROOT)}")
    print(f"\nTotal: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
