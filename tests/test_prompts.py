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

"""Tests for prompt loading, resolution, and externalization.

These tests ensure that every operator/evaluator loads its default prompts from
``configs/prompts/`` and that the resolution helpers obey the same priority
everywhere.
"""

import re
from pathlib import Path

import pytest
import yaml

from easydistill.eval import (
    CoTEvaluator,
    InstructionFollowingEvaluator,
    MMCoTEvaluator,
    MMInstructionFollowingEvaluator,
    T2IImageEvaluator,
    t2i_multi_model,
    t2i_single_model,
    ti2i_multi_model,
    ti2i_single_model,
)
from easydistill.eval.t2v_checkers import OmniChecker, VLMChecker
from easydistill.operators.agent import (
    AgentFuzzyTaskOperator,
    AgentRubricOperator,
    AgentTaskSynthesisOperator,
    AgentToolCheckOperator,
    AgentTrajectoryOperator,
)
from easydistill.operators.cot import CoTGenerationOperator
from easydistill.operators.mm import MMCoTGenerationOperator, MMGenerationOperator
from easydistill.operators.t2i import T2IPromptOptimizer
from easydistill.operators.t2v import T2VComposeStage, T2VExtractStage
from easydistill.prompts import (
    load_prompt_template_from_file,
    load_prompts_from_file,
    resolve_prompt,
    resolve_prompts,
)
from easydistill.rewrite import (
    CoTLong2ShortOperator,
    CoTShort2LongOperator,
    InstructionExpansionOperator,
    InstructionRefinementOperator,
    InstructionResponseExtractionOperator,
    MMCoTLong2ShortOperator,
    MMCoTShort2LongOperator,
)
from tests._fake_backend import FakeBackend

PROMPTS_DIR = Path("configs/prompts")
REPO_ROOT = Path(__file__).resolve().parents[1]

_TEMPLATE_OPERATORS = [
    (InstructionExpansionOperator, "configs/prompts/expansion_prompt.txt"),
    (InstructionRefinementOperator, "configs/prompts/refinement_prompt.txt"),
    (InstructionResponseExtractionOperator, "configs/prompts/extraction_prompt.txt"),
    (CoTGenerationOperator, "configs/prompts/cot_generation_prompt.txt"),
    (CoTLong2ShortOperator, "configs/prompts/cot_long2short_prompt.txt"),
    (CoTShort2LongOperator, "configs/prompts/cot_short2long_prompt.txt"),
    (MMCoTLong2ShortOperator, "configs/prompts/cot_long2short_prompt.txt"),
    (MMCoTShort2LongOperator, "configs/prompts/cot_short2long_prompt.txt"),
    (MMCoTGenerationOperator, "configs/prompts/cot_generation_prompt.txt"),
    (AgentTaskSynthesisOperator, "configs/prompts/agent_task_synthesis_prompt.txt"),
    (AgentFuzzyTaskOperator, "configs/prompts/agent_fuzzy_task_prompt.txt"),
    (AgentToolCheckOperator, "configs/prompts/agent_tool_check_prompt.txt"),
    (AgentRubricOperator, "configs/prompts/agent_rubrics_prompt.txt"),
    (T2IPromptOptimizer, "configs/prompts/t2i_prompt_optimize_prompt.txt"),
    (T2VExtractStage, "configs/prompts/t2v_extract_prompt.txt"),
    (T2VComposeStage, "configs/prompts/t2v_compose_prompt.txt"),
    (VLMChecker, "configs/prompts/t2v_vlm_judge_prompt.txt"),
    (OmniChecker, "configs/prompts/t2v_omni_judge_prompt.txt"),
]

_SYSTEM_PROMPT_OPERATORS = [
    (MMGenerationOperator, "system_prompt", "configs/prompts/mm_generation_prompt.txt"),
    (
        AgentTrajectoryOperator,
        "solve_system_prompt_template",
        "configs/prompts/agent_solve_system_prompt.txt",
    ),
    (
        AgentTrajectoryOperator,
        "mock_tool_prompt_template",
        "configs/prompts/agent_mock_tool_prompt.txt",
    ),
    (
        AgentTrajectoryOperator,
        "mock_user_prompt_template",
        "configs/prompts/agent_mock_user_prompt.txt",
    ),
    (
        T2VExtractStage,
        "i2v_prompt_template",
        "configs/prompts/i2v_extract_prompt.txt",
    ),
]

_EVALUATORS = [
    (InstructionFollowingEvaluator, "configs/prompts/default_eval_prompts.yaml"),
    (CoTEvaluator, "configs/prompts/default_cot_eval_prompts.yaml"),
    (MMInstructionFollowingEvaluator, "configs/prompts/default_eval_prompts.yaml"),
    (MMCoTEvaluator, "configs/prompts/default_cot_eval_prompts.yaml"),
    (T2IImageEvaluator, "configs/prompts/t2i_eval_prompts.yaml"),
]

# Standalone single-file evaluators (T2I/TI2I) resolve their own default
# prompts via module-level DEFAULT_PROMPTS_FILE / REQUIRED_PROMPTS constants.
_STANDALONE_EVAL_MODULES = [
    t2i_multi_model,
    t2i_single_model,
    ti2i_multi_model,
    ti2i_single_model,
]


def _standalone_prompts_path(module) -> str:
    return str(Path(module.DEFAULT_PROMPTS_FILE).resolve().relative_to(REPO_ROOT))


def _read_template(path: str) -> str:
    return load_prompt_template_from_file(path).rstrip("\n")


def _read_prompts(path: str) -> dict:
    raw = load_prompts_from_file(path)
    return {k: (v.rstrip("\n") if isinstance(v, str) else v) for k, v in raw.items()}


class TestResolvePrompt:
    """Priority and edge-case coverage for resolve_prompt."""

    def test_default_file_is_used_when_nothing_configured(self, tmp_path):
        default_file = tmp_path / "default.txt"
        default_file.write_text("default prompt\n")
        assert resolve_prompt({}, default_file=str(default_file)) == "default prompt"

    def test_inline_template_overrides_default_file(self, tmp_path):
        default_file = tmp_path / "default.txt"
        default_file.write_text("default prompt\n")
        assert (
            resolve_prompt({"prompt_template": "inline"}, default_file=str(default_file))
            == "inline"
        )

    def test_template_file_overrides_inline(self, tmp_path):
        template_file = tmp_path / "template.txt"
        template_file.write_text("from file\n")
        assert (
            resolve_prompt(
                {"prompt_template": "inline", "prompt_template_file": str(template_file)}
            )
            == "from file"
        )

    def test_none_inline_falls_back_to_default_file(self, tmp_path):
        default_file = tmp_path / "default.txt"
        default_file.write_text("default prompt\n")
        assert (
            resolve_prompt({"prompt_template": None}, default_file=str(default_file))
            == "default prompt"
        )

    def test_missing_default_file_raises(self, tmp_path):
        missing = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            resolve_prompt({}, default_file=str(missing))

    def test_loaded_files_have_trailing_newline_stripped(self, tmp_path):
        template_file = tmp_path / "template.txt"
        template_file.write_text("prompt\n")
        assert resolve_prompt({"prompt_template_file": str(template_file)}) == "prompt"


class TestResolvePrompts:
    """Priority and edge-case coverage for resolve_prompts."""

    def test_default_file_is_used_when_nothing_configured(self, tmp_path):
        default_file = tmp_path / "default.yaml"
        default_file.write_text(yaml.dump({"m1": "p1\n", "m2": "p2\n"}))
        prompts = resolve_prompts({}, default_file=str(default_file))
        assert prompts == {"m1": "p1", "m2": "p2"}

    def test_inline_prompts_override_default_file(self, tmp_path):
        default_file = tmp_path / "default.yaml"
        default_file.write_text(yaml.dump({"m1": "p1\n"}))
        prompts = resolve_prompts({"prompts": {"m1": "inline"}}, default_file=str(default_file))
        assert prompts == {"m1": "inline"}

    def test_inline_prompts_override_prompts_file(self, tmp_path):
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(yaml.dump({"m1": "from file\n"}))
        prompts = resolve_prompts({"prompts": {"m1": "inline"}, "prompts_file": str(prompts_file)})
        assert prompts == {"m1": "inline"}

    def test_inline_prompts_extend_default_file(self, tmp_path):
        default_file = tmp_path / "default.yaml"
        default_file.write_text(yaml.dump({"m1": "p1\n"}))
        prompts = resolve_prompts({"prompts": {"m2": "p2"}}, default_file=str(default_file))
        assert prompts == {"m1": "p1", "m2": "p2"}

    def test_loaded_files_have_trailing_newline_stripped(self, tmp_path):
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(yaml.dump({"m1": "p1\n"}))
        prompts = resolve_prompts({"prompts_file": str(prompts_file)})
        assert prompts == {"m1": "p1"}


class TestOperatorDefaultPrompts:
    """Every template operator loads its default from the expected config file."""

    @pytest.mark.parametrize("operator_cls, prompt_file", _TEMPLATE_OPERATORS)
    def test_default_template_matches_config_file(self, operator_cls, prompt_file):
        backend = FakeBackend(response_template="{}")
        operator = operator_cls(backend=backend, config={"show_progress": False})
        expected = _read_template(prompt_file)
        assert operator.prompt_template == expected
        assert operator.prompt_template is not None
        assert operator.prompt_template != ""

    @pytest.mark.parametrize("operator_cls, attr, prompt_file", _SYSTEM_PROMPT_OPERATORS)
    def test_default_system_prompt_matches_config_file(self, operator_cls, attr, prompt_file):
        backend = FakeBackend(response_template="{}")
        operator = operator_cls(backend=backend, config={"show_progress": False, "max_workers": 1})
        expected = _read_template(prompt_file)
        assert getattr(operator, attr) == expected


class TestEvaluatorDefaultPrompts:
    """Every evaluator loads its default metric prompts from the expected YAML file."""

    @pytest.mark.parametrize("evaluator_cls, prompts_file", _EVALUATORS)
    def test_default_prompts_match_config_file(self, evaluator_cls, prompts_file):
        backend = FakeBackend(response_template="{}")
        evaluator = evaluator_cls(backend=backend, config={"show_progress": False})
        expected = _read_prompts(prompts_file)
        assert evaluator.prompts == expected
        assert evaluator.metrics == list(expected.keys())


class TestStandaloneEvaluatorPrompts:
    """T2I/TI2I single-file evaluators ship complete default prompt files."""

    @pytest.mark.parametrize(
        "module", _STANDALONE_EVAL_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1]
    )
    def test_default_prompts_file_has_required_keys(self, module):
        prompts = _read_prompts(str(module.DEFAULT_PROMPTS_FILE))
        for key in module.REQUIRED_PROMPTS:
            assert prompts.get(key), f"missing prompt '{key}' in {module.DEFAULT_PROMPTS_FILE}"


class TestPromptFilesCoverage:
    """All prompt files under configs/prompts/ are referenced by at least one component."""

    @staticmethod
    def _config_referenced_prompts() -> set:
        """Prompt paths referenced from YAML configs under configs/."""
        referenced = set()
        pattern = re.compile(r"configs/prompts/[\w./-]+\.(?:txt|yaml)")
        for config_file in Path("configs").rglob("*.yaml"):
            referenced.update(pattern.findall(config_file.read_text()))
        return referenced

    def test_no_orphan_prompt_files(self):
        referenced = {path for _, path in _TEMPLATE_OPERATORS}
        referenced.update(path for _, _, path in _SYSTEM_PROMPT_OPERATORS)
        referenced.update(path for _, path in _EVALUATORS)
        referenced.update(_standalone_prompts_path(m) for m in _STANDALONE_EVAL_MODULES)
        referenced.update(self._config_referenced_prompts())

        for prompt_file in sorted(PROMPTS_DIR.glob("*.txt")):
            assert str(prompt_file) in referenced, f"orphan template file: {prompt_file}"

        for prompt_file in sorted(PROMPTS_DIR.glob("*.yaml")):
            assert str(prompt_file) in referenced, f"orphan prompts file: {prompt_file}"


class TestPromptPackage:
    """The prompts package no longer exposes prompt text constants."""

    def test_public_api_only_exports_loaders(self):
        from easydistill.prompts import __all__

        assert set(__all__) == {
            "load_prompt_template_from_file",
            "load_prompts_from_file",
            "resolve_prompt",
            "resolve_prompts",
        }

    def test_removed_constants_no_longer_exported(self):
        import easydistill.prompts as prompts

        assert not hasattr(prompts, "EXPANSION_PROMPT")
        assert not hasattr(prompts, "DEFAULT_EVAL_PROMPTS")
