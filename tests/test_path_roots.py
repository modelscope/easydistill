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

"""Tests for resolving relative config data paths against INPUT_ROOT/OUTPUT_ROOT.

The point of the feature is that the same config runs unchanged in a container
where the data is mounted somewhere else. So the two properties worth pinning
down are that an unset root changes nothing at all, and that a set root reaches
paths wherever they are nested -- including the per-stage ``config`` blocks and
the backend sections, which a positional walk would miss.

The roots are also a foot-gun: prefixing the wrong key would rewrite an HTTP
route or a packaged prompt template into nonsense, so the exclusions are
asserted as explicitly as the inclusions.
"""

from pathlib import Path

import pytest
import yaml

from easydistill.utils import (
    ENV_INPUT_ROOT,
    ENV_OUTPUT_ROOT,
    load_expanded_config,
    resolve_path_roots,
)


def set_roots(monkeypatch, input_root=None, output_root=None):
    """Export only the roots the test cares about, clearing the others."""
    for name, value in ((ENV_INPUT_ROOT, input_root), (ENV_OUTPUT_ROOT, output_root)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def joined(*parts):
    """The expected join, spelled the way the resolver spells it."""
    return str(Path(*parts))


# --------------------------------------------------------------------------
# The roots are read independently
# --------------------------------------------------------------------------


def test_no_root_leaves_the_config_untouched(monkeypatch):
    set_roots(monkeypatch)
    cfg = {"dataset": {"input_path": "data/in.jsonl", "output_path": "out/o.jsonl"}}

    resolved = resolve_path_roots(cfg)

    assert resolved["dataset"]["input_path"] == "data/in.jsonl"
    assert resolved["dataset"]["output_path"] == "out/o.jsonl"


def test_input_root_alone_leaves_outputs_relative(monkeypatch):
    set_roots(monkeypatch, input_root="/mnt/in")
    cfg = {"dataset": {"input_path": "data/in.jsonl", "output_path": "out/o.jsonl"}}

    resolved = resolve_path_roots(cfg)

    assert resolved["dataset"]["input_path"] == joined("/mnt/in", "data/in.jsonl")
    assert resolved["dataset"]["output_path"] == "out/o.jsonl"


def test_output_root_alone_leaves_inputs_relative(monkeypatch):
    set_roots(monkeypatch, output_root="/mnt/out")
    cfg = {"dataset": {"input_path": "data/in.jsonl", "output_path": "out/o.jsonl"}}

    resolved = resolve_path_roots(cfg)

    assert resolved["dataset"]["input_path"] == "data/in.jsonl"
    assert resolved["dataset"]["output_path"] == joined("/mnt/out", "out/o.jsonl")


def test_both_roots_resolve_their_own_side(monkeypatch):
    set_roots(monkeypatch, input_root="/mnt/in", output_root="/mnt/out")
    cfg = {"dataset": {"input_path": "data/in.jsonl", "output_path": "out/o.jsonl"}}

    resolved = resolve_path_roots(cfg)

    assert resolved["dataset"]["input_path"] == joined("/mnt/in", "data/in.jsonl")
    assert resolved["dataset"]["output_path"] == joined("/mnt/out", "out/o.jsonl")


def test_a_blank_root_counts_as_unset(monkeypatch):
    # Exporting INPUT_ROOT= is how a wrapper script switches the feature off;
    # joining onto "" would silently turn the path into a bare relative one.
    set_roots(monkeypatch, input_root="   ")
    cfg = {"dataset": {"input_path": "data/in.jsonl"}}

    assert resolve_path_roots(cfg)["dataset"]["input_path"] == "data/in.jsonl"


# --------------------------------------------------------------------------
# What gets rewritten, and where it may hide
# --------------------------------------------------------------------------


def test_an_absolute_path_is_already_complete(monkeypatch):
    set_roots(monkeypatch, input_root="/mnt/in", output_root="/mnt/out")
    cfg = {"dataset": {"input_path": "/abs/in.jsonl", "output_path": "/abs/o.jsonl"}}

    resolved = resolve_path_roots(cfg)

    assert resolved["dataset"]["input_path"] == "/abs/in.jsonl"
    assert resolved["dataset"]["output_path"] == "/abs/o.jsonl"


def test_an_empty_path_has_nothing_to_resolve(monkeypatch):
    set_roots(monkeypatch, input_root="/mnt/in")

    assert resolve_path_roots({"dataset": {"input_path": ""}})["dataset"]["input_path"] == ""


def test_paths_nested_in_pipeline_stages_are_reached(monkeypatch):
    set_roots(monkeypatch, input_root="/mnt/in", output_root="/mnt/out")
    cfg = {
        "pipeline": [
            {
                "stage": "tool_crop",
                "output_path": "stage/out.jsonl",
                "config": {"work_dir": "outputs/crops", "input_path": "stage/in.jsonl"},
            }
        ]
    }

    stage = resolve_path_roots(cfg)["pipeline"][0]

    assert stage["output_path"] == joined("/mnt/out", "stage/out.jsonl")
    assert stage["config"]["work_dir"] == joined("/mnt/out", "outputs/crops")
    assert stage["config"]["input_path"] == joined("/mnt/in", "stage/in.jsonl")


def test_the_system1_schema_path_follows_the_input_root(monkeypatch):
    # The build stage reads the workflow schema alongside the dataset, so it
    # belongs on the input side even though the key is not named input_path.
    set_roots(monkeypatch, input_root="/mnt/in")
    cfg = {
        "pipeline": [
            {"stage": "build_cases", "config": {"schema_path": "schemas/sms.yaml"}}
        ]
    }

    stage = resolve_path_roots(cfg)["pipeline"][0]

    assert stage["config"]["schema_path"] == joined("/mnt/in", "schemas/sms.yaml")


def test_backend_output_dirs_are_reached(monkeypatch):
    # Downloaded media lands here, so it belongs on the output side even though
    # the key is not named output_path.
    set_roots(monkeypatch, output_root="/mnt/out")
    cfg = {"t2i_backend": {"type": "pai_token", "output_dir": "images"}}

    resolved = resolve_path_roots(cfg)

    assert resolved["t2i_backend"]["output_dir"] == joined("/mnt/out", "images")


def test_stream_output_path_is_an_output(monkeypatch):
    set_roots(monkeypatch, output_root="/mnt/out")
    cfg = {"dataset": {"stream_output_path": "stream.jsonl"}}

    resolved = resolve_path_roots(cfg)

    assert resolved["dataset"]["stream_output_path"] == joined("/mnt/out", "stream.jsonl")


def test_output_dir_on_the_dataset_is_an_output(monkeypatch):
    set_roots(monkeypatch, output_root="/mnt/out")
    cfg = {"dataset": {"output_dir": "shards"}}

    assert resolve_path_roots(cfg)["dataset"]["output_dir"] == joined("/mnt/out", "shards")


def test_videos_dir_on_the_dataset_is_an_output(monkeypatch):
    # T2V downloads finished videos here; the key sits on the dataset but is
    # still a written directory, so it has to follow OUTPUT_ROOT.
    set_roots(monkeypatch, output_root="/mnt/out")
    cfg = {"dataset": {"videos_dir": "outputs/t2v_videos"}}

    assert resolve_path_roots(cfg)["dataset"]["videos_dir"] == joined(
        "/mnt/out", "outputs/t2v_videos"
    )


def test_the_search_cache_follows_the_output_root(monkeypatch):
    # The SQLite cache is written, and it sits two levels down under the stage's
    # tools block, so only the recursive walk reaches it.
    set_roots(monkeypatch, output_root="/mnt/out")
    cfg = {
        "pipeline": [
            {
                "stage": "search_agent_trajectory",
                "config": {"tools": {"mode": "real", "cache_db_path": "outputs/search_cache.db"}},
            }
        ]
    }

    tools = resolve_path_roots(cfg)["pipeline"][0]["config"]["tools"]

    assert tools["cache_db_path"] == joined("/mnt/out", "outputs/search_cache.db")
    assert tools["mode"] == "real"


def test_a_resume_checkpoint_follows_the_output_root(monkeypatch):
    # The T2V generator appends completed rows here, so a run that is resumed
    # has to find the file in the same place it wrote it.
    set_roots(monkeypatch, output_root="/mnt/out")
    cfg = {"generation": {"checkpoint_path": "outputs/t2v_rows.jsonl"}}

    resolved = resolve_path_roots(cfg)["generation"]["checkpoint_path"]

    assert resolved == joined("/mnt/out", "outputs/t2v_rows.jsonl")


# --------------------------------------------------------------------------
# What must never be rewritten
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        # An HTTP route appended to a base URL; a filesystem join breaks the URL.
        ("submit_path", "/api/v1/submit"),
        # These point at templates bundled in the package and have their own
        # disk-first, packaged-fallback resolution.
        ("prompt_template_file", "configs/prompts/t2i.yaml"),
        ("prompts_file", "configs/prompts/list.yaml"),
        ("scene_prompt_dir", "configs/prompts/scenes"),
        ("system_prompt_zh_file", "configs/prompts/sys_zh.txt"),
        ("system_prompt_en_file", "configs/prompts/sys_en.txt"),
        ("dimensions_path", "configs/dimensions.json"),
    ],
)
def test_excluded_keys_survive_both_roots(monkeypatch, key, value):
    set_roots(monkeypatch, input_root="/mnt/in", output_root="/mnt/out")

    resolved = resolve_path_roots({"stage": {key: value}})

    assert resolved["stage"][key] == value


def test_non_string_values_are_left_alone(monkeypatch):
    # A null output_path is how a stage says "no output"; it must stay null
    # rather than become the root itself.
    set_roots(monkeypatch, output_root="/mnt/out")
    cfg = {"dataset": {"output_path": None, "max_samples": 10, "shuffle": True}}

    resolved = resolve_path_roots(cfg)

    assert resolved["dataset"]["output_path"] is None
    assert resolved["dataset"]["max_samples"] == 10
    assert resolved["dataset"]["shuffle"] is True


def test_the_original_config_is_not_mutated(monkeypatch):
    set_roots(monkeypatch, input_root="/mnt/in")
    cfg = {"dataset": {"input_path": "data/in.jsonl"}}

    resolve_path_roots(cfg)

    assert cfg["dataset"]["input_path"] == "data/in.jsonl"


# --------------------------------------------------------------------------
# End to end, through the loader every runner uses
# --------------------------------------------------------------------------


def write_config(tmp_path, cfg):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_load_expanded_config_resolves_against_the_roots(tmp_path, monkeypatch):
    input_root = tmp_path / "in"
    (input_root / "data").mkdir(parents=True)
    (input_root / "data" / "seed.jsonl").write_text("{}\n", encoding="utf-8")
    set_roots(monkeypatch, input_root=str(input_root), output_root=str(tmp_path / "out"))
    config_path = write_config(
        tmp_path,
        {
            "backend": {"type": "openai"},
            "dataset": {"input_path": "data/seed.jsonl", "output_path": "res.jsonl"},
        },
    )

    cfg = load_expanded_config(config_path)

    assert cfg["dataset"]["input_path"] == str(input_root / "data" / "seed.jsonl")
    assert cfg["dataset"]["output_path"] == str(tmp_path / "out" / "res.jsonl")


def test_a_placeholder_inside_a_path_is_expanded_before_the_join(tmp_path, monkeypatch):
    input_root = tmp_path / "in"
    (input_root / "v2").mkdir(parents=True)
    (input_root / "v2" / "seed.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("EASYDISTILL_TEST_VERSION", "v2")
    set_roots(monkeypatch, input_root=str(input_root))
    config_path = write_config(
        tmp_path,
        {
            "backend": {"type": "openai"},
            "dataset": {"input_path": "${EASYDISTILL_TEST_VERSION}/seed.jsonl"},
        },
    )

    cfg = load_expanded_config(config_path)

    assert cfg["dataset"]["input_path"] == str(input_root / "v2" / "seed.jsonl")


def test_a_missing_input_under_a_root_names_the_root(tmp_path, monkeypatch):
    # A wrong mount and a genuinely absent dataset produce the same missing
    # path, so the message has to say which root it was resolved against.
    input_root = tmp_path / "in"
    input_root.mkdir()
    set_roots(monkeypatch, input_root=str(input_root))
    config_path = write_config(
        tmp_path,
        {"backend": {"type": "openai"}, "dataset": {"input_path": "absent.jsonl"}},
    )

    with pytest.raises(ValueError) as excinfo:
        load_expanded_config(config_path)

    message = str(excinfo.value)
    assert ENV_INPUT_ROOT in message
    assert str(input_root) in message
    assert str(input_root / "absent.jsonl") in message


def test_a_missing_input_without_a_root_reports_only_the_path(tmp_path, monkeypatch):
    set_roots(monkeypatch)
    config_path = write_config(
        tmp_path,
        {"backend": {"type": "openai"}, "dataset": {"input_path": "absent.jsonl"}},
    )

    with pytest.raises(ValueError) as excinfo:
        load_expanded_config(config_path)

    message = str(excinfo.value)
    assert "absent.jsonl" in message
    assert ENV_INPUT_ROOT not in message
