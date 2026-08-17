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

"""Unit tests for the Model Zoo metadata and generator."""

import pytest

from easydistill.models.zoo import (
    format_model_table,
    list_datasets,
    list_models,
    load_model_zoo,
    render_model_zoo_markdown,
)


def test_load_model_zoo_has_models_and_datasets():
    zoo = load_model_zoo()
    assert "version" in zoo
    assert isinstance(zoo["models"], list)
    assert isinstance(zoo["datasets"], list)
    assert len(zoo["models"]) > 0
    assert len(zoo["datasets"]) > 0


def test_list_models_returns_distilqwen_models():
    models = list_models()
    names = {m["name"] for m in models}
    assert "DistilQwen2.5-7B-Instruct" in names
    assert "DistilQwen-ThoughtX-7B" in names
    assert "DistilQwen-ThoughtY-8B" in names


def test_list_datasets_returns_released_datasets():
    datasets = list_datasets()
    names = {d["name"] for d in datasets}
    assert "OmniThought" in names
    assert "OmniThoughtV_Filter_0.5M" in names
    assert "DistilQwen_1M" in names


def test_model_entries_have_required_fields():
    for model in list_models():
        assert "name" in model
        assert "family" in model
        assert "size" in model
        assert "type" in model
        assert "description" in model
        assert "downloads" in model
        assert "huggingface" in model["downloads"]
        assert "modelscope" in model["downloads"]


def test_format_model_table_contains_models():
    table = format_model_table()
    assert "Model Zoo" in table
    assert "DistilQwen2.5-7B-Instruct" in table
    assert "DistilQwen-ThoughtX-7B" in table


def test_render_model_zoo_markdown_contains_sections():
    markdown = render_model_zoo_markdown()
    assert "# EasyDistill 2 Model Zoo" in markdown
    assert "## Model summary" in markdown
    assert "## Models by family" in markdown
    assert "## Released datasets" in markdown
    assert "huggingface.co" in markdown
    assert "modelscope.cn" in markdown
    assert "auto-generated" in markdown


def test_render_model_zoo_markdown_chinese():
    markdown = render_model_zoo_markdown(lang="zh")
    assert "# EasyDistill 2 Model Zoo" in markdown
    assert "## 模型总览" in markdown
    assert "## 按家族分类" in markdown
    assert "## 已发布数据集" in markdown
    assert "**能力：**" in markdown
    assert "**下载：**" in markdown
    assert "**在 EasyDistill 2 中的推荐用法：**" in markdown
    assert "自动生成" in markdown


def test_render_model_zoo_markdown_invalid_lang_raises():
    with pytest.raises(ValueError, match="Unsupported lang"):
        render_model_zoo_markdown(lang="fr")
