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

"""Load and query the Model Zoo metadata."""

import logging
import os
import re
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_ZOO_PATH = os.path.join(_PACKAGE_DIR, "model_zoo.yaml")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_zoo(path: str = _MODEL_ZOO_PATH) -> Dict[str, Any]:
    """Load the Model Zoo metadata from YAML.

    Args:
        path: Path to the model zoo YAML file. Defaults to the bundled
            ``model_zoo.yaml``.

    Returns:
        A dict with ``version``, ``models``, and ``datasets`` keys.
    """
    data = _load_yaml(path)
    return {
        "version": data.get("version", "unknown"),
        "models": data.get("models", []),
        "datasets": data.get("datasets", []),
    }


def list_models(path: str = _MODEL_ZOO_PATH) -> List[Dict[str, Any]]:
    """Return the list of models in the Model Zoo."""
    return load_model_zoo(path).get("models", [])  # type: ignore[no-any-return]


def list_datasets(path: str = _MODEL_ZOO_PATH) -> List[Dict[str, Any]]:
    """Return the list of datasets in the Model Zoo."""
    return load_model_zoo(path).get("datasets", [])  # type: ignore[no-any-return]


def _sort_key(model: Dict[str, Any]) -> tuple:
    """Sort models by family first, then by name for stable output."""
    return (model.get("family", ""), model.get("name", ""))


def format_model_table(path: str = _MODEL_ZOO_PATH) -> str:
    """Format the model list as a compact, terminal-friendly string."""
    models = sorted(list_models(path), key=_sort_key)
    if not models:
        return "No models found in the Model Zoo."

    lines = ["Model Zoo (open-source models):", ""]
    for model in models:
        name = model["name"]
        family = model.get("family", "")
        size = model.get("size", "")
        mtype = model.get("type", "")
        lines.append(f"  {name}  ({family}, {size}, {mtype})")
    return "\n".join(lines)


def render_model_zoo_markdown(
    path: str = _MODEL_ZOO_PATH, lang: str = "en"
) -> str:
    """Render the Model Zoo metadata as a Markdown document.

    Args:
        path: Path to the model zoo YAML file.
        lang: Output language. Either ``en`` (English) or ``zh`` (Chinese).

    Returns:
        Markdown document rendered in the requested language.
    """
    if lang not in {"en", "zh"}:
        raise ValueError(f"Unsupported lang: {lang!r}. Use 'en' or 'zh'.")

    data = load_model_zoo(path)
    models = sorted(data.get("models", []), key=_sort_key)
    datasets = data.get("datasets", [])

    zh = lang == "zh"

    def localized_text(item: Dict[str, Any], key: str) -> str:
        """Return ``key`` or ``key_zh`` depending on the output language."""
        zh_key = f"{key}_zh"
        raw = item.get(zh_key, "") if zh else item.get(key, "")
        text = re.sub(r"\s+", " ", str(raw)).strip()
        # YAML folded scalars insert a space at line breaks. Remove those
        # spaces inside Chinese sentences where they should not appear.
        text = re.sub(
            r"([\u4e00-\u9fa5\u3001-\u3003\u3008-\u3011\uff08-\uff09"
            r"\uff0c-\uff0e\uff1a\uff1b\uff1f\uff01])"
            r"\s+(?=[\u4e00-\u9fa5])",
            r"\1",
            text,
        )
        return text

    lines: List[str] = []
    if zh:
        lines.append("# EasyDistill 2 Model Zoo")
        lines.append("")
        lines.append(
            "本页面汇总开源的 **DistilQwen** 模型家族以及上游 "
            "[EasyDistill](https://github.com/modelscope/easydistill) 项目发布的公开数据集。"
            "所有模型均同时托管在 "
            "[HuggingFace](https://huggingface.co/alibaba-pai) 与 "
            "[ModelScope](https://modelscope.cn/organization/PAI)。"
        )
    else:
        lines.append("# EasyDistill 2 Model Zoo")
        lines.append("")
        lines.append(
            "This page catalogs the open-source **DistilQwen** model family and the "
            "public datasets released alongside the upstream "
            "[EasyDistill](https://github.com/modelscope/easydistill) project. "
            "All models are hosted on both "
            "[HuggingFace](https://huggingface.co/alibaba-pai) and "
            "[ModelScope](https://modelscope.cn/organization/PAI)."
        )
    lines.append("")

    # Summary table
    lines.append("## " + ("模型总览" if zh else "Model summary"))
    lines.append("")
    if zh:
        lines.append("| 模型 | 家族 | 规模 | 类型 | HuggingFace | ModelScope |")
    else:
        lines.append("| Model | Family | Size | Type | HuggingFace | ModelScope |")
    lines.append("|---|---|---|---|---|---|")
    for model in models:
        name = model["name"]
        family = model.get("family", "")
        size = model.get("size", "")
        mtype = model.get("type", "")
        hf = model.get("downloads", {}).get("huggingface", "")
        ms = model.get("downloads", {}).get("modelscope", "")
        hf_link = f"[HF](https://huggingface.co/{hf})" if hf else ""
        ms_link = f"[MS](https://modelscope.cn/models/{ms})" if ms else ""
        lines.append(f"| {name} | {family} | {size} | {mtype} | {hf_link} | {ms_link} |")
    lines.append("")

    # Per-model details
    lines.append("## " + ("按家族分类" if zh else "Models by family"))
    lines.append("")

    current_family: str = ""
    for model in models:
        family = model.get("family", "Uncategorized")
        if family != current_family:
            lines.append(f"### {family}")
            lines.append("")
            current_family = family

        name = model["name"]
        size = model.get("size", "")
        mtype = model.get("type", "")
        description = localized_text(model, "description")
        capabilities = model.get("capabilities", [])
        usage = model.get("usage", {})
        downloads = model.get("downloads", {})
        hf = downloads.get("huggingface", "")
        ms = downloads.get("modelscope", "")

        if zh:
            lines.append(f"#### {name}（{size}，{mtype}）")
        else:
            lines.append(f"#### {name} ({size}, {mtype})")
        lines.append("")
        if description:
            lines.append(description)
            lines.append("")

        if capabilities:
            label = "**能力：**" if zh else "**Capabilities:** "
            lines.append(label + "、".join(capabilities) if zh else label + ", ".join(capabilities))
            lines.append("")

        if hf or ms:
            links = []
            if hf:
                links.append(f"[HuggingFace](https://huggingface.co/{hf})")
            if ms:
                links.append(f"[ModelScope](https://modelscope.cn/models/{ms})")
            lines.append(("**下载：**" if zh else "**Downloads:** ") + " | ".join(links))
            lines.append("")

        if usage:
            pipeline = usage.get("pipeline", "")
            backend = usage.get("recommended_backend", "")
            notes = localized_text(usage, "notes")
            if pipeline or backend:
                lines.append(
                    "**在 EasyDistill 2 中的推荐用法：**"
                    if zh
                    else "**Recommended usage in EasyDistill 2:**"
                )
                lines.append("")
                if pipeline:
                    label = "流水线" if zh else "Pipeline"
                    lines.append(f"- {label}: `{pipeline}`")
                if backend:
                    label = "后端" if zh else "Backend"
                    lines.append(f"- {label}: `{backend}`")
                if notes:
                    label = "说明" if zh else "Notes"
                    lines.append(f"- {label}: {notes}")
                lines.append("")

    # Datasets
    if datasets:
        lines.append("## " + ("已发布数据集" if zh else "Released datasets"))
        lines.append("")
        if zh:
            lines.append("| 数据集 | 规模 | 类型 | HuggingFace | ModelScope |")
        else:
            lines.append("| Dataset | Size | Type | HuggingFace | ModelScope |")
        lines.append("|---|---|---|---|---|")
        for dataset in datasets:
            dname = dataset["name"]
            dsize = dataset.get("size", "")
            dtype = dataset.get("type", "")
            dhf = dataset.get("downloads", {}).get("huggingface", "")
            dms = dataset.get("downloads", {}).get("modelscope", "")
            dhf_link = f"[HF](https://huggingface.co/{dhf})" if dhf else ""
            dms_link = f"[MS](https://modelscope.cn/datasets/{dms})" if dms else ""
            lines.append(
                f"| {dname} | {dsize} | {dtype} | {dhf_link} | {dms_link} |"
            )
        lines.append("")
        for dataset in datasets:
            dname = dataset["name"]
            ddesc = localized_text(dataset, "description")
            if ddesc:
                lines.append(f"### {dname}")
                lines.append("")
                lines.append(ddesc)
                lines.append("")

    lines.append("---")
    lines.append("")
    if zh:
        lines.append(
            "_本页面由 [`easydistill/models/model_zoo.yaml`](../easydistill/models/model_zoo.yaml) "
            "自动生成。如需更新，请修改 YAML 源文件或 generator 脚本，而非直接编辑本 Markdown。_"
        )
    else:
        lines.append(
            "_This page is auto-generated from "
            "[`easydistill/models/model_zoo.yaml`](../easydistill/models/model_zoo.yaml). "
            "Do not edit it manually; run `python scripts/generate_model_zoo.py` instead._"
        )
    lines.append("")

    return "\n".join(lines)
