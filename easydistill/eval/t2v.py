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

"""Composable evaluation orchestrator for T2V/I2V (text/image-to-video) datasets.

Unlike :class:`~easydistill.eval.t2i.T2IImageEvaluator` — a single
LLM-judge class, because image evaluation has one mechanism — video
evaluation mixes heterogeneous mechanisms (local objective metrics, image-VLM
frame judging, video-native models).  Each mechanism is a
:class:`~easydistill.eval.t2v_checkers.BaseVideoChecker` subclass, and this
orchestrator composes an arbitrary checker list from config:

.. code-block:: yaml

    eval:
      checkers:
        - type: vbench          # fast objective screen (default dims only)
          dimensions: [motion_smoothness, dynamic_degree]
        - type: vlm             # image VLM over sampled frames
          metrics: [prompt_consistency, visual_quality]
          frame_sample_count: 8
        - type: omni            # video-native holistic check
          enabled: false

When ``checkers`` is omitted, a single ``vlm`` checker is built from the
top-level eval config (so ``eval.metrics`` keeps working unchanged).
"""

import logging
from typing import Any, Dict, List, Optional

from .t2v_checkers import build_video_checkers

logger = logging.getLogger(__name__)

# Config keys that belong to the orchestrator itself and must not leak into
# the implicit vlm checker config.
_ORCHESTRATOR_KEYS = {"checkers"}


class T2VVideoEvaluator:
    """Evaluate T2V/I2V-generated videos through a composable checker chain.

    Checkers run in configured order; each one merges its own metric
    columns into the rows.  A failing checker only blanks its own columns
    and never kills the batch, so downstream ``quality_filter`` stages always
    see a complete score table.

    Row fields used (see individual checkers for details):
      - ``optimized_prompt`` (or ``prompt``), ``video_urls``,
        optional ``frame_urls`` and ``first_frame_image`` (I2V).

    Configurable fields:
      - checkers: list of checker config dicts (``type`` +
        checker-specific keys).  Defaults to a single ``vlm`` checker
        built from the remaining top-level config for backward compatibility.
    """

    name = "t2v_video_evaluator"

    def __init__(self, backend, config: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.config = config or {}
        checker_configs = self.config.get("checkers")
        if not checker_configs:
            # Back-compat shortcut: top-level metrics / frame knobs configure
            # a single vlm checker.
            shortcut = {
                k: v for k, v in self.config.items() if k not in _ORCHESTRATOR_KEYS
            }
            checker_configs = [{"type": "vlm", **shortcut}]
        self.checkers = build_video_checkers(checker_configs, backend=backend)
        if not self.checkers:
            raise ValueError("T2VVideoEvaluator has no enabled checkers.")

    @property
    def metrics(self) -> List[str]:
        """Union of all metric columns produced by the enabled checkers."""
        metrics: List[str] = []
        for checker in self.checkers:
            for metric in checker.metrics:
                if metric not in metrics:
                    metrics.append(metric)
        return metrics

    def run(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run every enabled checker over the samples in configured order."""
        if not samples:
            return []
        rows = []
        for idx, sample in enumerate(samples):
            row = dict(sample)
            row.setdefault("id", str(idx))
            rows.append(row)

        for checker in self.checkers:
            try:
                rows = checker.check(rows)
            except Exception as exc:  # noqa: BLE001 - isolate checker failures
                logger.error(
                    "Video checker '%s' failed (%s); blanking its metrics %s.",
                    checker.name,
                    exc,
                    checker.metrics,
                )
                rows = checker._blank(rows)
        logger.info(
            "Evaluated %d videos with checkers: %s",
            len(rows),
            ", ".join(checker.name for checker in self.checkers),
        )
        return rows

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute average scores per metric across all rows."""
        aggregates = {}
        for metric in self.metrics:
            values = [r[metric] for r in results if r.get(metric) is not None]
            aggregates[metric] = sum(values) / len(values) if values else None
        return aggregates
