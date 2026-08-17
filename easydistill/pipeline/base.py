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

"""Base class for advanced distillation pipelines."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

from easydistill.backends.base import ModelBackend
from easydistill.utils import load_dataset_rows, save_jsonl

from .common import _save_intermediate_output

logger = logging.getLogger(__name__)


class BaseDistillationPipeline(ABC):
    """End-to-end distillation pipeline with stage dispatch.

    Subclasses define:
      - _last_stage: the required final stage name.
      - _default_eval_metrics: optional default metrics for quality filtering.
      - _dispatch_stage: how to run a single stage.
    """

    _last_stage: Union[str, List[str], Set[str]] = ""
    _default_eval_metrics: Optional[List[str]] = None

    def __init__(
        self,
        backend: ModelBackend,
        pipeline_config: List[Dict[str, Any]],
        dataset_config: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None,
        sft_config: Optional[Dict[str, Any]] = None,
        eval_config: Optional[Dict[str, Any]] = None,
    ):
        self.backend = backend
        self.pipeline_config = pipeline_config
        self.dataset_config = dataset_config
        self.generation_config = generation_config or {}
        self.sft_config = sft_config or {}
        self.eval_config = eval_config or {}

        if not self.pipeline_config:
            raise ValueError("Pipeline config must contain at least one stage.")

        last_stage = self.pipeline_config[-1].get("stage")
        allowed = self._last_stage
        allowed = {allowed} if isinstance(allowed, str) else set(allowed)
        if last_stage not in allowed:
            if len(allowed) == 1:
                raise ValueError(f"The last pipeline stage must be '{next(iter(allowed))}'.")
            raise ValueError(f"The last pipeline stage must be one of {sorted(allowed)}.")

    def run_with_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data:
            raise ValueError("Input data is empty.")
        return self._run_stages(data)

    def run(self) -> List[Dict[str, Any]]:
        data = load_dataset_rows(self.dataset_config["input_path"])
        return self._run_stages(data)

    def _run_stages(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(
            "%s started with %d seed rows.",
            self.__class__.__name__,
            len(data),
        )

        eval_metrics = list(self.eval_config.get("metrics", self._default_eval_metrics or []))

        for stage_idx, stage in enumerate(self.pipeline_config):
            stage_name = stage["stage"]
            stage_config = stage.get("config", {})
            output_path = stage.get("output_path")

            logger.info(
                "Running pipeline stage %d/%d: %s",
                stage_idx + 1,
                len(self.pipeline_config),
                stage_name,
            )

            try:
                data = self._dispatch_stage(stage_name, stage_config, data, eval_metrics)
            except Exception:
                self._save_recovery_checkpoint(data, stage_idx, stage_name)
                raise
            _save_intermediate_output(output_path, data)

        final_output_path = self.dataset_config.get("output_path")
        if final_output_path:
            save_jsonl(final_output_path, data)
            logger.info("Pipeline finished. Saved final SFT dataset to %s.", final_output_path)
        return data

    def _save_recovery_checkpoint(
        self,
        data: List[Dict[str, Any]],
        failed_stage_idx: int,
        failed_stage_name: str,
    ) -> None:
        """Save the last known good data before a failing stage so users can resume."""
        final_output_path = self.dataset_config.get("output_path")
        if not final_output_path:
            logger.warning(
                "Pipeline stage %s failed; no final output_path configured, "
                "so no recovery checkpoint was written.",
                failed_stage_name,
            )
            return
        recovery_path = (
            f"{final_output_path}.recovery.stage_{failed_stage_idx}.{failed_stage_name}.jsonl"
        )
        try:
            save_jsonl(recovery_path, data)
            logger.warning(
                "Pipeline stage %d (%s) failed. Recovery checkpoint written to %s "
                "containing output from the previous successful stage.",
                failed_stage_idx + 1,
                failed_stage_name,
                recovery_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write recovery checkpoint to %s: %s", recovery_path, exc)

    @abstractmethod
    def _dispatch_stage(
        self,
        stage_name: str,
        stage_config: Dict[str, Any],
        data: List[Dict[str, Any]],
        eval_metrics: List[str],
    ) -> List[Dict[str, Any]]:
        """Run a single pipeline stage and return the updated data."""
        raise NotImplementedError
