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

"""Preference data build runners."""

import logging

from easydistill.pipeline import PreferenceDistillationPipeline
from easydistill.utils import load_expanded_config

from ..backend_factory import build_backend, check_backend_health, close_backends

logger = logging.getLogger(__name__)


def run_dpo_data_build(config_path: str) -> None:
    """Run DPO preference data pipeline."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        preference_config = cfg.get("preference") or {}

        pipeline = PreferenceDistillationPipeline(
            backend=backend,
            pipeline_config=cfg["pipeline"],
            dataset_config=cfg["dataset"],
            generation_config=cfg.get("generation", {}),
            preference_config=preference_config,
        )
        pipeline.run()
    finally:
        close_backends(backend)
