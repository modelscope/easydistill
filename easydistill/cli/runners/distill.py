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

"""Basic distillation runner."""

import logging

from easydistill.basic import BasicDistillationPipeline
from easydistill.utils import load_expanded_config

from ..backend_factory import build_backend, check_backend_health, close_backends
from ..data_loaders import load_requests

logger = logging.getLogger(__name__)


def run_distill(config_path: str) -> None:
    """Run the basic distillation pipeline: seed instructions -> SFT dataset."""
    cfg = load_expanded_config(config_path)

    backend = build_backend(cfg["backend"])
    try:
        check_backend_health(backend)

        requests = load_requests(cfg)

        pipeline = BasicDistillationPipeline(
            backend=backend,
            dataset_config=cfg.get("dataset", {}),
            generation_config=cfg.get("generation", {}),
            sft_config=cfg.get("sft", {}),
        )
        pipeline.run(requests)
    finally:
        close_backends(backend)
