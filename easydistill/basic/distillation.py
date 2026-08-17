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

"""Basic distillation pipeline: seed instructions or problems -> SFT dataset."""

import logging
from typing import Any, Dict, List

from easydistill.backends.base import ModelBackend
from easydistill.data.models import SFTSample
from easydistill.operators import SFTDatasetBuilder, TextGenerationOperator
from easydistill.utils import save_jsonl

logger = logging.getLogger(__name__)


class BasicDistillationPipeline:
    """Run basic distillation.

    Generates teacher responses for pre-loaded requests, builds an SFT dataset,
    and writes the result JSONL. This is the implementation backing the
    `instruct_distill`, `cot_distill`, `mm_instruct_distill`, and
    `mm_cot_distill` job types.
    """

    def __init__(
        self,
        backend: ModelBackend,
        dataset_config: Dict[str, Any],
        generation_config: Dict[str, Any],
        sft_config: Dict[str, Any],
    ):
        self.backend = backend
        self.dataset_config = dataset_config
        self.generation_config = generation_config
        self.sft_config = sft_config

    def run(self, requests: List[Any]) -> List[SFTSample]:
        """Generate teacher responses and return the SFT samples."""
        generator = TextGenerationOperator(backend=self.backend, config=self.generation_config)
        results = generator.run(requests)
        logger.info("Generated %d teacher responses.", len(results))

        builder = SFTDatasetBuilder(config=self.sft_config)
        samples = builder.run(results)

        output_path = self.dataset_config["output_path"]
        save_jsonl(output_path, [sample.model_dump() for sample in samples])
        logger.info("Saved SFT dataset to %s.", output_path)
        return samples
