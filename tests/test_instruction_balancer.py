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

"""Tests for the instruction balancing operator."""

import pytest

from easydistill.operators import InstructionBalancer
from easydistill.operators.balance import DEFAULT_DISTILQWEN2_DISTRIBUTION
from tests._fake_backend import FakeBackend


@pytest.fixture
def fake_backend():
    return FakeBackend(response_template="<answer>Math</answer>")


def test_extract_category_with_tags():
    balancer = InstructionBalancer(backend=FakeBackend())
    assert balancer._extract_category("some text <answer>Math</answer> more") == "Math"


def test_extract_category_fallback_to_text():
    balancer = InstructionBalancer(backend=FakeBackend())
    assert balancer._extract_category("The answer is Reasoning here") == "Reasoning"


def test_extract_category_defaults_to_others():
    balancer = InstructionBalancer(backend=FakeBackend())
    assert balancer._extract_category("I have no idea") == "Others"


def test_default_distribution_loaded():
    assert "Math" in DEFAULT_DISTILQWEN2_DISTRIBUTION
    assert abs(sum(DEFAULT_DISTILQWEN2_DISTRIBUTION.values()) - 1.0) < 1e-6


def test_resample_downsample_and_upsample():
    backend = FakeBackend()
    balancer = InstructionBalancer(
        backend=backend,
        config={
            "categories": ["A", "B"],
            "target_distribution": {"A": 0.5, "B": 0.5},
            "seed": 1,
        },
    )
    rows = [{"instruction": f"inst {i}", "category": "A" if i < 8 else "B"} for i in range(10)]
    balanced = balancer._resample(rows)

    assert len(balanced) == 10
    a_count = sum(1 for row in balanced if row["category"] == "A")
    b_count = sum(1 for row in balanced if row["category"] == "B")
    assert a_count == 5
    assert b_count == 5


def test_resample_upsample_by_repeating():
    backend = FakeBackend()
    balancer = InstructionBalancer(
        backend=backend,
        config={
            "categories": ["A", "B"],
            "target_distribution": {"A": 0.75, "B": 0.25},
            "seed": 1,
        },
    )
    rows = [
        {"instruction": "a1", "category": "A"},
        {"instruction": "b1", "category": "B"},
        {"instruction": "b2", "category": "B"},
        {"instruction": "b3", "category": "B"},
    ]
    balanced = balancer._resample(rows)
    assert len(balanced) == 4
    a_count = sum(1 for row in balanced if row["category"] == "A")
    b_count = sum(1 for row in balanced if row["category"] == "B")
    assert a_count == 3
    assert b_count == 1


def test_resample_skips_empty_categories():
    """A target category with no samples must not cause a crash."""
    balancer = InstructionBalancer(
        backend=FakeBackend(),
        config={
            "categories": ["A", "B"],
            "target_distribution": {"A": 0.5, "B": 0.5},
            "seed": 1,
        },
    )
    rows = [{"instruction": "a1", "category": "A"}]
    balanced = balancer._resample(rows)
    assert len(balanced) == 1
    assert balanced[0]["category"] == "A"


def test_end_to_end_classification_and_resampling(fake_backend):
    balancer = InstructionBalancer(
        backend=fake_backend,
        config={
            "categories": ["Math", "Others"],
            "target_distribution": {"Math": 1.0, "Others": 0.0},
            "seed": 7,
            "show_progress": False,
            "max_workers": 1,
        },
    )
    rows = [{"instruction": f"question {i}"} for i in range(6)]
    balanced = balancer.run(rows)

    assert len(balanced) == 6
    for row in balanced:
        assert row["category"] == "Math"


def test_end_to_end_preserves_metadata(fake_backend):
    balancer = InstructionBalancer(
        backend=fake_backend,
        config={
            "categories": ["Math", "Others"],
            "target_distribution": {"Math": 1.0, "Others": 0.0},
            "seed": 7,
            "show_progress": False,
            "max_workers": 1,
        },
    )
    rows = [{"instruction": "q", "source": "test"} for _ in range(3)]
    balanced = balancer.run(rows)
    assert len(balanced) == 3
    for row in balanced:
        assert row["source"] == "test"


def test_empty_input_returns_empty():
    balancer = InstructionBalancer(backend=FakeBackend())
    assert balancer.run([]) == []
