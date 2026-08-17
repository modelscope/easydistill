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

"""Unit tests for video frame sampling utilities (requires OpenCV)."""

import pytest

from easydistill.utils import sample_video_frames

cv2 = pytest.importorskip("cv2", reason="opencv is an optional dependency")
np = pytest.importorskip("numpy")


@pytest.fixture
def tiny_video(tmp_path):
    """Write a small 24-frame test video with a moving square."""
    path = str(tmp_path / "tiny.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48)
    )
    for i in range(24):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, (i * 2) % 64 : (i * 2) % 64 + 8] = 255
        writer.write(frame)
    writer.release()
    return path


class TestSampleVideoFrames:
    def test_uniform_sampling(self, tiny_video):
        frames = sample_video_frames(tiny_video, count=4)
        assert len(frames) == 4
        for frame in frames:
            assert frame.data_url.startswith("data:image/jpeg;base64,")
        # Timestamps are monotonically increasing.
        timestamps = [f.timestamp for f in frames]
        assert all(t is not None for t in timestamps)
        assert timestamps == sorted(timestamps)
        # First frame at t=0, last near the end of the 2s clip.
        assert timestamps[0] == 0.0
        assert timestamps[-1] > 1.5

    def test_count_capped_by_total_frames(self, tiny_video):
        frames = sample_video_frames(tiny_video, count=100)
        assert len(frames) <= 24

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            sample_video_frames(str(tmp_path / "nope.mp4"))

    def test_resize_bounds_long_side(self, tmp_path):
        path = str(tmp_path / "big.mp4")
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (256, 128)
        )
        for _ in range(6):
            writer.write(np.zeros((128, 256, 3), dtype=np.uint8))
        writer.release()

        import base64

        frames = sample_video_frames(path, count=2, max_size=64)
        raw = base64.b64decode(frames[0].data_url.split(",", 1)[1])
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert max(image.shape[:2]) <= 64
