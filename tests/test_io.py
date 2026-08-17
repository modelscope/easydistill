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

"""Unit tests for I/O utilities."""

from easydistill.utils import load_json, load_jsonl, safe_filename_stem, save_json, save_jsonl


def test_save_and_load_jsonl(tmp_path):
    path = str(tmp_path / "data.jsonl")
    data = [{"a": 1}, {"b": 2}]
    save_jsonl(path, data)
    loaded = load_jsonl(path)
    assert loaded == data


def test_load_jsonl_missing_file_raises(tmp_path):
    path = str(tmp_path / "missing.jsonl")
    try:
        load_jsonl(path)
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert path in str(exc)


def test_load_jsonl_skips_malformed_lines(tmp_path):
    path = str(tmp_path / "partial.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"a": 1}\n')
        f.write("not valid json\n")
        f.write('{"b": 2}\n')
    loaded = load_jsonl(path)
    assert loaded == [{"a": 1}, {"b": 2}]


def test_load_jsonl_strict_raises_on_malformed_lines(tmp_path):
    path = str(tmp_path / "partial.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"a": 1}\n')
        f.write("not valid json\n")
    try:
        load_jsonl(path, strict=True)
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "Malformed JSONL line 2" in str(exc)
        assert path in str(exc)


def test_save_and_load_json(tmp_path):
    path = str(tmp_path / "data.json")
    data = {"key": "value", "list": [1, 2, 3]}
    save_json(path, data)
    loaded = load_json(path)
    assert loaded == data


def test_safe_filename_stem_sanitizes_dangerous_characters():
    assert safe_filename_stem("abc-123_456") == "abc-123_456"
    # Dots, path separators and other special characters are replaced.
    assert safe_filename_stem("../etc/passwd") == "___etc_passwd"
    assert safe_filename_stem("a/b\\c:d<e>f|g*h?i") == "a_b_c_d_e_f_g_h_i"
    assert safe_filename_stem(123) == "123"
