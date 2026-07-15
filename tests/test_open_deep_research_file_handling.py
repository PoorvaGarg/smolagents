# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
"""Tests for examples/open_deep_research's attached-file handling, used by baseline/naiveReAct.ipynb
to surface GAIA file attachments (xlsx/pdf/png/...) to the agent's task."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "open_deep_research"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

run_agents = pytest.importorskip(
    "scripts.run_agents", reason="requires examples/open_deep_research's extra dependencies (see its requirements.txt)"
)
text_inspector_tool = pytest.importorskip("scripts.text_inspector_tool")


class TestGetSingleFileDescription:
    def test_short_xlsx_returns_document_content_without_calling_model(self, tmp_path):
        file_path = tmp_path / "inventory.xlsx"
        pd.DataFrame({"Title": ["Item A", "Item B"], "Status": ["Available", "Rented"]}).to_excel(
            file_path, index=False
        )

        mock_model = MagicMock()
        ti_tool = text_inspector_tool.TextInspectorTool(mock_model, text_limit=100000)
        visual_inspection_tool = MagicMock()

        description = run_agents.get_single_file_description(
            str(file_path), "What items are available?", visual_inspection_tool, ti_tool
        )

        assert " - Attached document: " in description
        assert "Item A" in description
        assert "Available" in description
        # Content is short enough (<4000 chars) that TextInspectorTool.forward_initial_exam_mode
        # returns it directly instead of asking the model for a caption.
        mock_model.assert_not_called()
        visual_inspection_tool.assert_not_called()
