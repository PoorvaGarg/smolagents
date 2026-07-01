import unittest

from smolagents.context_pruning.subtask_detector import SubtaskDetector, extract_primary_tool, get_tool_group


TOOLS = {"web_search", "find_archived_url", "visit_page", "page_up", "page_down",
         "find_on_page_ctrl_f", "find_next", "inspect_file_as_text", "visualizer", "final_answer"}


class TestExtractPrimaryTool(unittest.TestCase):
    def setUp(self):
        self.tools = TOOLS

    def test_finds_first_tool_on_first_line(self):
        self.assertEqual(extract_primary_tool('web_search(query="foo")', self.tools), "web_search")

    def test_finds_tool_on_later_line(self):
        code = "x = 1\nvisit_page(url='http://example.com')"
        self.assertEqual(extract_primary_tool(code, self.tools), "visit_page")

    def test_returns_none_for_pure_compute(self):
        self.assertIsNone(extract_primary_tool("x = 1 + 2\nprint(x)", self.tools))

    def test_returns_first_tool_when_multiple_present(self):
        code = "web_search(query='a')\nvisit_page(url='b')"
        self.assertEqual(extract_primary_tool(code, self.tools), "web_search")


class TestGetToolGroup(unittest.TestCase):
    def test_web_search_is_search(self):
        self.assertEqual(get_tool_group("web_search"), "search")

    def test_find_archived_url_is_search(self):
        self.assertEqual(get_tool_group("find_archived_url"), "search")

    def test_visit_page_is_browse(self):
        self.assertEqual(get_tool_group("visit_page"), "browse")

    def test_page_down_is_browse(self):
        self.assertEqual(get_tool_group("page_down"), "browse")

    def test_none_is_compute(self):
        self.assertEqual(get_tool_group(None), "compute")

    def test_unknown_tool_gets_singleton_group(self):
        self.assertEqual(get_tool_group("some_custom_tool"), "tool:some_custom_tool")


class TestSubtaskDetector(unittest.TestCase):
    def setUp(self):
        self.det = SubtaskDetector(TOOLS)

    def test_search_to_search_same(self):
        self.assertTrue(self.det.same_subtask('web_search(query="foo")', 'web_search(query="bar")'))

    def test_search_to_browse_different(self):
        self.assertFalse(self.det.same_subtask('web_search(query="foo")', "visit_page(url='x')"))

    def test_browse_to_browse_same(self):
        self.assertTrue(self.det.same_subtask("visit_page(url='x')", "page_down()"))

    def test_browse_to_search_different(self):
        self.assertFalse(self.det.same_subtask("visit_page(url='x')", 'web_search(query="y")'))

    def test_compute_to_compute_same(self):
        self.assertTrue(self.det.same_subtask("x = 1 + 1", "y = x * 2"))

    def test_compute_to_search_different(self):
        self.assertFalse(self.det.same_subtask("x = 1 + 1", 'web_search(query="z")'))

    def test_different_tools_same_group(self):
        self.assertTrue(self.det.same_subtask('web_search(query="a")', 'find_archived_url("b")'))

    def test_file_to_browse_different(self):
        self.assertFalse(self.det.same_subtask("inspect_file_as_text('doc.pdf')", "page_down()"))

    def test_final_answer_singleton_group(self):
        self.assertFalse(self.det.same_subtask('web_search(query="x")', 'final_answer("done")'))


if __name__ == "__main__":
    unittest.main()
