"""Subtask boundary detection for context-pruning agents.

A "subtask" is a coherent sequence of steps working toward the same intermediate
goal (e.g. "find the right URL via web search", "read and paginate a webpage",
"do a computation"). When the primary tool changes group, it signals that the
agent has finished one subtask and started another.
"""

# Tools that naturally belong together in a subtask.
TOOL_GROUPS: dict[str, set[str]] = {
    "search": {"web_search", "find_archived_url"},
    "browse": {"visit_page", "page_up", "page_down", "find_on_page_ctrl_f", "find_next"},
    "file": {"inspect_file_as_text", "visualizer"},
}


def extract_primary_tool(code: str, known_tools: set[str]) -> str | None:
    """Return the first tool called in *code*, or None for pure-compute steps."""
    for line in code.splitlines():
        for tool in known_tools:
            if f"{tool}(" in line:
                return tool
    return None


def get_tool_group(tool: str | None) -> str:
    """Map a tool name to its subtask group label."""
    if tool is None:
        return "compute"
    for group, members in TOOL_GROUPS.items():
        if tool in members:
            return group
    return f"tool:{tool}"  # unknown tool gets its own singleton group


class SubtaskDetector:
    """Detects whether two consecutive agent steps belong to the same subtask."""

    def __init__(self, known_tools: set[str]):
        self.known_tools = known_tools

    def primary_tool(self, code: str) -> str | None:
        return extract_primary_tool(code, self.known_tools)

    def same_subtask(self, code_a: str, code_b: str) -> bool:
        """Return True if *code_b* is in service of the same subtask as *code_a*."""
        group_a = get_tool_group(self.primary_tool(code_a))
        group_b = get_tool_group(self.primary_tool(code_b))
        return group_a == group_b
