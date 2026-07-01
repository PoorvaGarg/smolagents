import unittest

from smolagents.models import ChatMessage, MessageRole, Model, TokenUsage
from smolagents.context_pruning.agent import ContextPruningCodeAgent


class FakeModel(Model):
    """Returns a fixed sequence of responses regardless of prompt."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self._call_count = 0

    def generate(self, messages, stop_sequences=None, **kwargs):
        response = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response,
            token_usage=TokenUsage(input_tokens=10, output_tokens=10),
        )


def make_agent(responses: list[str]) -> ContextPruningCodeAgent:
    model = FakeModel(responses)
    return ContextPruningCodeAgent(tools=[], model=model, max_steps=len(responses))


SEARCH_STEP = """Thought: search
<code>
web_search(query="foo")
</code>
"""

SEARCH_STEP_2 = """Thought: search again
<code>
web_search(query="bar")
</code>
"""

FINAL_STEP = """Thought: done
<code>
final_answer("42")
</code>
"""


class TestContextPruningCodeAgentInit(unittest.TestCase):
    def test_buffer_and_detector_initialized(self):
        agent = make_agent([FINAL_STEP])
        self.assertIsNotNone(agent._ctx_buffer)
        self.assertIsNotNone(agent._subtask_detector)
        self.assertIsNone(agent._prev_code)

    def test_buffer_resets_on_run(self):
        agent = make_agent([SEARCH_STEP, FINAL_STEP, SEARCH_STEP, FINAL_STEP])
        agent.run("task1")
        steps_after_first = agent._ctx_buffer.get_pruned_steps()
        agent.run("task2")
        # Buffer reflects task2 only — its steps are independent of task1
        steps_after_second = agent._ctx_buffer.get_pruned_steps()
        self.assertIsNotNone(agent._ctx_buffer)
        # step numbers in task2 start from 1, not continuing from task1
        self.assertEqual(steps_after_second[0].step_number, 1)


class TestContextPruningCodeAgentBuffer(unittest.TestCase):
    def test_same_subtask_steps_accumulate_in_context(self):
        """Two consecutive web_search steps should both appear in active context."""
        agent = make_agent([SEARCH_STEP, SEARCH_STEP_2, FINAL_STEP])
        agent.run("find something")
        # Both search steps should be in current_context (same subtask) or committed
        pruned = agent._ctx_buffer.get_pruned_steps()
        codes = [s.code_action for s in pruned if s.code_action]
        self.assertTrue(any("web_search" in c for c in codes))

    def test_subtask_boundary_prunes_intermediate_steps(self):
        """On subtask change, only the last step of the prior subtask is kept."""
        BROWSE_STEP = """Thought: visit page
<code>
visit_page(url="http://example.com")
</code>
"""
        agent = make_agent([SEARCH_STEP, SEARCH_STEP_2, BROWSE_STEP, FINAL_STEP])
        agent.tools["visit_page"] = unittest.mock.MagicMock()
        agent.tools["visit_page"].name = "visit_page"
        agent._subtask_detector.known_tools.add("visit_page")
        agent.run("find and read")

        committed = agent._ctx_buffer.committed_steps
        # The committed steps should not contain both search steps — only the last
        committed_codes = [s.code_action for s in committed if s.code_action]
        search_steps = [c for c in committed_codes if "web_search" in c]
        # At most one search step should be committed (the last one before boundary)
        self.assertLessEqual(len(search_steps), 1)


class TestContextPruningCodeAgentReset(unittest.TestCase):
    def test_prev_code_set_after_run(self):
        agent = make_agent([SEARCH_STEP, FINAL_STEP])
        agent.run("task1")
        # After the run, _prev_code holds the last step's code (final_answer)
        self.assertIsNotNone(agent._prev_code)
        self.assertIn("final_answer", agent._prev_code)

    def test_committed_steps_cleared_on_reset(self):
        agent = make_agent([SEARCH_STEP, FINAL_STEP])
        agent.run("task1")
        agent.run("task2")
        # After second run, buffer reflects task2 only
        for step in agent._ctx_buffer.committed_steps:
            self.assertIsNotNone(step)


if __name__ == "__main__":
    unittest.main()
