import unittest
from unittest.mock import MagicMock, patch

from smolagents.context_pruning.context_buffer import SideContextBuffer
from smolagents.context_pruning.pruned_memory import PrunedMemoryMixin
from smolagents.memory import ActionStep


def make_step(name: str, messages=None):
    """Minimal ActionStep mock whose to_messages() returns a predictable list."""
    step = MagicMock()
    step.name = name
    step.to_messages.return_value = messages or [{"role": "step", "content": name}]
    return step


def make_action_step(name: str, messages=None):
    """Real ActionStep subtype so isinstance(step, ActionStep) is True."""
    step = MagicMock(spec=ActionStep)
    step.name = name
    step.to_messages.return_value = messages or [{"role": "assistant", "content": name}]
    return step


def make_task_step(content: str):
    """Non-ActionStep memory step (simulates TaskStep)."""
    step = MagicMock()
    step.to_messages.return_value = [{"role": "user", "content": content}]
    return step


def make_agent_with_buffer(steps_in_buffer, memory_steps=None):
    """Build a minimal object that satisfies PrunedMemoryMixin's requirements.

    memory_steps: list of non-ActionStep objects placed in memory.steps
                  (simulates TaskStep / PlanningStep).
    """

    class FakeAgent(PrunedMemoryMixin):
        def __init__(self):
            self.memory = MagicMock()
            self.memory.system_prompt.to_messages.return_value = [{"role": "system", "content": "sys"}]
            self.memory.steps = memory_steps or []
            self._ctx_buffer = SideContextBuffer()

    agent = FakeAgent()
    for step in steps_in_buffer:
        agent._ctx_buffer.add_step(step)
    return agent


class TestPrunedMemoryMixin(unittest.TestCase):
    def test_includes_system_prompt(self):
        agent = make_agent_with_buffer([])
        messages = agent.write_memory_to_messages()
        self.assertEqual(messages[0], {"role": "system", "content": "sys"})

    def test_empty_buffer_returns_only_system_prompt(self):
        agent = make_agent_with_buffer([])
        self.assertEqual(agent.write_memory_to_messages(), [{"role": "system", "content": "sys"}])

    def test_single_step_included(self):
        s = make_step("s1", [{"role": "assistant", "content": "s1"}])
        agent = make_agent_with_buffer([s])
        messages = agent.write_memory_to_messages()
        self.assertIn({"role": "assistant", "content": "s1"}, messages)

    def test_pruned_view_after_subtask_commit(self):
        s1 = make_step("s1", [{"role": "assistant", "content": "s1"}])
        s2 = make_step("s2", [{"role": "assistant", "content": "s2"}])
        s3 = make_step("s3", [{"role": "assistant", "content": "s3"}])

        agent = make_agent_with_buffer([])
        agent._ctx_buffer.add_step(s1)
        agent._ctx_buffer.add_step(s2)
        agent._ctx_buffer.commit_subtask()   # s1 dropped, s2 committed
        agent._ctx_buffer.add_step(s3)       # active subtask

        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages if m["role"] == "assistant"]

        self.assertIn("s2", contents)   # last of completed subtask kept
        self.assertIn("s3", contents)   # active subtask kept
        self.assertNotIn("s1", contents)  # intermediate step pruned

    def test_all_completed_subtask_representatives_preserved(self):
        """One representative (last step) from every completed subtask must appear."""
        # Subtask A: 3 steps — only s2 should survive
        s1 = make_step("s1", [{"role": "assistant", "content": "s1"}])
        s2 = make_step("s2", [{"role": "assistant", "content": "s2"}])
        # Subtask B: 2 steps — only s4 should survive
        s3 = make_step("s3", [{"role": "assistant", "content": "s3"}])
        s4 = make_step("s4", [{"role": "assistant", "content": "s4"}])
        # Subtask C (active): s5 and s6 both visible
        s5 = make_step("s5", [{"role": "assistant", "content": "s5"}])
        s6 = make_step("s6", [{"role": "assistant", "content": "s6"}])

        agent = make_agent_with_buffer([])
        for s in (s1, s2):
            agent._ctx_buffer.add_step(s)
        agent._ctx_buffer.commit_subtask()   # A done → s2 kept
        for s in (s3, s4):
            agent._ctx_buffer.add_step(s)
        agent._ctx_buffer.commit_subtask()   # B done → s4 kept
        for s in (s5, s6):
            agent._ctx_buffer.add_step(s)   # C active

        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages if m["role"] == "assistant"]

        # representatives of completed subtasks preserved
        self.assertIn("s2", contents)
        self.assertIn("s4", contents)
        # full active subtask preserved
        self.assertIn("s5", contents)
        self.assertIn("s6", contents)
        # intermediate steps dropped
        self.assertNotIn("s1", contents)
        self.assertNotIn("s3", contents)

    def test_summary_mode_passed_through(self):
        s = make_step("s1")
        agent = make_agent_with_buffer([s])
        agent.write_memory_to_messages(summary_mode=True)
        s.to_messages.assert_called_once_with(summary_mode=True)
        agent.memory.system_prompt.to_messages.assert_called_once_with(summary_mode=True)


class TestPrunedMemoryMixinTaskStep(unittest.TestCase):
    def test_task_step_included_before_buffer_steps(self):
        """TaskStep from memory.steps must appear before buffer ActionSteps."""
        task = make_task_step("What is the capital of France?")
        action = make_action_step("a1", [{"role": "assistant", "content": "a1"}])
        agent = make_agent_with_buffer([action], memory_steps=[task])

        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages]

        task_idx = contents.index("What is the capital of France?")
        action_idx = contents.index("a1")
        self.assertLess(task_idx, action_idx)

    def test_task_step_present_with_empty_buffer(self):
        """TaskStep shows up even when there are no buffer steps."""
        task = make_task_step("my task")
        agent = make_agent_with_buffer([], memory_steps=[task])

        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages]
        self.assertIn("my task", contents)

    def test_action_steps_in_memory_steps_are_skipped(self):
        """ActionSteps in memory.steps must not be double-included (buffer handles them)."""
        task = make_task_step("task")
        stale_action = make_action_step("stale", [{"role": "assistant", "content": "stale"}])
        buffer_action = make_action_step("current", [{"role": "assistant", "content": "current"}])

        agent = make_agent_with_buffer([buffer_action], memory_steps=[task, stale_action])

        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages]

        self.assertIn("task", contents)
        self.assertIn("current", contents)
        # stale_action was in memory.steps as an ActionStep — should be skipped
        self.assertNotIn("stale", contents)

    def test_summary_mode_forwarded_to_task_step(self):
        task = make_task_step("task")
        agent = make_agent_with_buffer([], memory_steps=[task])
        agent.write_memory_to_messages(summary_mode=True)
        task.to_messages.assert_called_once_with(summary_mode=True)


if __name__ == "__main__":
    unittest.main()
