import unittest
from unittest.mock import MagicMock

from smolagents.markov_react import MarkovReActCodeAgent
from smolagents.memory import ActionStep


def make_action_step(name: str):
    """Real ActionStep subtype so isinstance(step, ActionStep) is True."""
    step = MagicMock(spec=ActionStep)
    step.to_messages.return_value = [{"role": "assistant", "content": name}]
    return step


def make_task_step(content: str):
    """Non-ActionStep memory step (simulates TaskStep)."""
    step = MagicMock()
    step.to_messages.return_value = [{"role": "user", "content": content}]
    return step


def make_fake_agent(memory_steps, window_size=3, resend_reasoning=True):
    """Object exposing exactly what MarkovReActCodeAgent.write_memory_to_messages needs,
    without going through CodeAgent.__init__ (which requires a real model/tools)."""

    class FakeAgent:
        write_memory_to_messages = MarkovReActCodeAgent.write_memory_to_messages

    agent = FakeAgent()
    agent.memory = MagicMock()
    agent.memory.system_prompt.to_messages.return_value = [{"role": "system", "content": "sys"}]
    agent.memory.steps = memory_steps
    agent.window_size = window_size
    agent.resend_reasoning = resend_reasoning
    return agent


class TestMarkovReActCodeAgentMemory(unittest.TestCase):
    def test_includes_system_prompt(self):
        agent = make_fake_agent([])
        messages = agent.write_memory_to_messages()
        self.assertEqual(messages[0], {"role": "system", "content": "sys"})

    def test_includes_task_step(self):
        task = make_task_step("What is the capital of France?")
        agent = make_fake_agent([task])
        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages]
        self.assertIn("What is the capital of France?", contents)

    def test_only_last_window_size_action_steps_included(self):
        steps = [make_action_step(f"s{i}") for i in range(1, 6)]  # s1..s5
        agent = make_fake_agent(steps, window_size=3)

        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages if m["role"] == "assistant"]

        self.assertEqual(contents, ["s3", "s4", "s5"])
        self.assertNotIn("s1", contents)
        self.assertNotIn("s2", contents)

    def test_fewer_than_window_size_action_steps_all_included(self):
        steps = [make_action_step("s1"), make_action_step("s2")]
        agent = make_fake_agent(steps, window_size=3)

        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages if m["role"] == "assistant"]

        self.assertEqual(contents, ["s1", "s2"])

    def test_default_window_size_is_3(self):
        steps = [make_action_step(f"s{i}") for i in range(1, 6)]
        agent = make_fake_agent(steps)  # default window_size from make_fake_agent is 3
        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages if m["role"] == "assistant"]
        self.assertEqual(len(contents), 3)

    def test_task_step_appears_before_action_steps(self):
        task = make_task_step("task")
        steps = [task] + [make_action_step(f"s{i}") for i in range(1, 5)]
        agent = make_fake_agent(steps, window_size=2)

        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages]

        task_idx = contents.index("task")
        first_action_idx = contents.index("s3")
        self.assertLess(task_idx, first_action_idx)
        self.assertNotIn("s1", contents)
        self.assertNotIn("s2", contents)

    def test_summary_mode_passed_through(self):
        task = make_task_step("task")
        action = make_action_step("s1")
        agent = make_fake_agent([task, action])

        agent.write_memory_to_messages(summary_mode=True)

        agent.memory.system_prompt.to_messages.assert_called_once_with(summary_mode=True)
        task.to_messages.assert_called_once_with(summary_mode=True)
        action.to_messages.assert_called_once_with(summary_mode=True, include_reasoning=True)

    def test_resend_reasoning_forwarded_to_action_steps_only(self):
        task = make_task_step("task")
        action = make_action_step("s1")
        agent = make_fake_agent([task, action], resend_reasoning=False)

        agent.write_memory_to_messages()

        action.to_messages.assert_called_once_with(summary_mode=False, include_reasoning=False)
        task.to_messages.assert_called_once_with(summary_mode=False)

    def test_window_size_zero_drops_all_action_steps(self):
        task = make_task_step("task")
        steps = [task] + [make_action_step(f"s{i}") for i in range(1, 4)]
        agent = make_fake_agent(steps, window_size=0)

        messages = agent.write_memory_to_messages()
        contents = [m["content"] for m in messages]

        self.assertEqual(contents, ["sys", "task"])


if __name__ == "__main__":
    unittest.main()
