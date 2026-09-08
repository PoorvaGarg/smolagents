import unittest
from unittest.mock import patch

from smolagents.context_pruning import goal_agent
from smolagents.context_pruning.context_buffer import SideContextBuffer
from smolagents.context_pruning.goal_agent import GoalPruningCodeAgent, parse_expectation
from smolagents.memory import ActionStep, Timing
from smolagents.models import ChatMessage, MessageRole, Model, TokenUsage


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


def make_agent(responses: list[str], **kwargs) -> GoalPruningCodeAgent:
    return GoalPruningCodeAgent(
        tools=[], model=FakeModel(responses), max_steps=len(responses) or 1, **kwargs
    )


def make_step(number: int, probability: float | None = None) -> ActionStep:
    step = ActionStep(step_number=number, timing=Timing(start_time=0.0, end_time=1.0))
    step.completion_prob = probability
    return step


class TestParseExpectation(unittest.TestCase):
    def test_plain_line(self):
        assert parse_expectation("Expectation: the year of publication") == "the year of publication"

    def test_markdown_wrappers(self):
        assert parse_expectation("- **Expectation:** the population") == "the population"
        assert parse_expectation("**Expectation: the arXiv id**") == "the arXiv id"
        assert parse_expectation("Expectation: `the arXiv id`") == "the arXiv id"

    def test_expected_observation_synonym(self):
        assert parse_expectation("Expected observation: a list of albums") == "a list of albums"

    def test_ignores_lines_inside_code_fences(self):
        text = "```python\n# Expectation: fake\n```\nExpectation: real one"
        assert parse_expectation(text) == "real one"
        assert parse_expectation("```python\n# Expectation: fake\n```") is None

    def test_last_line_wins(self):
        # Models sometimes echo the format from the system prompt before committing to a value.
        text = "Expectation: <what you expect>\nExpectation: the density of honey"
        assert parse_expectation(text) == "the density of honey"

    def test_absent_or_empty(self):
        assert parse_expectation("Thought: x") is None
        assert parse_expectation("Expectation:   ") is None
        assert parse_expectation(None) is None
        assert parse_expectation("") is None


class TestBufferKeepArgument(unittest.TestCase):
    def test_default_keeps_last(self):
        buffer = SideContextBuffer()
        for i in (1, 2, 3):
            buffer.add_step(make_step(i))
        buffer.commit_subtask()
        assert [s.step_number for s in buffer.committed_steps] == [3]
        assert buffer.current_context == []

    def test_keep_promotes_the_named_step(self):
        buffer = SideContextBuffer()
        steps = [make_step(i) for i in (1, 2, 3)]
        for step in steps:
            buffer.add_step(step)
        buffer.commit_subtask(keep=steps[0])
        assert [s.step_number for s in buffer.committed_steps] == [1]

    def test_commit_on_empty_context_is_a_noop(self):
        buffer = SideContextBuffer()
        buffer.commit_subtask()
        assert buffer.committed_steps == []


class TestGoalPruningBoundaries(unittest.TestCase):
    """Drives _finalize_step directly with a stubbed judge, so no model call is made."""

    def _run_probabilities(self, probabilities, **kwargs):
        agent = make_agent([], **kwargs)
        agent._ctx_buffer.reset()
        steps = []
        for i, probability in enumerate(probabilities, start=1):
            step = make_step(i)
            step.model_output = f"Expectation: goal {i}\n<code>\npass\n</code>"
            with patch.object(GoalPruningCodeAgent, "_judge_step", return_value=probability):
                agent._finalize_step(step)
            steps.append(step)
        return agent, steps

    def test_satisfied_step_closes_subtask_and_drops_earlier_attempts(self):
        agent, _ = self._run_probabilities([0.0, 0.0, 0.99])
        buffer = agent._ctx_buffer
        assert [s.step_number for s in buffer.committed_steps] == [3]
        assert buffer.current_context == []
        # The two failed attempts are gone from the prompt the model will see.
        assert [s.step_number for s in buffer.get_pruned_steps()] == [3]

    def test_unsatisfied_steps_stay_visible(self):
        agent, _ = self._run_probabilities([0.0, 0.1, 0.2])
        buffer = agent._ctx_buffer
        assert buffer.committed_steps == []
        assert [s.step_number for s in buffer.get_pruned_steps()] == [1, 2, 3]

    def test_force_close_at_cap_keeps_the_best_scoring_step(self):
        agent, _ = self._run_probabilities([0.1, 0.4, 0.2, 0.3], max_open_steps=4)
        assert [s.step_number for s in agent._ctx_buffer.committed_steps] == [2]

    def test_force_close_tie_breaks_to_the_most_recent_step(self):
        # Every attempt scoring zero is the common case, and argmax alone would keep the first.
        agent, _ = self._run_probabilities([0.0, 0.0, 0.0, 0.0], max_open_steps=4)
        assert [s.step_number for s in agent._ctx_buffer.committed_steps] == [4]

    def test_threshold_is_respected(self):
        agent, _ = self._run_probabilities([0.6], completion_threshold=0.9)
        assert agent._ctx_buffer.committed_steps == []
        agent, _ = self._run_probabilities([0.6], completion_threshold=0.5)
        assert [s.step_number for s in agent._ctx_buffer.committed_steps] == [1]

    def test_unknown_probability_never_commits(self):
        agent, _ = self._run_probabilities([None, None])
        assert agent._ctx_buffer.committed_steps == []
        assert len(agent._ctx_buffer.current_context) == 2


class TestGoalPruningJudgeWiring(unittest.TestCase):
    def test_errored_step_is_unsatisfied_without_a_judge_call(self):
        agent = make_agent([])
        step = make_step(1)
        step.error = ValueError("boom")
        with patch.object(goal_agent, "judge_completion") as judge:
            assert agent._judge_step(step) == 0.0
            judge.assert_not_called()

    def test_missing_expectation_skips_the_judge(self):
        agent = make_agent([])
        step = make_step(1)
        step.expectation = None
        with patch.object(goal_agent, "judge_completion") as judge:
            assert agent._judge_step(step) is None
            judge.assert_not_called()

    def test_judge_disabled_skips_the_call(self):
        agent = make_agent([], judge_completion_calls=False)
        step = make_step(1)
        step.expectation = "something"
        with patch.object(goal_agent, "judge_completion") as judge:
            assert agent._judge_step(step) is None
            judge.assert_not_called()

    def test_judge_tokens_are_added_to_the_step(self):
        agent = make_agent([])
        step = make_step(1)
        step.expectation = "the height"
        step.token_usage = TokenUsage(input_tokens=100, output_tokens=20)
        usage = TokenUsage(input_tokens=7, output_tokens=1)
        with patch.object(goal_agent, "judge_completion", return_value=(0.9, "yes", usage)):
            assert agent._judge_step(step) == 0.9
        assert step.token_usage.input_tokens == 107
        assert step.token_usage.output_tokens == 21

    def test_final_answer_step_is_not_judged(self):
        agent = make_agent([])
        step = make_step(1)
        step.model_output = "Expectation: the answer\n<code>\nfinal_answer(1)\n</code>"
        step.is_final_answer = True
        with patch.object(GoalPruningCodeAgent, "_judge_step") as judge:
            agent._finalize_step(step)
            judge.assert_not_called()
        assert step.expectation == "the answer"
        assert [s.step_number for s in agent._ctx_buffer.current_context] == [1]


class TestActionStepFields(unittest.TestCase):
    def test_dict_carries_the_new_fields(self):
        step = make_step(1, 0.75)
        step.expectation = "the year"
        payload = step.dict()
        assert payload["expectation"] == "the year"
        assert payload["completion_prob"] == 0.75
