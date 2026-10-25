import math
import unittest
from unittest.mock import MagicMock, patch

from smolagents.beam_search_agent import Beam, BeamSearchCodeAgent
from smolagents.memory import ActionStep, AgentMemory, Timing
from smolagents.monitoring import LogLevel


def _beam(
    beam_id,
    errored=False,
    is_final=False,
    final_output=None,
    p_yes=(0.5,),
    thought_lps=None,
    code_lps=None,
    action_output=None,
):
    n = len(p_yes)
    memory = AgentMemory("system prompt")
    memory.steps = [
        ActionStep(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=0.0),
            action_output=action_output,
            is_final_answer=is_final,
        )
    ]
    return Beam(
        beam_id=beam_id,
        parent_id=None,
        memory=memory,
        executor_state={},
        tool_state={},
        expectation_scores=list(p_yes),
        thought_logprobs=thought_lps if thought_lps is not None else [0.0] * n,
        code_logprobs=code_lps if code_lps is not None else [0.0] * n,
        errored_this_step=errored,
        is_final=is_final,
        final_output=final_output,
    )


class TestBeamPruningScore(unittest.TestCase):
    """Beam.pruning_score: the log joint of judge P(yes) with thought/code generation
    likelihood, combined per step then averaged -- see TODO.md."""

    def test_combines_log_p_yes_with_thought_and_code_logprobs(self):
        beam = _beam("0", p_yes=(0.8, 0.4), thought_lps=[-0.5, -1.0], code_lps=[-0.2, -0.3])
        expected = ((math.log(0.8) - 0.5 - 0.2) + (math.log(0.4) - 1.0 - 0.3)) / 2
        self.assertAlmostEqual(beam.pruning_score, expected)

    def test_zero_p_yes_is_floored_not_negative_infinity(self):
        beam = _beam("0", p_yes=(0.0,), thought_lps=[0.0], code_lps=[0.0])
        self.assertTrue(math.isfinite(beam.pruning_score))

    def test_no_steps_yet_is_zero(self):
        beam = Beam(beam_id="0", parent_id=None, memory=AgentMemory("system prompt"), executor_state={}, tool_state={})
        self.assertEqual(beam.pruning_score, 0.0)


class TestPrune(unittest.TestCase):
    """_prune: lexicographic (not errored_this_step, pruning_score), top beam_width survive."""

    def _agent(self, beam_width=2):
        return BeamSearchCodeAgent(tools=[], model=MagicMock(), beam_width=beam_width, verbosity_level=LogLevel.ERROR)

    def test_unerrored_beam_always_outranks_errored_beam(self):
        agent = self._agent(beam_width=1)
        strong_errored = _beam("errored", errored=True, p_yes=(0.99,))
        weak_clean = _beam("clean", errored=False, p_yes=(0.01,), thought_lps=[-5.0], code_lps=[-5.0])
        survivors = agent._prune([strong_errored, weak_clean])
        self.assertEqual([b.beam_id for b in survivors], ["clean"])

    def test_ties_within_a_tier_broken_by_pruning_score(self):
        agent = self._agent(beam_width=1)
        high = _beam("high", p_yes=(0.9,))
        low = _beam("low", p_yes=(0.1,))
        survivors = agent._prune([low, high])
        self.assertEqual([b.beam_id for b in survivors], ["high"])

    def test_keeps_only_top_beam_width(self):
        agent = self._agent(beam_width=2)
        beams = [_beam(str(i), p_yes=(i / 10,)) for i in range(4)]
        survivors = agent._prune(beams)
        self.assertEqual([b.beam_id for b in survivors], ["3", "2"])

    def test_errored_beam_can_survive_when_not_enough_clean_ones(self):
        agent = self._agent(beam_width=2)
        clean = _beam("clean", errored=False, p_yes=(0.1,))
        errored_ok = _beam("errored_ok", errored=True, p_yes=(0.9,))
        errored_bad = _beam("errored_bad", errored=True, p_yes=(0.1,))
        survivors = agent._prune([clean, errored_ok, errored_bad])
        self.assertEqual([b.beam_id for b in survivors], ["clean", "errored_ok"])


class TestRunLoopFrozenBeamsAndWinnerSelection(unittest.TestCase):
    """run(): a finished beam stops branching but stays in the pool, and at the end any
    finished beam beats any unfinished one regardless of pruning_score (TODO.md decision 2)."""

    def _agent(self, beam_width=2, max_steps=20):
        return BeamSearchCodeAgent(
            tools=[], model=MagicMock(), beam_width=beam_width, max_steps=max_steps, verbosity_level=LogLevel.ERROR
        )

    def test_finished_beam_is_never_expanded_again(self):
        agent = self._agent(beam_width=2)
        root = _beam("root")
        finished_child = _beam("finished", is_final=True, final_output="7", p_yes=(0.9,))
        unfinished_child = _beam("unfinished", is_final=False, p_yes=(0.9,))
        finished_grandchild = _beam("finished2", is_final=True, final_output="99", p_yes=(0.99,))

        with (
            patch.object(agent, "_build_root_beam", return_value=root),
            patch.object(
                agent, "_expand_beam", side_effect=[[finished_child, unfinished_child], [finished_grandchild]]
            ) as expand,
        ):
            result = agent.run("task")

        expanded_ids = [call.args[0].beam_id for call in expand.call_args_list]
        self.assertEqual(expanded_ids, ["root", "unfinished"])
        self.assertEqual(result, "99")

    def test_finished_beam_wins_over_higher_scoring_unfinished_beam(self):
        agent = self._agent(beam_width=2, max_steps=1)
        root = _beam("root")
        finished_low = _beam("finished_low", is_final=True, final_output="low", p_yes=(0.01,))
        unfinished_high = _beam("unfinished_high", is_final=False, p_yes=(0.99,))

        with (
            patch.object(agent, "_build_root_beam", return_value=root),
            patch.object(agent, "_expand_beam", return_value=[finished_low, unfinished_high]),
        ):
            result = agent.run("task")

        self.assertEqual(result, "low")
        self.assertIs(agent.memory.steps[0], finished_low.memory.steps[0])

    def test_falls_back_to_best_unfinished_when_nothing_finished(self):
        agent = self._agent(beam_width=1, max_steps=1)
        root = _beam("root")
        low = _beam("low", is_final=False, p_yes=(0.1,), action_output="low_out")
        high = _beam("high", is_final=False, p_yes=(0.9,), action_output="high_out")

        with (
            patch.object(agent, "_build_root_beam", return_value=root),
            patch.object(agent, "_expand_beam", return_value=[low, high]),
        ):
            result = agent.run("task")

        self.assertEqual(result, "high_out")


if __name__ == "__main__":
    unittest.main()
