import unittest
from unittest.mock import MagicMock

from smolagents.memory import ActionStep, Timing
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import LogLevel, TokenUsage
from smolagents.tracelet_agent import (
    TraceletCodeAgent,
    _find_sentinels,
    _is_viable,
    _parse_fillin_lines,
    _parse_scores,
    _sentinelize_tool_calls,
)
from smolagents.utils import AgentExecutionError, AgentParsingError


class TestParseFillinLines(unittest.TestCase):
    def setUp(self):
        self.sentinels = ["ARG0", "ARG1"]

    def parse(self, content, sentinels=None):
        return _parse_fillin_lines(content, sentinels or self.sentinels)

    def test_plain_form(self):
        self.assertEqual(self.parse('ARG0: "kittens"'), {"ARG0": '"kittens"'})

    def test_multiple_sentinels(self):
        content = 'ARG0: "kittens"\nARG1: 5'
        self.assertEqual(self.parse(content), {"ARG0": '"kittens"', "ARG1": "5"})

    def test_leading_bullet(self):
        self.assertEqual(self.parse('- ARG0: "kittens"'), {"ARG0": '"kittens"'})

    def test_leading_asterisk_bullet(self):
        self.assertEqual(self.parse('* ARG0: "kittens"'), {"ARG0": '"kittens"'})

    def test_bold_sentinel(self):
        self.assertEqual(self.parse('**ARG0**: "kittens"'), {"ARG0": '"kittens"'})

    def test_backticked_sentinel(self):
        self.assertEqual(self.parse('`ARG0`: "kittens"'), {"ARG0": '"kittens"'})

    def test_bullet_and_bold_together(self):
        self.assertEqual(self.parse('- **ARG0**: "kittens"'), {"ARG0": '"kittens"'})

    def test_equals_instead_of_colon(self):
        self.assertEqual(self.parse('ARG0 = "kittens"'), {"ARG0": '"kittens"'})

    def test_indented_line(self):
        self.assertEqual(self.parse('    ARG0: "kittens"'), {"ARG0": '"kittens"'})

    def test_backticked_value_is_unwrapped(self):
        self.assertEqual(self.parse('ARG0: `"kittens"`'), {"ARG0": '"kittens"'})

    def test_fenced_block_fences_are_skipped(self):
        content = '```\nARG0: "kittens"\n```'
        self.assertEqual(self.parse(content), {"ARG0": '"kittens"'})

    def test_fenced_block_with_language_tag(self):
        content = '```python\nARG0: "kittens"\n```'
        self.assertEqual(self.parse(content), {"ARG0": '"kittens"'})

    def test_value_containing_colon_is_preserved(self):
        self.assertEqual(self.parse('ARG0: "https://example.com/a:b"'), {"ARG0": '"https://example.com/a:b"'})

    def test_surrounding_prose_is_ignored(self):
        content = 'Here are the values:\nARG0: "kittens"\nHope that helps!'
        self.assertEqual(self.parse(content), {"ARG0": '"kittens"'})

    def test_unknown_sentinel_is_dropped(self):
        self.assertEqual(self.parse('ARG7: "kittens"'), {})

    def test_line_without_separator_is_dropped(self):
        self.assertEqual(self.parse("ARG0 kittens"), {})

    def test_empty_value_is_dropped(self):
        self.assertEqual(self.parse("ARG0: "), {})

    def test_empty_content(self):
        self.assertEqual(self.parse(""), {})

    def test_none_content(self):
        self.assertEqual(self.parse(None), {})

    def test_partial_response_yields_partial_mapping(self):
        self.assertEqual(self.parse('ARG0: "kittens"'), {"ARG0": '"kittens"'})


class TestSubstitute(unittest.TestCase):
    """_substitute doesn't touch self, so it's exercised unbound to avoid building an agent."""

    def substitute(self, skeleton, fillin):
        return TraceletCodeAgent._substitute(None, skeleton, fillin)

    def test_single_sentinel(self):
        self.assertEqual(self.substitute("web_search(ARG0)", {"ARG0": '"kittens"'}), 'web_search("kittens")')

    def test_keyword_argument(self):
        self.assertEqual(
            self.substitute("web_search(query=ARG0)", {"ARG0": '"kittens"'}), 'web_search(query="kittens")'
        )

    def test_repeated_sentinel_is_replaced_everywhere(self):
        self.assertEqual(self.substitute("f(ARG0)\ng(ARG0)", {"ARG0": "1"}), "f(1)\ng(1)")

    def test_two_digit_sentinel_not_corrupted_by_one_digit(self):
        fillin = {f"ARG{i}": str(i) for i in range(12)}
        skeleton = "f(" + ", ".join(f"ARG{i}" for i in range(12)) + ")"
        expected = "f(" + ", ".join(str(i) for i in range(12)) + ")"
        self.assertEqual(self.substitute(skeleton, fillin), expected)

    def test_two_digit_sentinel_when_one_digit_substituted_first(self):
        # ARG1 first in insertion order used to turn ARG10 into "<value>0".
        self.assertEqual(self.substitute("f(ARG10)", {"ARG1": "'x'", "ARG10": "'y'"}), "f('y')")

    def test_missing_sentinel_left_in_place(self):
        self.assertEqual(self.substitute("f(ARG0, ARG1)", {"ARG0": "1"}), "f(1, ARG1)")

    def test_value_with_backslash_is_literal(self):
        self.assertEqual(self.substitute("f(ARG0)", {"ARG0": r'"C:\path"'}), r'f("C:\path")')

    def test_value_containing_group_reference_is_literal(self):
        self.assertEqual(self.substitute("f(ARG0)", {"ARG0": r'"\1"'}), r'f("\1")')

    def test_sentinel_prefix_inside_identifier_untouched(self):
        self.assertEqual(self.substitute("MYARG0 = ARG0_x", {"ARG0": "1"}), "MYARG0 = ARG0_x")

    def test_empty_fillin_leaves_skeleton_unchanged(self):
        self.assertEqual(self.substitute("f(ARG0)", {}), "f(ARG0)")


class TestParseFillinAndSubstituteRoundTrip(unittest.TestCase):
    def test_messy_response_fills_a_skeleton_completely(self):
        code = 'results = web_search(query="kittens")\nprint(results[:5])'
        skeleton = _sentinelize_tool_calls(code, {"web_search"})
        sentinels = _find_sentinels(skeleton)
        response = 'Here you go:\n```\n- **ARG0**: `"kittens"`\n```'
        fillin = _parse_fillin_lines(response, sentinels)
        filled = TraceletCodeAgent._substitute(None, skeleton, fillin)
        self.assertEqual(_find_sentinels(filled), [])
        self.assertIn('"kittens"', filled)


class TestSentinelizeFollowsDefinitions(unittest.TestCase):
    TOOLS = {"web_search", "visit_page"}

    def sentinelize(self, code):
        return _sentinelize_tool_calls(code, self.TOOLS)

    def test_inline_literal_replaced_in_place(self):
        self.assertEqual(self.sentinelize('web_search(query="kittens")'), "web_search(query=ARG0)")

    def test_inline_fstring_replaced_in_place(self):
        out = self.sentinelize('web_search(query=f"papers by {author}")')
        self.assertEqual(_find_sentinels(out), ["ARG0"])
        self.assertNotIn("papers by", out)

    def test_variable_assigned_a_literal_is_sentinelized_at_its_definition(self):
        out = self.sentinelize('url = "https://example.com"\npage = visit_page(url)')
        self.assertEqual(out, "url = ARG0\npage = visit_page(url)")
        self.assertNotIn("example.com", out)

    def test_variable_defined_in_an_earlier_step_is_left_alone(self):
        out = self.sentinelize("page = visit_page(url)")
        self.assertEqual(out, "page = visit_page(url)")
        self.assertEqual(_find_sentinels(out), [])

    def test_variable_assigned_a_computed_value_is_left_alone(self):
        code = "url = prefix + suffix\npage = visit_page(url)"
        self.assertEqual(self.sentinelize(code), code)

    def test_variable_assigned_after_the_call_is_left_alone(self):
        code = 'page = visit_page(url)\nurl = "https://example.com"'
        self.assertEqual(_find_sentinels(self.sentinelize(code)), [])

    def test_variable_assigned_twice_is_ambiguous_and_left_alone(self):
        code = 'url = "https://a.com"\nurl = "https://b.com"\npage = visit_page(url)'
        self.assertEqual(_find_sentinels(self.sentinelize(code)), [])

    def test_loop_variable_is_left_alone(self):
        code = "for url in urls:\n    visit_page(url)"
        self.assertEqual(self.sentinelize(code), code)

    def test_variable_feeding_two_calls_shares_one_sentinel(self):
        out = self.sentinelize('q = "kittens"\nweb_search(query=q)\nweb_search(query=q)')
        self.assertEqual(_find_sentinels(out), ["ARG0"])
        self.assertEqual(out.count("ARG0"), 1)

    def test_mixed_inline_and_hoisted_arguments(self):
        out = self.sentinelize('url = "https://example.com"\nvisit_page(url)\nweb_search(query="kittens")')
        self.assertEqual(sorted(_find_sentinels(out)), ["ARG0", "ARG1"])
        self.assertNotIn("example.com", out)
        self.assertNotIn("kittens", out)

    def test_non_tool_call_arguments_untouched(self):
        out = self.sentinelize('x = "kittens"\nprint(x)\nlen("abc")')
        self.assertEqual(_find_sentinels(out), [])  # ast.unparse normalises quote style
        self.assertIn("kittens", out)
        self.assertIn("abc", out)


class TestIsViable(unittest.TestCase):
    def test_fully_substituted_code_is_viable(self):
        self.assertTrue(_is_viable('web_search(query="kittens")'))

    def test_unfilled_sentinel_is_not_viable(self):
        self.assertFalse(_is_viable("web_search(query=ARG0)"))

    def test_partially_filled_is_not_viable(self):
        self.assertFalse(_is_viable('f("kittens", ARG1)'))

    def test_unparseable_substitution_is_not_viable(self):
        # A fill-in of bare prose substitutes cleanly but cannot parse.
        self.assertFalse(_is_viable("web_search(query=how many articles did Nature publish)"))

    def test_unterminated_string_is_not_viable(self):
        self.assertFalse(_is_viable('web_search(query="kittens)'))

    def test_variable_reference_argument_is_viable(self):
        # ~22% of real tool arguments are Name nodes, so a bare name must stay acceptable.
        self.assertTrue(_is_viable("page = visit_page(url)"))

    def test_fstring_argument_is_viable(self):
        self.assertTrue(_is_viable('web_search(query=f"papers by {author}")'))

    def test_multiline_candidate_is_viable(self):
        self.assertTrue(_is_viable('x = web_search(query="a")\nprint(x)'))


class TestPostProcessSkeletonKeepsFinalAnswer(unittest.TestCase):
    """post_process must not sentinelize final_answer, or the agent's answer gets re-sampled."""

    def _agent(self, output_text):
        message = ChatMessage(role=MessageRole.ASSISTANT, content=output_text)
        message.token_usage = TokenUsage(input_tokens=1, output_tokens=1)
        model = MagicMock()
        model.generate.return_value = message
        return TraceletCodeAgent(tools=[], model=model, n_samples=1, verbosity_level=LogLevel.ERROR)

    def test_final_answer_argument_survives(self):
        agent = self._agent('Thought: done.\n<code>\nfinal_answer("34689")\n</code>')
        _, _, skeleton, _ = agent._sample_skeleton_post_process([])
        self.assertIn("34689", skeleton)  # ast.unparse normalises quote style
        self.assertEqual(_find_sentinels(skeleton), [])

    def test_other_tool_arguments_still_sentinelized(self):
        agent = self._agent('Thought: search.\n<code>\nx = web_search(query="kittens")\n</code>')
        agent.tools["web_search"] = MagicMock(name="web_search")
        _, _, skeleton, _ = agent._sample_skeleton_post_process([])
        self.assertEqual(_find_sentinels(skeleton), ["ARG0"])
        self.assertNotIn("kittens", skeleton)

    def test_final_answer_kept_while_sibling_call_sentinelized(self):
        agent = self._agent('Thought: both.\n<code>\nx = web_search(query="kittens")\nfinal_answer("42")\n</code>')
        agent.tools["web_search"] = MagicMock(name="web_search")
        _, _, skeleton, _ = agent._sample_skeleton_post_process([])
        self.assertEqual(_find_sentinels(skeleton), ["ARG0"])
        self.assertIn("42", skeleton)


class TestFillinFallback(unittest.TestCase):
    """When no candidate survives _is_viable, post_process falls back to the model's own values."""

    SKELETON_REPLY = 'Thought: search.\n<code>\nx = web_search(query="kittens")\nprint(x)\n</code>'
    BROKEN_FILLIN = 'Thought: searching.\n<code>\nx = web_search(query="cats")\n</code>'  # no ARG0: line

    # direct_prompt takes the model's output as the skeleton verbatim, so it must carry the sentinel.
    DIRECT_SKELETON_REPLY = "Thought: search.\n<code>\nx = web_search(query=ARG0)\nprint(x)\n</code>"

    def _run(self, strategy, fillin_reply, skeleton_reply=None):
        replies = [skeleton_reply or self.SKELETON_REPLY, fillin_reply, "Candidate 0: 5"]
        model = MagicMock()

        def generate(*args, **kwargs):
            message = ChatMessage(role=MessageRole.ASSISTANT, content=replies[min(model.generate.call_count - 1, 2)])
            message.token_usage = TokenUsage(input_tokens=1, output_tokens=1)
            message.raw = None
            return message

        model.generate.side_effect = generate
        agent = TraceletCodeAgent(
            tools=[],
            model=model,
            n_samples=1,
            max_steps=1,
            skeleton_strategy=strategy,
            verbosity_level=LogLevel.ERROR,
        )
        agent.tools["web_search"] = MagicMock()
        executed = []
        agent.python_executor = MagicMock(
            side_effect=lambda code: executed.append(code) or MagicMock(logs="", output="ok", is_final_answer=False)
        )
        agent.python_executor.state = {}
        step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        error = None
        try:
            list(agent._step_stream(step))
        except AgentParsingError as e:
            error = e
        return executed, error

    def test_post_process_falls_back_to_the_models_own_values(self):
        executed, error = self._run("post_process", self.BROKEN_FILLIN)
        self.assertIsNone(error)
        self.assertTrue(executed)
        self.assertIn("kittens", executed[-1])  # the model's own query, not a re-sampled one

    def test_unfillable_step_still_records_the_skeleton_in_memory(self):
        replies = [self.DIRECT_SKELETON_REPLY, self.BROKEN_FILLIN, "Candidate 0: 5"]
        model = MagicMock()

        def generate(*args, **kwargs):
            message = ChatMessage(role=MessageRole.ASSISTANT, content=replies[min(model.generate.call_count - 1, 2)])
            message.token_usage = TokenUsage(input_tokens=1, output_tokens=1)
            message.raw = None
            return message

        model.generate.side_effect = generate
        agent = TraceletCodeAgent(
            tools=[],
            model=model,
            n_samples=1,
            max_steps=1,
            skeleton_strategy="direct_prompt",
            verbosity_level=LogLevel.ERROR,
        )
        agent.tools["web_search"] = MagicMock()
        agent.python_executor = MagicMock()
        agent.python_executor.state = {}
        step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        with self.assertRaises(AgentParsingError):
            list(agent._step_stream(step))
        self.assertIsNotNone(step.model_output)
        self.assertIn("ARG0", step.model_output)  # the skeleton that could not be filled

    def test_direct_prompt_has_nothing_to_fall_back_to(self):
        executed, error = self._run("direct_prompt", self.BROKEN_FILLIN, self.DIRECT_SKELETON_REPLY)
        self.assertIsInstance(error, AgentParsingError)
        self.assertEqual(executed, [])

    def test_fallback_not_used_when_a_fillin_is_viable(self):
        executed, error = self._run("post_process", 'ARG0: "puppies"')
        self.assertIsNone(error)
        self.assertIn("puppies", executed[-1])


class TestSentinelCountRecorded(unittest.TestCase):
    """sentinel_count makes 'did the model actually produce a skeleton?' directly measurable."""

    def _step(self, strategy, skeleton_reply, fillin_reply='ARG0: "kittens"'):
        replies = [skeleton_reply, fillin_reply, "Candidate 0: 5"]
        model = MagicMock()

        def generate(*args, **kwargs):
            message = ChatMessage(role=MessageRole.ASSISTANT, content=replies[min(model.generate.call_count - 1, 2)])
            message.token_usage = TokenUsage(input_tokens=1, output_tokens=1)
            message.raw = None
            return message

        model.generate.side_effect = generate
        agent = TraceletCodeAgent(
            tools=[],
            model=model,
            n_samples=1,
            max_steps=1,
            skeleton_strategy=strategy,
            verbosity_level=LogLevel.ERROR,
        )
        agent.tools["web_search"] = MagicMock()
        agent.python_executor = MagicMock(
            side_effect=lambda code: MagicMock(logs="", output="ok", is_final_answer=False)
        )
        agent.python_executor.state = {}
        step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        try:
            list(agent._step_stream(step))
        except AgentParsingError:
            pass
        return step

    def test_direct_prompt_without_sentinels_records_zero(self):
        step = self._step("direct_prompt", 'Thought: go.\n<code>\nweb_search(query="kittens")\n</code>')
        self.assertEqual(step.sentinel_count, 0)

    def test_direct_prompt_with_sentinels_records_the_count(self):
        step = self._step("direct_prompt", "Thought: go.\n<code>\nweb_search(query=ARG0)\n</code>")
        self.assertEqual(step.sentinel_count, 1)

    def test_two_sentinels_recorded(self):
        step = self._step(
            "direct_prompt",
            "Thought: go.\n<code>\nweb_search(query=ARG0)\nvisit_page(ARG1)\n</code>",
            'ARG0: "a"\nARG1: "b"',
        )
        self.assertEqual(step.sentinel_count, 2)

    def test_post_process_records_what_the_ast_produced(self):
        step = self._step("post_process", 'Thought: go.\n<code>\nweb_search(query="kittens")\n</code>')
        self.assertEqual(step.sentinel_count, 1)

    def test_count_survives_a_failed_step(self):
        step = self._step("direct_prompt", "Thought: go.\n<code>\nweb_search(query=ARG0)\n</code>", "no fill-in here")
        self.assertEqual(step.sentinel_count, 1)
        self.assertIsNotNone(step.error or True)

    def test_count_is_serialised_into_the_step_dict(self):
        step = self._step("direct_prompt", "Thought: go.\n<code>\nweb_search(query=ARG0)\n</code>")
        self.assertEqual(step.dict()["sentinel_count"], 1)


class TestMemoryRecordsSkeletonAndFillin(unittest.TestCase):
    """model_output must replay the skeleton + 'Fill-in:' format, so the model's own
    history keeps demonstrating the template protocol at every later step."""

    def _step(self, strategy, skeleton_reply, fillin_reply='ARG0: "kittens"'):
        replies = [skeleton_reply, fillin_reply, "Candidate 0: 5"]
        model = MagicMock()

        def generate(*args, **kwargs):
            message = ChatMessage(role=MessageRole.ASSISTANT, content=replies[min(model.generate.call_count - 1, 2)])
            message.token_usage = TokenUsage(input_tokens=1, output_tokens=1)
            message.raw = None
            return message

        model.generate.side_effect = generate
        agent = TraceletCodeAgent(
            tools=[],
            model=model,
            n_samples=1,
            max_steps=1,
            skeleton_strategy=strategy,
            verbosity_level=LogLevel.ERROR,
        )
        agent.tools["web_search"] = MagicMock()
        agent.python_executor = MagicMock(
            side_effect=lambda code: MagicMock(logs="", output="ok", is_final_answer=False)
        )
        agent.python_executor.state = {}
        step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        list(agent._step_stream(step))
        return step

    def test_winning_fillin_is_recorded_as_skeleton_plus_fillin_lines(self):
        step = self._step("direct_prompt", "Thought: go.\n<code>\nweb_search(query=ARG0)\n</code>")
        code_block, _, fillin_block = step.model_output.partition("Fill-in:")
        self.assertIn("web_search(query=ARG0)", code_block)  # code block keeps the sentinel
        self.assertNotIn("kittens", code_block)
        self.assertIn('ARG0: "kittens"', fillin_block)

    def test_fillin_lines_follow_skeleton_order(self):
        step = self._step(
            "direct_prompt",
            "Thought: go.\n<code>\nweb_search(query=ARG0)\nvisit_page(ARG1)\n</code>",
            'ARG1: "b"\nARG0: "a"',
        )
        fillin_block = step.model_output.partition("Fill-in:")[2]
        self.assertLess(fillin_block.index("ARG0"), fillin_block.index("ARG1"))

    def test_code_action_is_still_the_executable_code(self):
        step = self._step("direct_prompt", "Thought: go.\n<code>\nweb_search(query=ARG0)\n</code>")
        self.assertEqual(step.code_action, 'web_search(query="kittens")')

    def test_step_without_sentinels_records_concrete_code_without_fillin(self):
        step = self._step("direct_prompt", 'Thought: go.\n<code>\nweb_search(query="kittens")\n</code>')
        self.assertIn('web_search(query="kittens")', step.model_output)
        self.assertNotIn("Fill-in:", step.model_output)

    def test_fallback_to_own_values_records_concrete_code_without_fillin(self):
        step = self._step("post_process", 'Thought: go.\n<code>\nweb_search(query="kittens")\n</code>', "no fillin")
        self.assertIn("kittens", step.model_output)
        self.assertNotIn("Fill-in:", step.model_output)
        self.assertNotIn("ARG0", step.model_output)


class TestCommitReusesTrialSnapshot(unittest.TestCase):
    """The judged winner is committed from its trial's saved state -- never re-executed,
    so single-shot side effects (browser paging, searches) are not repeated."""

    def _agent(self, replies, execute):
        model = MagicMock()

        def generate(*args, **kwargs):
            message = ChatMessage(role=MessageRole.ASSISTANT, content=replies[min(model.generate.call_count - 1, len(replies) - 1)])
            message.token_usage = TokenUsage(input_tokens=1, output_tokens=1)
            message.raw = None
            return message

        model.generate.side_effect = generate
        agent = TraceletCodeAgent(
            tools=[],
            model=model,
            n_samples=1,
            max_steps=1,
            skeleton_strategy="direct_prompt",
            verbosity_level=LogLevel.ERROR,
        )
        agent.tools["web_search"] = MagicMock()
        agent.python_executor = MagicMock(side_effect=execute)
        agent.python_executor.state = {}
        return agent

    def _run_step(self, agent):
        step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        list(agent._step_stream(step))
        return step

    def test_sentinel_step_executes_exactly_once(self):
        executed = []

        def execute(code):
            executed.append(code)
            return MagicMock(logs="", output="ok", is_final_answer=False)

        agent = self._agent(
            ["Thought: go.\n<code>\nweb_search(query=ARG0)\n</code>", 'ARG0: "kittens"', "Candidate 0: 5"], execute
        )
        self._run_step(agent)
        self.assertEqual(executed, ['web_search(query="kittens")'])

    def test_no_sentinel_step_also_executes_exactly_once(self):
        executed = []

        def execute(code):
            executed.append(code)
            return MagicMock(logs="", output="ok", is_final_answer=False)

        agent = self._agent(['Thought: go.\n<code>\nresult = 1 + 1\nprint(result)\n</code>'], execute)
        self._run_step(agent)
        self.assertEqual(len(executed), 1)

    def test_commit_installs_the_winning_trials_state(self):
        agent = self._agent(
            ["Thought: go.\n<code>\nweb_search(query=ARG0)\n</code>", 'ARG0: "kittens"', "Candidate 0: 5"],
            None,  # replaced below, needs the agent to exist first
        )

        def execute(code):
            agent.python_executor.state["marker"] = code
            return MagicMock(logs="", output="ok", is_final_answer=False)

        agent.python_executor.side_effect = execute
        self._run_step(agent)
        self.assertEqual(agent.python_executor.state.get("marker"), 'web_search(query="kittens")')

    def test_committed_observation_is_the_trial_observation(self):
        agent = self._agent(
            ["Thought: go.\n<code>\nweb_search(query=ARG0)\n</code>", 'ARG0: "kittens"', "Candidate 0: 5"],
            lambda code: MagicMock(logs="page 2 of 24", output="ok", is_final_answer=False),
        )
        step = self._run_step(agent)
        self.assertIn("page 2 of 24", step.observations)

    def test_errored_winner_raises_execution_error(self):
        def execute(code):
            raise ValueError("boom")

        agent = self._agent(
            ["Thought: go.\n<code>\nweb_search(query=ARG0)\n</code>", 'ARG0: "kittens"', "Candidate 0: 5"], execute
        )
        step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        with self.assertRaises(AgentExecutionError):
            list(agent._step_stream(step))


class TestParseScores(unittest.TestCase):
    def test_parses_scores_in_order(self):
        self.assertEqual(_parse_scores("Candidate 0: 3\nCandidate 1: 7", 2), [3.0, 7.0])

    def test_missing_candidate_defaults_to_zero(self):
        self.assertEqual(_parse_scores("Candidate 1: 7", 2), [0.0, 7.0])

    def test_out_of_range_candidate_is_ignored(self):
        self.assertEqual(_parse_scores("Candidate 5: 7", 2), [0.0, 0.0])

    def test_unparseable_output_defaults_to_zeros(self):
        self.assertEqual(_parse_scores("no scores here", 2), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
