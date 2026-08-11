import ast
import copy
import re
from typing import Any, Generator, Literal

from .agents import ActionOutput, CodeAgent, ToolCall
from .local_python_executor import fix_final_answer_code
from .memory import ActionStep
from .models import ChatMessage, MessageRole, agglomerate_stream_deltas, agglomerate_stream_deltas_by_index
from .monitoring import LogLevel, TokenUsage
from .utils import AgentExecutionError, AgentParsingError, parse_code_blobs, truncate_content

# Prefix identifying a sentinel a code skeleton uses in place of a tool-call argument
# value, to be filled in by _sample_arg_fillins. Sentinels are indexed (ARG0, ARG1, ...)
# so a tool call with multiple arguments gets one sentinel each.
ARG_SENTINEL_PREFIX = "ARG"


def _make_sentinel(index: int) -> str:
    return f"{ARG_SENTINEL_PREFIX}{index}"


# Used by the "direct_prompt" skeleton strategy: instead of generating real code and
# post-processing it (_sentinelize_tool_calls), ask the model to emit the sentinel-marked
# skeleton itself. No structural guarantee like the AST-based approach -- relies on the
# model reliably following this instruction.
_SENTINEL_INSTRUCTION = (
    "For this step's code, whenever you call one of your provided tools, replace the "
    "value of each argument to that tool call with a distinct sentinel identifier of the "
    f"form {_make_sentinel(0)}, {_make_sentinel(1)}, ... -- numbered in the order the "
    "arguments appear across the whole snippet. The sentinel completely replaces the "
    "value -- it is not a label, tag, or keyword name attached to the real value, and the "
    "real value must not appear anywhere in the code.\n\n"
    f"For a positional argument, write `web_search({_make_sentinel(0)})` instead of "
    f"`web_search(\"some query\")`.\n"
    f"For a keyword argument, keep the original keyword name and replace only the value: "
    f"write `web_search(query={_make_sentinel(0)})` instead of "
    f"`web_search(query=\"some query\")`.\n\n"
    f"Do NOT write `web_search({_make_sentinel(0)}=\"some query\")` -- that invents a fake "
    "keyword name out of the sentinel and still leaves the real value in the code, which "
    "defeats the purpose.\n\n"
    "Leave everything else in the code as normal: variable assignments, print statements, "
    "control flow, and which tool you call. Only replace arguments passed directly to "
    "tool calls; do not sentinel arguments to other functions (e.g. print, len)."
)


def _safe_deepcopy(value: Any) -> Any:
    """Deepcopy value, falling back to the original reference if it cannot be copied
    (e.g. modules, C extensions). Same pattern as probabilistic_agent.py."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


class _ToolArgSentinelizer(ast.NodeTransformer):
    """Replaces every argument value in calls to a known tool/managed-agent name with an
    indexed sentinel identifier. Leaves calls to anything else (print, len, helper
    variables, ...) untouched, including tool calls nested inside their arguments."""

    def __init__(self, tool_names: set[str]):
        self.tool_names = tool_names
        self.count = 0

    def _next_sentinel(self) -> ast.Name:
        node = ast.Name(id=_make_sentinel(self.count), ctx=ast.Load())
        self.count += 1
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name) and node.func.id in self.tool_names:
            node.args = [ast.copy_location(self._next_sentinel(), arg) for arg in node.args]
            for keyword in node.keywords:
                keyword.value = ast.copy_location(self._next_sentinel(), keyword.value)
            return node
        self.generic_visit(node)
        return node


def _sentinelize_tool_calls(code: str, tool_names: set[str]) -> str:
    """Parse code, replace tool-call arguments with sentinels, and unparse it back."""
    tree = ast.parse(code)
    _ToolArgSentinelizer(tool_names).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _find_sentinels(code: str) -> list[str]:
    """Sentinel names appearing in code, in order of first appearance, deduplicated."""
    seen: list[str] = []
    pattern = rf"{re.escape(ARG_SENTINEL_PREFIX)}\d+"
    for match in re.finditer(pattern, code):
        if match.group(0) not in seen:
            seen.append(match.group(0))
    return seen


def _parse_fillin_lines(content: str, sentinels: list[str]) -> dict[str, str]:
    """Parse a completion of the form '<sentinel>: <literal>' (one line per sentinel)
    into a {sentinel: literal} mapping. Lines for unknown or malformed sentinels are
    dropped rather than raising, since a missing entry surfaces naturally as
    _substitute leaving that sentinel unreplaced."""
    values: dict[str, str] = {}
    for line in (content or "").splitlines():
        name, sep, value = line.partition(":")
        name = name.strip()
        if sep and name in sentinels:
            values[name] = value.strip()
    return values


def _parse_scores(judge_output: str, n: int) -> list[float]:
    """Parse 'Candidate <i>: <score>' lines into a list of length n, defaulting to 0.0 for
    any candidate the judge didn't score or scored unparseably."""
    scores = [0.0] * n
    for line in (judge_output or "").splitlines():
        match = re.match(r"\s*Candidate\s+(\d+)\s*:\s*([\d.]+)", line)
        if match:
            index, score = int(match.group(1)), float(match.group(2))
            if 0 <= index < n:
                scores[index] = score
    return scores


class TraceletCodeAgent(CodeAgent):
    """
    A tracelet agent that samples multiple arguments for tool calls in agentic workflows
    """

    def __init__(
        self,
        *args,
        n_samples: int = 3,
        skeleton_strategy: Literal["post_process", "direct_prompt"] = "post_process",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples
        self.skeleton_strategy = skeleton_strategy

    def _step_stream(self, memory_step: ActionStep) -> Generator:
        """Orchestrates one Tracelet step.

        1. Sample (thought, code_skeleton): code_skeleton has tool-call arguments
           replaced by indexed sentinels (see ARG_SENTINEL_PREFIX).
        2. If no sentinel is present (no tool call with arguments in this step), commit
           the skeleton directly -- same as plain CodeAgent, one execution.
        3. Otherwise, sample n argument fill-ins, substitute each into the skeleton,
           trial-execute all n candidates, and use an LLM judge to pick a winner.
        4. Commit the winning code (a single real execution) and yield the
           ToolCall/ActionOutput.
        """
        memory_messages = self.write_memory_to_messages()
        memory_step.model_input_messages = memory_messages

        try:
            thought, code, code_skeleton, usage = self._sample_skeleton(memory_messages)
        except AgentParsingError as e:
            # The billed call still counts, and the unparseable output must stay visible in memory.
            if getattr(e, "token_usage", None) is not None:
                memory_step.token_usage = e.token_usage
            memory_step.model_output = getattr(e, "raw_output", None)
            raise
        total_input, total_output = usage.input_tokens, usage.output_tokens

        if ARG_SENTINEL_PREFIX not in code_skeleton:
            winning_code = code
            self.logger.log(
                "[Tracelet] no tool-call arguments to sample this step -- direct execution "
                "(no fill-in sampling, no judge).",
                level=LogLevel.INFO,
            )
        else:
            # TODO: code (the model's own real argument values) could be reused as a free
            # first candidate here instead of only sampling n fresh fill-ins.
            num_sentinels = len(_find_sentinels(code_skeleton))
            self.logger.log(
                f"[Tracelet] {num_sentinels} sentinel(s) found -- sampling {self.n_samples} candidates.",
                level=LogLevel.INFO,
            )
            fillins, fillin_usage = self._sample_arg_fillins(memory_messages, code_skeleton, self.n_samples)
            total_input += fillin_usage.input_tokens
            total_output += fillin_usage.output_tokens

            candidates = [self._substitute(code_skeleton, fillin) for fillin in fillins]
            trials = self._execute_candidates(candidates)  # list of (code, observation)

            winner_index, judge_usage = self._judge_select(thought, trials)
            total_input += judge_usage.input_tokens
            total_output += judge_usage.output_tokens

            winning_code = candidates[winner_index]

        memory_step.token_usage = TokenUsage(input_tokens=total_input, output_tokens=total_output)

        # Record the action before executing it, so a failed execution still leaves it in memory.
        memory_step.model_output = f"{thought}\n{self.code_block_tags[0]}\n{winning_code}\n{self.code_block_tags[1]}"
        memory_step.code_action = winning_code
        tool_call = ToolCall(
            name="python_interpreter",
            arguments=winning_code,
            id=f"call_{len(self.memory.steps)}",
        )
        memory_step.tool_calls = [tool_call]
        yield tool_call

        _, observation, action_output, is_final_answer = self._commit(winning_code)
        memory_step.observations = observation
        memory_step.action_output = action_output
        yield ActionOutput(output=action_output, is_final_answer=is_final_answer)

    # --- Building blocks below: not yet implemented ---

    def _sample_skeleton(self, memory_messages: list[ChatMessage]) -> tuple[str, str, str, TokenUsage]:
        """Sample (thought, code, code_skeleton, token_usage): dispatches to one of two
        strategies based on self.skeleton_strategy (see _sample_skeleton_post_process and
        _sample_skeleton_direct_prompt for the tradeoffs)."""
        if self.skeleton_strategy == "direct_prompt":
            return self._sample_skeleton_direct_prompt(memory_messages)
        return self._sample_skeleton_post_process(memory_messages)

    def _generate(self, messages: list[ChatMessage], **kwargs) -> ChatMessage:
        """Single-completion model call that transparently uses streaming when
        self.stream_outputs is set. Some backends (e.g. Qwen served via TogetherAI)
        reject non-streaming requests outright with 'This model only supports
        streaming' -- self.model.generate(...) would fail on them unconditionally.
        Collects the stream and agglomerates it into one ChatMessage, matching what
        generate() returns directly, so every other method here can stay agnostic to
        which path was taken. Unlike CodeAgent._step_stream, doesn't render a live
        console view of the stream -- deltas are collected, not surfaced to the caller."""
        if self.stream_outputs:
            deltas = list(self.model.generate_stream(messages, **kwargs))
            return agglomerate_stream_deltas(deltas)
        return self.model.generate(messages, **kwargs)

    def _sample_skeleton_post_process(self, memory_messages: list[ChatMessage]) -> tuple[str, str, str, TokenUsage]:
        """Sample (thought, code, code_skeleton, token_usage): one ordinary LLM call,
        identical to a plain CodeAgent step -- no special instruction. code is the raw
        generated code with real argument values; code_skeleton is the post-processed
        version with every tool-call argument replaced by an indexed sentinel (see
        _sentinelize_tool_calls). code is returned too so its real argument values can be
        reused later -- e.g. as a free first candidate -- instead of discarding them.

        Structural guarantee: sentinel placement comes from an AST walk against the real
        tool registry, not from the model following an instruction. See
        _sample_skeleton_direct_prompt for the alternative."""
        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            stop_sequences.append(self.code_block_tags[1])

        response = self._generate(memory_messages, stop_sequences=stop_sequences)
        output_text = response.content or ""
        if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
            output_text += self.code_block_tags[1]

        try:
            code = fix_final_answer_code(parse_code_blobs(output_text, self.code_block_tags))
            tool_names = set(self.tools) | set(self.managed_agents)
            code_skeleton = _sentinelize_tool_calls(code, tool_names)
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            parsing_error = AgentParsingError(error_msg, self.logger)
            parsing_error.token_usage = response.token_usage  # billed call -- don't let _step_stream lose it
            parsing_error.raw_output = output_text  # keep the failed attempt visible in memory
            raise parsing_error from e

        thought = re.split(self.code_block_tags[0], output_text, maxsplit=1)[0].strip()
        return thought, code, code_skeleton, response.token_usage

    def _sample_skeleton_direct_prompt(self, memory_messages: list[ChatMessage]) -> tuple[str, str, str, TokenUsage]:
        """Sample (thought, code, code_skeleton, token_usage): one LLM call with an added
        instruction (_SENTINEL_INSTRUCTION) asking the model to emit the sentinel-marked
        skeleton directly, instead of generating real code and post-processing it.

        code and code_skeleton are identical here -- there's no separately-generated
        real-valued version, so unlike _sample_skeleton_post_process there's nothing to
        reuse as a free first candidate. Also weaker than the post-process strategy: sentinel
        placement depends on the model reliably following the instruction, with no
        structural guarantee."""
        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            stop_sequences.append(self.code_block_tags[1])

        messages = memory_messages + [
            ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": _SENTINEL_INSTRUCTION}])
        ]
        response = self._generate(messages, stop_sequences=stop_sequences)
        output_text = response.content or ""
        if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
            output_text += self.code_block_tags[1]

        try:
            code_skeleton = fix_final_answer_code(parse_code_blobs(output_text, self.code_block_tags))
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            parsing_error = AgentParsingError(error_msg, self.logger)
            parsing_error.token_usage = response.token_usage  # billed call -- don't let _step_stream lose it
            parsing_error.raw_output = output_text  # keep the failed attempt visible in memory
            raise parsing_error from e

        thought = re.split(self.code_block_tags[0], output_text, maxsplit=1)[0].strip()
        return thought, code_skeleton, code_skeleton, response.token_usage

    def _sample_arg_fillins(
        self, memory_messages: list[ChatMessage], code_skeleton: str, n: int
    ) -> tuple[list[dict[str, str]], TokenUsage]:
        """Sample n candidate fill-ins for code_skeleton's sentinels, in one API call using
        the provider's `n` parameter (n completions sharing one billed prompt).

        Each fillin is a {sentinel: literal} dict covering every sentinel in the skeleton.

        Non-streaming path: reads the extra completions off the raw API response
        (ChatMessage.raw) since Model.generate() only surfaces choices[0] itself -- works
        for OpenAI-compatible backends, not local backends like TransformersModel/MLXModel.

        Streaming path (self.stream_outputs): a request with n>1 interleaves n parallel
        completions over one stream, distinguished by each delta's index -- see
        agglomerate_stream_deltas_by_index (models.py) and the corresponding change in
        OpenAIModel.generate_stream, which used to only ever surface choices[0], silently
        dropping n-1 completions. Demultiplexing here means streaming-only backends (e.g.
        Qwen models served via TogetherAI) still get the "prompt billed once" efficiency
        of the n= path, instead of falling back to n separate single-completion calls.
        """
        sentinels = _find_sentinels(code_skeleton)
        instruction = (
            "The following code has tool-call argument values replaced by sentinels:\n\n"
            f"```python\n{code_skeleton}\n```\n\n"
            "For each sentinel below, respond with exactly one line of the form "
            "`<sentinel>: <value>`, where <value> is a valid Python literal to substitute "
            "for it. Do not include any other text.\n" + "\n".join(sentinels)
        )
        messages = memory_messages + [
            ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": instruction}])
        ]

        if self.stream_outputs:
            deltas = list(self.model.generate_stream(messages, n=n))
            messages_by_index, token_usage = agglomerate_stream_deltas_by_index(deltas)
            contents = [messages_by_index[i].content for i in sorted(messages_by_index)]
        else:
            response = self.model.generate(messages, n=n)
            raw_choices = getattr(response.raw, "choices", None)
            contents = [choice.message.content for choice in raw_choices] if raw_choices else [response.content]
            token_usage = response.token_usage

        fillins = [_parse_fillin_lines(content, sentinels) for content in contents]
        return fillins, token_usage

    def _substitute(self, code_skeleton: str, fillin: dict[str, str]) -> str:
        """Replace each sentinel in code_skeleton with its value from fillin. Pure string
        substitution, no LLM call. A sentinel missing from fillin (the model dropped a
        line) is left in place -- it surfaces naturally as a NameError when the candidate
        is executed, handled like any other per-candidate execution failure."""
        code = code_skeleton
        for sentinel, value in fillin.items():
            code = code.replace(sentinel, value)
        return code

    def _execute_candidates(self, candidates: list[str]) -> list[tuple[str, str]]:
        """Trial-execute each candidate against a snapshot/restore of executor state (side
        effects included -- e.g. a real web search per candidate). Returns a list of
        (code, observation) pairs for the judge; does not mutate live executor state."""
        snapshot = {k: _safe_deepcopy(v) for k, v in self.python_executor.state.items()}
        trials = []
        try:
            for code in candidates:
                self.python_executor.state.clear()
                self.python_executor.state.update({k: _safe_deepcopy(v) for k, v in snapshot.items()})
                try:
                    code_output = self.python_executor(code)
                    observation = "Execution logs:\n" + code_output.logs
                    observation += "\nLast output from code snippet:\n" + truncate_content(str(code_output.output))
                except Exception as e:
                    observation = f"Error: {e}"
                trials.append((code, observation))
        finally:
            self.python_executor.state.clear()
            self.python_executor.state.update(snapshot)
        return trials

    def _score_trials(self, thought: str, trials: list[tuple[str, str]]) -> tuple[str, TokenUsage]:
        """One LLM call: given the current thought and the n (code, observation) trial
        pairs, ask whether each represents significant progress toward the task and score
        it. Conditioned on the system prompt + task only, not the full running memory --
        keeps the judge's context small regardless of how large memory has grown.
        Returns the judge's raw text output (parsed separately by _pick_best) and
        token_usage."""
        trials_text = "\n\n".join(
            f"Candidate {i}:\nThought: {thought}\nAction:\n```python\n{code}\n```\nObservation:\n{observation}"
            for i, (code, observation) in enumerate(trials)
        )
        instruction = (
            f"Task: {self.task}\n\n"
            "Below are candidate next steps for this task -- each pairs the same thought "
            "with a different action and its resulting observation.\n\n"
            f"{trials_text}\n\n"
            "For each candidate, judge whether it represents significant progress toward "
            "completing the task. Respond with exactly one line per candidate, in order, "
            "of the form `Candidate <i>: <score>`, where <score> is a number from 0 (no "
            "progress -- e.g. an error or irrelevant result) to 10 (major progress -- e.g. "
            "found the final answer). Do not include any other text."
        )
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}]),
            ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": instruction}]),
        ]
        response = self._generate(messages)
        return response.content or "", response.token_usage

    def _pick_best(self, judge_output: str, n: int) -> int:
        """Aggregate: parse the judge's per-candidate scores and return the index of the
        highest-scoring one. Ties, and total parse failure (all scores default to 0.0),
        both resolve to candidate 0."""
        scores = _parse_scores(judge_output, n)
        return max(range(n), key=lambda i: scores[i])

    def _judge_select(self, thought: str, trials: list[tuple[str, str]]) -> tuple[int, TokenUsage]:
        """Score each trial via one LLM call, then pick the highest-scoring candidate."""
        judge_output, token_usage = self._score_trials(thought, trials)
        winner_index = self._pick_best(judge_output, len(trials))
        scores = _parse_scores(judge_output, len(trials))
        self.logger.log(
            f"[Tracelet] judge scores: {scores} -- picked candidate {winner_index}.",
            level=LogLevel.INFO,
        )
        return winner_index, token_usage

    def _commit(self, code: str) -> tuple[str, str, Any, bool]:
        """Execute code for real against live executor state (the one authoritative
        execution for this step, unlike the trial runs in _execute_candidates). Returns
        (code_action, observation, action_output, is_final_answer). Unlike a trial's
        caught-and-scored failure, an error here propagates as an AgentExecutionError --
        this is the committed action, not one candidate among many."""
        self.logger.log_code(title="Executing parsed code:", content=code, level=LogLevel.INFO)
        try:
            code_output = self.python_executor(code)
        except Exception as e:
            raise AgentExecutionError(str(e), self.logger) from e

        observation = "Execution logs:\n" + code_output.logs
        observation += "\nLast output from code snippet:\n" + truncate_content(str(code_output.output))
        return code, observation, code_output.output, code_output.is_final_answer