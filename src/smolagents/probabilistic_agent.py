import copy
from typing import Any, Generator


def _safe_deepcopy(value: Any) -> Any:
    """Deepcopy ``value``, falling back to the original reference if it cannot be copied (e.g. modules, C extensions)."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value

from .agents import ActionOutput, CodeAgent, ToolCall
from .local_python_executor import fix_final_answer_code
from .memory import ActionStep
from .models import ChatMessage, MessageRole
from .utils import AgentExecutionError, AgentGenerationError, AgentParsingError, parse_code_blobs, truncate_content


class ProbabilisticCodeAgent(CodeAgent):
    """CodeAgent where each step samples diverse tool-call arguments before committing.

    For each step:
      1. Generate one thought + code to identify the intended tool call.
      2. Extract the first tool call from the code as a template.
      3. Generate ``n_samples`` alternative argument sets for that tool call.
      4. If the argument sets are sufficiently diverse, pick one and execute.
         Otherwise keep sampling until diversity is reached or budget is exhausted.
    """

    def __init__(self, *args, n_samples: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

    def _extract_prefix_tool_call(self, code: str) -> tuple[str, int] | None:
        """Return (tool_name, open_paren_pos) for the first known tool call in ``code``.

        Scans for the earliest occurrence of any known tool name followed by ``(``.
        Returns ``None`` if no known tool call is found.
        """
        first_pos = len(code)
        first_tool = None
        for name in self.tools:
            if name == "final_answer":
                continue
            pattern = name + "("
            idx = code.find(pattern)
            if idx != -1 and idx < first_pos:
                first_pos = idx
                first_tool = name

        if first_tool is None:
            return None

        open_paren_pos = first_pos + len(first_tool)  # index of '('

        # If the first argument is a keyword arg, extend prefix to right after '='
        # so completions diverge on the value rather than on the argument name.
        after_paren = code[open_paren_pos + 1:]
        eq_idx = after_paren.find("=")
        first_end = min((i for i in [after_paren.find(","), after_paren.find(")")] if i != -1), default=len(after_paren))
        if eq_idx != -1 and eq_idx < first_end:
            return first_tool, open_paren_pos + 1 + eq_idx  # index of '=' in code

        return first_tool, open_paren_pos

    def _generate_completions(
        self, memory_messages: list, output_prefix: str, n: int
    ) -> list[str]:
        """Generate ``n`` completions by prefilling the assistant turn with ``output_prefix``.

        Passes ``n`` directly to the underlying model API (e.g. OpenAI's
        ``chat.completions.create`` supports this natively).  Each completion
        diverges from the open parenthesis of the first tool call onward.
        """
        messages = memory_messages + [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": output_prefix}],
            )
        ]
        breakpoint()
        stop_sequences = ["Observation:", self.code_block_tags[1]]
        response = self.model.generate(messages, stop_sequences=stop_sequences, n=n)

        closing_tag = self.code_block_tags[1]

        def _restore(suffix: str) -> str:
            full = output_prefix + suffix
            if not full.rstrip().endswith(closing_tag):
                full += closing_tag
            return full

        # When n>1 OpenAI returns all choices in response.raw; extract them all.
        raw = getattr(response, "raw", None)
        if raw is not None and hasattr(raw, "choices") and len(raw.choices) > 1:
            for i, ch in enumerate(raw.choices):
                print(f"\n  [raw choice {i}] content: {repr((ch.message.content or '')[:200])}")
            return [_restore(choice.message.content or "") for choice in raw.choices]
        print(f"\n  [raw single choice] content: {repr((response.content or '')[:200])}")
        return [_restore(response.content or "")]

    def _generate_trajectories(
        self, memory_messages: list, output_prefix: str, n: int
    ) -> list[tuple[str, str, str]]:
        """For each of ``n`` completions: execute the code, observe, generate next thought+code.

        Returns a list of ``(first_output, observation, next_output)`` triples.
        ``next_output`` is what we compare for similarity across trajectories.

        Each completion is executed from the same executor snapshot so they are
        independent of each other's side-effects.
        """
        completions = self._generate_completions(memory_messages, output_prefix, n)
        stop_sequences = ["Observation:", self.code_block_tags[1]]
        snapshot = {k: _safe_deepcopy(v) for k, v in self.python_executor.state.items()}
        trajectories = []

        try:
            for completion in completions:
                # Restore state so each completion starts from the same context.
                self.python_executor.state.clear()
                self.python_executor.state.update({k: _safe_deepcopy(v) for k, v in snapshot.items()})

                is_final = False
                try:
                    code = fix_final_answer_code(parse_code_blobs(completion, self.code_block_tags))
                    code_output = self.python_executor(code)
                    is_final = code_output.is_final_answer
                    observation = "Execution logs:\n" + code_output.logs
                    observation += "\nLast output from code snippet:\n" + truncate_content(str(code_output.output))
                except Exception as e:
                    observation = f"Error: {e}"

                next_messages = memory_messages + [
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=[{"type": "text", "text": completion}],
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": f"Observation:\n{observation}"}],
                    ),
                ]

                try:
                    next_response = self.model.generate(next_messages, stop_sequences=stop_sequences)
                    next_output = next_response.content or ""
                    if not next_output.rstrip().endswith(self.code_block_tags[1]):
                        next_output += self.code_block_tags[1]
                except Exception:
                    next_output = ""

                trajectories.append((completion, observation, next_output, is_final))
        finally:
            # Always restore executor to the state it was in before this method ran.
            self.python_executor.state.clear()
            self.python_executor.state.update(snapshot)

        return trajectories

    def _trajectory_similarity(self, trajectory: tuple[str, str, str]) -> float:
        """Compute Jaccard similarity between the first and second code blocks in a trajectory.

        ``trajectory`` is a ``(first_output, observation, next_output, is_final)`` 4-tuple.
        Returns a score in ``[0, 1]`` where 1 means identical and 0 means no shared tokens.
        Returns 0.0 if either code block cannot be parsed.
        """
        first_output, _, next_output, *_ = trajectory
        try:
            first_code = parse_code_blobs(first_output, self.code_block_tags)
            next_code = parse_code_blobs(next_output, self.code_block_tags)
        except Exception:
            return 0.0

        tokens_a = set(first_code.split())
        tokens_b = set(next_code.split())
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _pick_and_commit(
        self, trajectories: list[tuple[str, str, str]]
    ) -> tuple[str, str, Any, bool] | None:
        """Pick the trajectory with most work done and commit its first step to executor state.

        Selects the trajectory with the lowest intra-trajectory similarity
        (first and second code blocks diverge most, indicating the first step
        produced genuinely new information).

        Commits by executing the winning first code on the live executor
        (intentionally updating state, unlike ``_generate_trajectories``).

        Returns ``(first_output, observation, action_output, is_final_answer)``
        of the winner, or ``None`` if ``trajectories`` is empty.
        """
        if not trajectories:
            return None

        # Prefer any trajectory where the first step already produced a final answer.
        final_traj = next((t for t in trajectories if t[3]), None)
        winner = final_traj if final_traj is not None else min(trajectories, key=self._trajectory_similarity)
        first_output, observation, *_ = winner

        action_output = None
        is_final_answer = False
        try:
            code = parse_code_blobs(first_output, self.code_block_tags)
            code_output = self.python_executor(code)
            action_output = code_output.output
            is_final_answer = code_output.is_final_answer
        except Exception:
            pass  # observation already captured during _generate_trajectories

        return first_output, observation, action_output, is_final_answer

    def _step_stream(self, memory_step: ActionStep) -> Generator:
        """Probabilistic step: sample diverse completions, pick winner, commit."""
        memory_messages = self.write_memory_to_messages()
        memory_step.model_input_messages = memory_messages

        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            stop_sequences.append(self.code_block_tags[1])

        # Step 1: initial generation to identify the intended tool call.
        try:
            chat_message = self.model.generate(memory_messages, stop_sequences=stop_sequences)
            output_text = chat_message.content or ""
            if not output_text.strip().endswith(self.code_block_tags[1]):
                output_text += self.code_block_tags[1]
                chat_message.content = output_text
            memory_step.model_output_message = chat_message
            memory_step.model_output = output_text
            memory_step.token_usage = chat_message.token_usage
            print("\n=== initial model output ===\n", repr(output_text))
        except Exception as e:
            raise AgentGenerationError(f"Error generating model output:\n{e}", self.logger) from e

        # Step 2: parse code and extract the first tool call position.
        try:
            code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
        except Exception as e:
            raise AgentParsingError(f"Error in code parsing:\n{e}", self.logger)

        prefix_result = self._extract_prefix_tool_call(code_action)
        committed = None

        if prefix_result is not None:
            _, open_paren_pos = prefix_result
            code_start = output_text.find(code_action)
            if code_start != -1:
                output_prefix = output_text[:code_start + open_paren_pos + 1]
                # Steps 3-4: sample trajectories, pick the one with most work done.
                trajectories = self._generate_trajectories(memory_messages, output_prefix, self.n_samples)
                committed = self._pick_and_commit(trajectories)

        if committed is not None:
            first_output, observation, action_output, is_final_answer = committed
            try:
                code_action = fix_final_answer_code(parse_code_blobs(first_output, self.code_block_tags))
                memory_step.model_output = first_output
            except Exception:
                pass  # keep the original code_action
            memory_step.observations = observation
            memory_step.action_output = action_output
        else:
            # Fallback: no tool call found or sampling failed — execute the original code.
            try:
                code_output = self.python_executor(code_action)
                observation = "Execution logs:\n" + code_output.logs
                observation += "\nLast output from code snippet:\n" + truncate_content(str(code_output.output))
                memory_step.observations = observation
                memory_step.action_output = code_output.output
                action_output = code_output.output
                is_final_answer = code_output.is_final_answer
            except Exception as e:
                raise AgentExecutionError(str(e), self.logger)

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        memory_step.code_action = code_action
        memory_step.tool_calls = [tool_call]
        yield tool_call
        yield ActionOutput(output=action_output, is_final_answer=is_final_answer)
