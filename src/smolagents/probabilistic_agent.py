import copy
from typing import Any, Generator

from .monitoring import TokenUsage


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
from .utils import AgentGenerationError, AgentParsingError, parse_code_blobs, truncate_content


class ProbabilisticCodeAgent(CodeAgent):
    """CodeAgent that samples ``n_samples`` diverse thought+code pairs each step.

    For each step:
      1. Generate ``n_samples`` independent thought+code completions.
      2. Execute each, observe, and generate the next thought+code.
      3. Pick the trajectory whose first and second code blocks diverge most.
    """

    def __init__(self, *args, n_samples: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

    def _generate_completions(self, memory_messages: list, n: int) -> list[str]:
        """Generate ``n`` diverse thought+code outputs using sequential conditioning.

        Each call sees the previous completions and is asked to try a different approach.
        """
        stop_sequences = ["Observation:", self.code_block_tags[1]]
        closing_tag = self.code_block_tags[1]
        completions = []
        total_tokens = 0
        for _ in range(n):
            messages = memory_messages
            if completions:
                prior_codes = []
                for c in completions:
                    try:
                        prior_codes.append(parse_code_blobs(c, self.code_block_tags))
                    except Exception:
                        prior_codes.append(c)
                prior = "\n\n".join(f"```python\n{code}\n```" for code in prior_codes)
                messages = memory_messages + [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": f"The following code approaches have already been tried:\n\n{prior}\n\nGenerate a different approach. Respond with a thought followed by a {self.code_block_tags[0]}...{self.code_block_tags[1]} code block."}],
                    )
                ]
            response = self.model.generate(messages, stop_sequences=stop_sequences)
            total_tokens += response.token_usage.total_tokens
            output = response.content or ""
            if not output.rstrip().endswith(closing_tag):
                output += closing_tag
            completions.append(output)
        # TokenUsage.total_tokens is init=False (computed as input+output), so we store the
        # running sum in input_tokens and leave output_tokens=0.
        return completions, TokenUsage(input_tokens=total_tokens, output_tokens=0)

    def _generate_trajectories(
        self, memory_messages: list, n: int
    ) -> tuple[list[tuple[str, str, str, bool]], TokenUsage]:
        """Generate ``n`` trajectories, each a pair of thought+code generations.

        For each of ``n`` independent completions (thought+code):
          1. Execute the code and collect the observation.
          2. Generate a second thought+code given the observation.
        Returns ``(trajectories, total_token_usage)`` where trajectories is a list of
        ``(first_output, observation, next_output, is_final)`` 4-tuples.
        """
        completions, token_usage = self._generate_completions(memory_messages, n)
        stop_sequences = ["Observation:", self.code_block_tags[1]]
        snapshot = {k: _safe_deepcopy(v) for k, v in self.python_executor.state.items()}
        trajectories = []
        total_tokens = token_usage.total_tokens

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
                    total_tokens += next_response.token_usage.total_tokens
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

        # TokenUsage.total_tokens is init=False (computed as input+output), so we store the
        # running sum in input_tokens and leave output_tokens=0.
        return trajectories, TokenUsage(input_tokens=total_tokens, output_tokens=0)

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
        """Probabilistic step: sample n diverse thought+code pairs, pick winner, commit."""
        memory_messages = self.write_memory_to_messages()
        memory_step.model_input_messages = memory_messages

        try:
            trajectories, token_usage = self._generate_trajectories(memory_messages, self.n_samples)
        except Exception as e:
            raise AgentGenerationError(f"Error generating trajectories:\n{e}", self.logger) from e
        memory_step.token_usage = token_usage

        committed = self._pick_and_commit(trajectories)
        if committed is None:
            raise AgentGenerationError("No trajectories were generated.", self.logger)

        first_output, observation, action_output, is_final_answer = committed

        try:
            code_action = fix_final_answer_code(parse_code_blobs(first_output, self.code_block_tags))
        except Exception as e:
            raise AgentParsingError(f"Error in code parsing:\n{e}", self.logger)

        memory_step.model_output = first_output
        memory_step.observations = observation
        memory_step.action_output = action_output

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        memory_step.code_action = code_action
        memory_step.tool_calls = [tool_call]
        yield tool_call
        yield ActionOutput(output=action_output, is_final_answer=is_final_answer)
