"""Context-pruning CodeAgent with goal-satisfaction subtask boundaries.

Implements the pseudocode in notes.ipynb: the model states an expectation for each step, a judge
checks whether the observation satisfied it, and a satisfied step closes the subtask -- committing
that step and discarding the failed attempts that preceded it. Unsatisfied steps accumulate, so
while a subtask is still open the agent can see what it already tried.

Differs from ContextPruningCodeAgent, which infers boundaries from tool-group changes and always
keeps the *last* step of a closed subtask rather than the one that actually succeeded.
"""

import importlib.resources
import re

import yaml

from ..agents import CodeAgent
from ..memory import ActionStep
from ..monitoring import LogLevel
from .completion_judge import judge_completion
from .context_buffer import SideContextBuffer
from .pruned_memory import PrunedMemoryMixin


# Tolerates the wrappers models add: bullets, bold, backticks, and 'Expected'/'Expectation'.
_EXPECTATION_RE = re.compile(
    r"^\s*[-*+]?\s*[`*_]*\s*Expect(?:ation|ed(?:\s+observation)?)\s*[`*_]*\s*:\s*[`*_]*\s*(\S.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def parse_expectation(model_output: str | None) -> str | None:
    """Last Expectation line outside any code fence, or None if the model emitted none.

    The last one wins: models sometimes echo the format from the system prompt before
    committing to a real value.
    """
    if not model_output:
        return None
    # Blank fenced regions while preserving newlines, so line ordering is unaffected.
    outside = _FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), model_output)
    matches = _EXPECTATION_RE.findall(outside)
    if not matches:
        return None
    return matches[-1].strip(" `*_").strip() or None


def default_prompt_templates() -> dict:
    """The Expectation-aware system prompt shipped with this package."""
    text = importlib.resources.files("smolagents.prompts").joinpath("context_pruning_agent.yaml").read_text()
    return yaml.safe_load(text)


class GoalPruningCodeAgent(PrunedMemoryMixin, CodeAgent):
    """CodeAgent that prunes context by goal satisfaction rather than by tool group.

    Args:
        completion_threshold: P(expectation satisfied) at or above which a subtask closes.
        max_open_steps: force a subtask closed after this many unsatisfied steps, keeping the
            step that came closest. Without this a permanently-stuck trajectory would retain
            its whole history and be worse off than with no pruning at all.
        judge_completion_calls: set False to disable the judge and never prune (a control arm).
    """

    def __init__(
        self,
        *args,
        completion_threshold: float = 0.5,
        max_open_steps: int = 4,
        judge_completion_calls: bool = True,
        prompt_templates: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, prompt_templates=prompt_templates or default_prompt_templates(), **kwargs)
        self.completion_threshold = completion_threshold
        self.max_open_steps = max_open_steps
        self.judge_completion_calls = judge_completion_calls
        self._ctx_buffer = SideContextBuffer()

    def run(self, *args, **kwargs):
        self._ctx_buffer.reset()
        return super().run(*args, **kwargs)

    def _judge_step(self, memory_step: ActionStep) -> float | None:
        """P(this step satisfied its expectation). Errors are unsatisfied without spending a call."""
        if memory_step.error is not None:
            return 0.0
        if not self.judge_completion_calls or not memory_step.expectation:
            return None
        probability, _, usage = judge_completion(
            self._generate_for_judge,
            thought=str(memory_step.model_output or ""),
            expectation=memory_step.expectation,
            code=memory_step.code_action or "",
            observation=memory_step.observations or "",
        )
        if usage and memory_step.token_usage:
            memory_step.token_usage.input_tokens += usage.input_tokens
            memory_step.token_usage.output_tokens += usage.output_tokens
            memory_step.token_usage.total_tokens += usage.total_tokens
        return probability

    def _generate_for_judge(self, messages, **kwargs):
        """Single completion, using streaming when the provider requires it."""
        if self.stream_outputs:
            from ..models import agglomerate_stream_deltas

            return agglomerate_stream_deltas(list(self.model.generate_stream(messages, **kwargs)))
        return self.model.generate(messages, **kwargs)

    def _finalize_step(self, memory_step):
        super()._finalize_step(memory_step)
        if not isinstance(memory_step, ActionStep):
            return

        memory_step.expectation = parse_expectation(str(memory_step.model_output or ""))
        # The final step ends the run, so judging it would buy nothing.
        if memory_step.is_final_answer:
            self._ctx_buffer.add_step(memory_step)
            return

        memory_step.completion_prob = self._judge_step(memory_step)
        self._ctx_buffer.add_step(memory_step)
        probability = memory_step.completion_prob

        if probability is not None and probability >= self.completion_threshold:
            self._ctx_buffer.commit_subtask()
            self.logger.log(
                f"[GoalPrune] expectation satisfied (p={probability:.3f}) -- subtask closed, "
                f"{len(self._ctx_buffer.committed_steps)} committed step(s).",
                level=LogLevel.INFO,
            )
        elif len(self._ctx_buffer.current_context) >= self.max_open_steps:
            # Ties are the common case here (every attempt scored 0), so fall back to the most
            # recent step rather than whichever happened to come first.
            best = max(
                self._ctx_buffer.current_context,
                key=lambda s: (s.completion_prob or 0.0, s.step_number),
            )
            self._ctx_buffer.commit_subtask(keep=best)
            self.logger.log(
                f"[GoalPrune] {self.max_open_steps} steps without satisfying the expectation -- "
                f"force-closing, keeping step {best.step_number} (p={best.completion_prob or 0.0:.3f}).",
                level=LogLevel.INFO,
            )
        else:
            self.logger.log(
                f"[GoalPrune] expectation not satisfied "
                f"(p={'n/a' if probability is None else f'{probability:.3f}'}) -- "
                f"{len(self._ctx_buffer.current_context)}/{self.max_open_steps} steps open.",
                level=LogLevel.INFO,
            )
