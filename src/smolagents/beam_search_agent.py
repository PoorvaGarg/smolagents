"""Beam search over naiveReAct: at each step, every live beam samples `branch_factor`
independent (thought, code) continuations, and only the top `beam_width` survive, ranked by
Beam.pruning_score -- the log joint of the expectation judge's P(yes) with how likely the model
was to generate its own thought and code, averaged over the whole trajectory so far.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any

# Floor for p_yes before taking its log, so a judge score of exactly 0.0 (its zero-information
# fallback, see _judge_step) doesn't send pruning_score to -inf.
_MIN_PROB = 1e-4

from .agents import CodeAgent
from .context_pruning.completion_judge import judge_step_achieved_expectation
from .local_python_executor import fix_final_answer_code
from .memory import ActionStep, AgentMemory, SystemPromptStep, TaskStep
from .models import ChatMessage
from .monitoring import Timing, TokenUsage
from .tracelet_agent import PLACEHOLDER_LOGPROB, _FillinLogprob
from .utils import AgentExecutionError, execute_against_snapshot, find_stateful_backends, parse_code_blobs


def _sum_scored_logprobs(tokens: list) -> _FillinLogprob | None:
    """Same placeholder-token filtering as tracelet_agent._choice_logprob, but for an arbitrary
    token sublist rather than a whole completion choice -- so it can be called once per half of
    a split completion instead of only on the whole thing."""
    values = [t.logprob for t in tokens if getattr(t, "logprob", None) is not None]
    scored = [v for v in values if v > PLACEHOLDER_LOGPROB]
    return _FillinLogprob(sum(scored), len(scored), len(values) - len(scored)) if scored else None


def _split_token_logprobs(
    content: list | None, output_text: str, split_at: str
) -> tuple[_FillinLogprob | None, _FillinLogprob | None]:
    """Split a completion's per-token logprobs at the first occurrence of split_at in the decoded
    text (here, the opening code-block tag) -- a token whose text starts before that point is
    "thought", everything from split_at onward (including split_at itself) is "code"."""
    if not content:
        return None, None
    boundary = output_text.find(split_at)
    if boundary == -1:
        boundary = len(output_text)
    thought_tokens, code_tokens = [], []
    position = 0
    for token_info in content:
        token_text = getattr(token_info, "token", "") or ""
        (thought_tokens if position < boundary else code_tokens).append(token_info)
        position += len(token_text)
    return _sum_scored_logprobs(thought_tokens), _sum_scored_logprobs(code_tokens)


@dataclass
class Beam:
    """One live branch of the search: its own step history, its own executor/tool state (never
    shared with another beam), and the running per-step expectation-judge scores that decide
    whether it survives the next prune.
    """

    beam_id: str
    parent_id: str | None
    memory: AgentMemory
    executor_state: dict[str, Any]
    tool_state: dict[int, Any]
    expectation_scores: list[float] = field(default_factory=list)
    # Mean per-token logprob of each sampled step's thought text and code text, kept separate
    # (not combined) since they come from different parts of the completion and the earlier
    # 9-way prior/likelihood framework in notes.ipynb treats them as distinguishable signals.
    # Mean, not the raw sum, since a raw sum would just rank shorter text higher regardless of
    # quality (same length-bias tracelet_agent.py's fillin_mean_logprob was introduced to avoid).
    thought_logprobs: list[float] = field(default_factory=list)
    code_logprobs: list[float] = field(default_factory=list)
    errored_this_step: bool = False
    is_final: bool = False
    final_output: Any = None

    @property
    def score(self) -> float:
        """Mean per-step P(yes) so far -- the aggregate validated in TODO.md (corr 0.528)."""
        if not self.expectation_scores:
            return 0.0
        return sum(self.expectation_scores) / len(self.expectation_scores)

    @property
    def mean_thought_logprob(self) -> float:
        """Mean over this beam's own per-step thought-logprobs -- tracked alongside `score` but
        not yet folded into it; how (or whether) to combine the two is still undecided."""
        if not self.thought_logprobs:
            return 0.0
        return sum(self.thought_logprobs) / len(self.thought_logprobs)

    @property
    def mean_code_logprob(self) -> float:
        """Same as mean_thought_logprob, for the code half of each step's completion."""
        if not self.code_logprobs:
            return 0.0
        return sum(self.code_logprobs) / len(self.code_logprobs)

    @property
    def pruning_score(self) -> float:
        """What _prune actually ranks by: mean over steps of log(p_yes) + thought_logprob +
        code_logprob -- the log joint P(thought) * P(code | thought) * P(judge=yes | thought,
        code) by the chain rule, each term already conditioned on what came before it in the
        same step. Combined per step, then averaged (not summed), so one bad step is averaged
        into the trajectory rather than compounding with the rest -- same reason `score` itself
        averages instead of summing."""
        if not self.expectation_scores:
            return 0.0
        combined = [
            math.log(max(p_yes, _MIN_PROB)) + thought_lp + code_lp
            for p_yes, thought_lp, code_lp in zip(self.expectation_scores, self.thought_logprobs, self.code_logprobs)
        ]
        return sum(combined) / len(combined)

    def extend(
        self,
        beam_id: str,
        step: ActionStep,
        p_yes: float,
        thought_logprob: float,
        code_logprob: float,
        errored: bool,
        executor_state: dict[str, Any],
        tool_state: dict[int, Any],
    ) -> "Beam":
        """A new beam: this one's history plus one more committed step. Never mutates self, so a
        pruned beam's state stays exactly as it was at the moment it was discarded."""
        child_memory = AgentMemory(self.memory.system_prompt.system_prompt)
        child_memory.steps = [*self.memory.steps, step]
        return Beam(
            beam_id=beam_id,
            parent_id=self.beam_id,
            memory=child_memory,
            executor_state=executor_state,
            tool_state=tool_state,
            expectation_scores=[*self.expectation_scores, p_yes],
            thought_logprobs=[*self.thought_logprobs, thought_logprob],
            code_logprobs=[*self.code_logprobs, code_logprob],
            errored_this_step=errored,
            is_final=step.is_final_answer,
            final_output=step.action_output if step.is_final_answer else None,
        )


class BeamSearchCodeAgent(CodeAgent):
    """Subclasses CodeAgent purely to reuse its __init__ (tool setup, executor construction,
    prompt loading) -- run() will be a fully custom beam loop, not an override of _step_stream,
    since the branching unit here is a whole trajectory, not one step."""

    def __init__(
        self,
        *args,
        beam_width: int = 2,
        branch_factor: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beam_width = beam_width
        self.branch_factor = branch_factor

    def _beam_messages(self, memory: AgentMemory) -> list[ChatMessage]:
        """Same as CodeAgent.write_memory_to_messages, but against a beam's own memory instead
        of self.memory -- beams diverge, so each needs its own history turned into messages."""
        messages = memory.system_prompt.to_messages(summary_mode=False)
        for step in memory.steps:
            messages.extend(step.to_messages(summary_mode=False, include_reasoning=self.resend_reasoning))
        return messages

    def _sample_step(self, memory: AgentMemory) -> tuple[ActionStep, str, float, float]:
        """One independent (thought, code) sample against a beam's history -- a full resample,
        not a fill-in on a fixed skeleton, since that's where real diversity lives (see TODO.md).
        Returns the step generated and parsed, but not yet executed (timing.end_time and the
        observation/action_output fields are set by whichever piece executes it), the thought
        text extracted for the judge, and this completion's mean per-token logprob for the
        thought half and the code half separately."""
        messages = self._beam_messages(memory)
        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            stop_sequences.append(self.code_block_tags[1])

        chat_message = self.model.generate(messages, stop_sequences=stop_sequences, logprobs=True)
        output_text = chat_message.content or ""
        if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
            output_text += self.code_block_tags[1]

        try:
            code_action = fix_final_answer_code(parse_code_blobs(output_text, self.code_block_tags))
        except Exception:
            # No parseable code block -- not an execution error, so not errored_this_step. Left
            # unexecuted; the judge scores the bare thought/no-code/no-observation on its own
            # merits and pruning decides its fate from that score alone (never mind that the
            # candidate never touched the executor).
            code_action = None
        thought = output_text.split(self.code_block_tags[0], 1)[0].strip()

        choices = getattr(chat_message.raw, "choices", None)
        content = getattr(getattr(choices[0], "logprobs", None), "content", None) if choices else None
        thought_lp, code_lp = _split_token_logprobs(content, output_text, self.code_block_tags[0])
        thought_logprob = thought_lp.mean if thought_lp else 0.0
        code_logprob = code_lp.mean if code_lp else 0.0

        action_step_number = len([s for s in memory.steps if isinstance(s, ActionStep)]) + 1
        step = ActionStep(step_number=action_step_number, timing=Timing(start_time=time.time()))
        step.model_input_messages = messages
        step.model_output_message = chat_message
        step.model_output = output_text
        step.code_action = code_action
        step.token_usage = chat_message.token_usage
        return step, thought, thought_logprob, code_logprob

    def _stateful_backends(self) -> dict[int, Any]:
        """Tools, or objects they hold, that opt into snapshotting via supports_state_snapshot
        plus a get_state/set_state pair. See find_stateful_backends in utils.py -- shared with
        TraceletCodeAgent, which needs the exact same lookup for its own branching."""
        return find_stateful_backends(self.tools)

    def _execute_step(
        self,
        step: ActionStep,
        executor_state: dict[str, Any],
        tool_state: dict[int, Any],
    ) -> tuple[dict[str, Any], dict[int, Any], bool]:
        """Run a freshly sampled step's code against one beam's own state (never the shared
        self.python_executor live state, since beams diverge). Writes the outcome onto step in
        place and returns the resulting state for the child Beam, plus whether it errored."""
        observation, output, is_final, error, post_executor_state, post_tool_state = execute_against_snapshot(
            self.python_executor, self.tools, step.code_action, executor_state, tool_state
        )
        step.observations = observation
        step.action_output = output
        step.is_final_answer = is_final
        step.timing.end_time = time.time()
        if error is not None:
            step.error = AgentExecutionError(error, self.logger)
        return post_executor_state, post_tool_state, error is not None

    def _judge_step(self, thought: str, step: ActionStep) -> tuple[float, TokenUsage]:
        """P(this step's observation achieved what the thought expected) -- the per-step judge
        validated for TraceletCodeAgent's "expectation" strategy (TODO.md, corr 0.528). Scored
        once per branch, not comparatively across candidates like tracelet's judge, since every
        sampled branch here becomes its own beam rather than one thought's set of fill-ins."""
        p_yes, _, usage = judge_step_achieved_expectation(
            self.model.generate, thought, step.code_action or "", step.observations or ""
        )
        return p_yes if p_yes is not None else 0.0, usage or TokenUsage(input_tokens=0, output_tokens=0)

    def _build_root_beam(self, task: str, images: list | None = None) -> Beam:
        """The single starting point every beam branches from: system prompt + task, no steps
        yet, and empty executor/tool state (a fresh trajectory has nothing to snapshot)."""
        memory = AgentMemory(self.system_prompt)
        memory.steps.append(TaskStep(task=task, task_images=images))
        return Beam(beam_id="0", parent_id=None, memory=memory, executor_state={}, tool_state={})

    def _expand_beam(self, beam: Beam) -> list[Beam]:
        """branch_factor independent (thought, code) samples off one live beam, each executed
        against that beam's own state and judged on its own -- never called on a beam that is
        already is_final, since a finished beam stops branching and just stays in the pool."""
        children = []
        for i in range(self.branch_factor):
            step, thought, thought_logprob, code_logprob = self._sample_step(beam.memory)
            if step.code_action is not None:
                post_executor_state, post_tool_state, errored = self._execute_step(
                    step, beam.executor_state, beam.tool_state
                )
            else:
                step.timing.end_time = time.time()
                post_executor_state, post_tool_state, errored = beam.executor_state, beam.tool_state, False
            p_yes, judge_usage = self._judge_step(thought, step)
            step.token_usage = TokenUsage(
                input_tokens=step.token_usage.input_tokens + judge_usage.input_tokens,
                output_tokens=step.token_usage.output_tokens + judge_usage.output_tokens,
            )
            children.append(
                beam.extend(
                    beam_id=f"{beam.beam_id}.{i}",
                    step=step,
                    p_yes=p_yes,
                    thought_logprob=thought_logprob,
                    code_logprob=code_logprob,
                    errored=errored,
                    executor_state=post_executor_state,
                    tool_state=post_tool_state,
                )
            )
        return children

    def _prune(self, beams: list[Beam]) -> list[Beam]:
        """Keep the top beam_width beams, ranked lexicographically: any beam that didn't error
        on its last step outranks any that did, regardless of score; ties within a tier break by
        pruning_score, the log joint of judge P(yes) with thought/code generation likelihood
        (TODO.md -- errors can still recover, so they're deprioritized, not dropped)."""
        ranked = sorted(beams, key=lambda b: (not b.errored_this_step, b.pruning_score), reverse=True)
        return ranked[: self.beam_width]

    def run(
        self,
        task: str,
        reset: bool = True,
        images: list | None = None,
        max_steps: int | None = None,
    ) -> Any:
        """Beam search over whole trajectories: every live beam branches into branch_factor
        children each round, all candidates are pruned down to beam_width survivors, and this
        repeats until every beam has finished or max_steps is hit. A finished beam freezes (no
        longer branches) but still competes on score for a pool slot each round (decisions 1-2,
        TODO.md). The winning beam's steps are copied into self.memory so callers that read
        agent.memory.steps afterward (e.g. evaluate_agent) see an ordinary-looking trajectory."""
        max_steps = max_steps or self.max_steps
        self.task = task
        if reset:
            self.memory.reset()
            self.monitor.reset()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        beams = [self._build_root_beam(task, images)]
        step_count = 0
        while step_count < max_steps and any(not beam.is_final for beam in beams):
            step_count += 1
            candidates = []
            for beam in beams:
                candidates.extend([beam] if beam.is_final else self._expand_beam(beam))
            beams = self._prune(candidates)

        finished = [beam for beam in beams if beam.is_final]
        winner = max(finished if finished else beams, key=lambda beam: beam.pruning_score)

        self.memory.steps = list(winner.memory.steps)
        # run() never goes through _finalize_step (there's no single per-step loop to hook), so
        # fire the same step callbacks here -- this is how self.monitor normally accumulates
        # token counts, and evaluate_agent asserts monitor totals match memory.steps totals.
        for step in self.memory.steps:
            self.step_callbacks.callback(step, agent=self)
        return winner.final_output if winner.is_final else winner.memory.steps[-1].action_output
