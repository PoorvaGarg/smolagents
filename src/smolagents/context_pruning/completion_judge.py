"""Completion judge: did a step deliver the result it said it would?

One small LLM call per step, seeing only that step -- never the running memory -- so the cost
does not grow with trajectory length. Returns a *probability* rather than a boolean, so the
commit threshold can be tuned after a run instead of by re-prompting: a lenient judge commits
everything and prunes nothing, a strict one never commits and the buffer grows without bound,
and which of those you get is the single most important property of this component.
"""

import math
from typing import Any

from ..models import ChatMessage, MessageRole
from ..monitoring import TokenUsage
from ..utils import truncate_content


# Observations reach 55k characters; truncate_content keeps head and tail, so both ends survive.
OBSERVATION_LIMIT = 3000

_PROMPT = """A single step of an agent's trajectory is shown below. Before running its code, the agent stated what a successful result of the step would contain.

Stated expectation:
{expectation}

The agent's reasoning:
{thought}

Code it ran:
```python
{code}
```

Observation the code produced:
{observation}

Did the observation actually deliver what the expectation asked for?

Answer "no" if the code raised an error, if the observation is empty or only echoes the query, if the content is truncated before the expected part, or if it delivers something related but not the specific thing that was expected. Answer "yes" only if the expected content is present in the observation.

Reply with exactly one word: yes or no."""


def _answer_probability(response: ChatMessage) -> float | None:
    """P(yes) from the first answer token's alternatives, or None if no logprobs came back."""
    choices = getattr(response.raw, "choices", None)
    content = getattr(getattr(choices[0], "logprobs", None), "content", None) if choices else None
    if not content:
        return None
    for token_info in content[:4]:
        alternatives = getattr(token_info, "top_logprobs", None) or [token_info]
        yes = no = 0.0
        for alt in alternatives:
            text = (getattr(alt, "token", "") or "").strip().lower()
            if text.startswith("yes"):
                yes += math.exp(alt.logprob)
            elif text.startswith("no"):
                no += math.exp(alt.logprob)
        if yes + no > 0:
            return yes / (yes + no)
    return None


def _answer_from_text(text: str | None) -> float | None:
    """Fall back to the written word when the provider reports no logprobs."""
    lowered = (text or "").strip().lower()
    if lowered.startswith("yes"):
        return 1.0
    if lowered.startswith("no"):
        return 0.0
    return None


def judge_completion(
    generate: Any,
    thought: str,
    expectation: str,
    code: str,
    observation: str,
    system_prompt: str | None = None,
    request_logprobs: bool = True,
) -> tuple[float | None, str, TokenUsage | None]:
    """Probability that this step's observation satisfied its stated expectation.

    `generate` is a callable taking (messages, **kwargs) and returning a ChatMessage -- pass the
    agent's own `_generate` so streaming providers work unchanged. Returns
    (probability, raw answer text, token usage); probability is None when the answer was
    unparseable, which callers should treat as "unknown" rather than as either verdict.
    """
    instruction = _PROMPT.format(
        expectation=expectation.strip() or "(none stated)",
        thought=truncate_content((thought or "").strip() or "(none)", 1000),
        code=(code or "").strip() or "(none)",
        observation=truncate_content((observation or "").strip() or "(empty)", OBSERVATION_LIMIT),
    )
    messages = []
    if system_prompt:
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": system_prompt}]))
    messages.append(ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": instruction}]))

    # top_logprobs caps at 5 on OpenAI; the answer is one token so no output cap is needed.
    kwargs = {"logprobs": True, "top_logprobs": 5} if request_logprobs else {}
    response = generate(messages, **kwargs)
    text = response.content or ""
    probability = _answer_probability(response)
    if probability is None:
        probability = _answer_from_text(text)
    return probability, text.strip(), response.token_usage
