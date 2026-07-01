"""Pruned memory writer mixin for context-pruning agents.

Overrides ``write_memory_to_messages`` so that the LLM prompt contains only the
pruned view of history (one representative step per completed subtask, plus the
full active-subtask context) rather than every step ever taken.
"""

from ..memory import ActionStep
from ..models import ChatMessage
from .context_buffer import SideContextBuffer


class PrunedMemoryMixin:
    """Mixin that replaces ``write_memory_to_messages`` with a pruned version.

    Requires the host class to have:
    - ``self.memory``        — standard AgentMemory
    - ``self._ctx_buffer``   — a SideContextBuffer instance
    """

    def write_memory_to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        # Include TaskStep and PlanningStep verbatim — only ActionSteps are pruned.
        for step in self.memory.steps:
            if not isinstance(step, ActionStep):
                messages.extend(step.to_messages(summary_mode=summary_mode))
        for step in self._ctx_buffer.get_pruned_steps():
            messages.extend(step.to_messages(summary_mode=summary_mode))
        return messages
