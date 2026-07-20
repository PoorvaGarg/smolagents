"""MarkovReAct CodeAgent.

Runs exactly like CodeAgent but restricts the LLM's context at each step to the
high-level task plus the last `window_size` ActionSteps, dropping everything
older than that. Not wired into smolagents.__init__ (experimental, like
context_pruning) -- import it directly:

    from smolagents.markov_react import MarkovReActCodeAgent
"""

from .agents import CodeAgent
from .memory import ActionStep
from .models import ChatMessage


class MarkovReActCodeAgent(CodeAgent):
    def __init__(self, *args, window_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size

    def write_memory_to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        task_steps = [s for s in self.memory.steps if not isinstance(s, ActionStep)]
        action_steps = [s for s in self.memory.steps if isinstance(s, ActionStep)]
        for step in task_steps:
            messages.extend(step.to_messages(summary_mode=summary_mode))
        windowed_action_steps = action_steps[-self.window_size :] if self.window_size > 0 else []
        for step in windowed_action_steps:
            messages.extend(
                step.to_messages(summary_mode=summary_mode, include_reasoning=self.resend_reasoning)
            )
        return messages
