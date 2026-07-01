"""Context-pruning CodeAgent.

Runs exactly like CodeAgent but maintains a SideContextBuffer that collapses
completed subtasks to their last step. The pruned view is used as the LLM
prompt, preventing context bloat from repeated failed web searches.
"""

from ..agents import CodeAgent
from ..memory import ActionStep
from .context_buffer import SideContextBuffer
from .pruned_memory import PrunedMemoryMixin
from .subtask_detector import SubtaskDetector


class ContextPruningCodeAgent(PrunedMemoryMixin, CodeAgent):
    """CodeAgent that prunes redundant subtask steps from the LLM context.

    At each step boundary:
    - If the new step continues the same subtask (same tool group), it is added
      to the active context window alongside its predecessors.
    - If the new step starts a new subtask, only the *last* step of the
      completed subtask is kept in committed memory; all prior attempts are
      dropped from the prompt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ctx_buffer = SideContextBuffer()
        self._subtask_detector = SubtaskDetector(set(self.tools.keys()))
        self._prev_code: str | None = None

    def run(self, *args, **kwargs):
        self._ctx_buffer.reset()
        self._prev_code = None
        return super().run(*args, **kwargs)

    def _finalize_step(self, memory_step):
        super()._finalize_step(memory_step)
        if not isinstance(memory_step, ActionStep):
            return
        code = memory_step.code_action or ""
        if self._prev_code is None:
            # First step: start the active context with no prior to compare against.
            self._ctx_buffer.add_step(memory_step)
        elif self._subtask_detector.same_subtask(self._prev_code, code):
            self._ctx_buffer.add_step(memory_step)
        else:
            self._ctx_buffer.commit_subtask()
            self._ctx_buffer.add_step(memory_step)
        self._prev_code = code
