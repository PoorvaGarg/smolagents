"""Side context buffer for context-pruning agents.

Tracks steps grouped by subtask. When a subtask boundary is crossed, only the
last step of the completed subtask is kept in committed memory — all intermediate
failed attempts are dropped. The active subtask's full context is always visible
so the model doesn't repeat queries it already tried within the current subtask.
"""

from dataclasses import dataclass, field

from ..memory import ActionStep


@dataclass
class SideContextBuffer:
    """Buffer that collapses completed subtasks to their last step."""

    committed_steps: list[ActionStep] = field(default_factory=list)
    current_context: list[ActionStep] = field(default_factory=list)

    def add_step(self, step: ActionStep) -> None:
        """Append *step* to the active subtask context."""
        self.current_context.append(step)

    def commit_subtask(self) -> None:
        """Close the active subtask: promote its last step to committed memory."""
        if self.current_context:
            self.committed_steps.append(self.current_context[-1])
            self.current_context = []

    def get_pruned_steps(self) -> list[ActionStep]:
        """Return the pruned view of history seen by the model.

        = one representative step per completed subtask
        + all steps of the active subtask (so the model avoids repeating them)
        """
        return self.committed_steps + self.current_context

    def reset(self) -> None:
        self.committed_steps = []
        self.current_context = []
