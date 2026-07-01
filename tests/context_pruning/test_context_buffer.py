import unittest
from unittest.mock import MagicMock

from smolagents.context_pruning.context_buffer import SideContextBuffer


def make_step(name: str):
    """Create a minimal ActionStep mock tagged with a name for easy assertion."""
    step = MagicMock()
    step.name = name
    return step


class TestSideContextBufferInitial(unittest.TestCase):
    def setUp(self):
        self.buf = SideContextBuffer()

    def test_starts_empty(self):
        self.assertEqual(self.buf.get_pruned_steps(), [])

    def test_committed_and_context_start_empty(self):
        self.assertEqual(self.buf.committed_steps, [])
        self.assertEqual(self.buf.current_context, [])


class TestSideContextBufferAddStep(unittest.TestCase):
    def setUp(self):
        self.buf = SideContextBuffer()

    def test_add_single_step(self):
        s = make_step("s1")
        self.buf.add_step(s)
        self.assertEqual(self.buf.get_pruned_steps(), [s])

    def test_add_multiple_steps_same_subtask(self):
        s1, s2, s3 = make_step("s1"), make_step("s2"), make_step("s3")
        for s in (s1, s2, s3):
            self.buf.add_step(s)
        self.assertEqual(self.buf.get_pruned_steps(), [s1, s2, s3])


class TestSideContextBufferCommitSubtask(unittest.TestCase):
    def setUp(self):
        self.buf = SideContextBuffer()

    def test_commit_keeps_only_last_step(self):
        s1, s2, s3 = make_step("s1"), make_step("s2"), make_step("s3")
        for s in (s1, s2, s3):
            self.buf.add_step(s)
        self.buf.commit_subtask()
        # Only the last step (s3) should survive
        self.assertEqual(self.buf.committed_steps, [s3])
        self.assertEqual(self.buf.current_context, [])

    def test_commit_empty_context_is_noop(self):
        self.buf.commit_subtask()
        self.assertEqual(self.buf.committed_steps, [])

    def test_pruned_view_after_commit_and_new_step(self):
        s1, s2 = make_step("s1"), make_step("s2")
        s3 = make_step("s3")
        self.buf.add_step(s1)
        self.buf.add_step(s2)
        self.buf.commit_subtask()   # s2 committed, s1 dropped
        self.buf.add_step(s3)       # new subtask starts
        self.assertEqual(self.buf.get_pruned_steps(), [s2, s3])

    def test_two_subtask_transitions(self):
        s1, s2 = make_step("s1"), make_step("s2")  # subtask A
        s3, s4 = make_step("s3"), make_step("s4")  # subtask B
        s5 = make_step("s5")                        # subtask C (active)

        for s in (s1, s2):
            self.buf.add_step(s)
        self.buf.commit_subtask()   # keeps s2

        for s in (s3, s4):
            self.buf.add_step(s)
        self.buf.commit_subtask()   # keeps s4

        self.buf.add_step(s5)

        # pruned view: s2, s4 (one each from completed subtasks) + s5 (active)
        self.assertEqual(self.buf.get_pruned_steps(), [s2, s4, s5])


class TestSideContextBufferReset(unittest.TestCase):
    def test_reset_clears_everything(self):
        buf = SideContextBuffer()
        buf.add_step(make_step("s1"))
        buf.add_step(make_step("s2"))
        buf.commit_subtask()
        buf.add_step(make_step("s3"))
        buf.reset()
        self.assertEqual(buf.get_pruned_steps(), [])
        self.assertEqual(buf.committed_steps, [])
        self.assertEqual(buf.current_context, [])


if __name__ == "__main__":
    unittest.main()
