"""MATH-500 baseline setup: dataset loading and scoring.

Dataset: HuggingFaceH4/MATH-500, the 500-problem MATH test subset created by OpenAI for
"Let's Verify Step by Step" (Lightman et al., 2023, https://arxiv.org/abs/2305.20050).

Scorer: OpenAI's PRM800K grader (`grade_answer`), vendored unmodified apart from one import line
in third_party/prm800k_grading/ (MIT, see LICENSE there). It is the grader that matches the
dataset's own provenance: normalize -> exact match -> tuple/interval handling -> sympy equivalence
(correct iff simplify(gold - pred) == 0). Source:
https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py

Tolerance extension (ours, not PRM800K): an answer that fails strict grading is also accepted if
every tuple element evaluates numerically to within REL_TOL of the gold. This catches CodeAgent
returning full-precision floats from Python (1.5707963267948966 for pi/2). It reuses the grader's
own normalization, tuple and bracket rules, and eval guard, and it keeps the grader's refusal of
unreduced fractions. The scorer records which path matched in last_label ("CORRECT_STRICT",
"CORRECT_TOLERANCE", "INCORRECT"), so strict PRM800K accuracy can always be recovered.

Returns a Dataset shaped like load_gaia_dataset's output, so evaluate_agent works unmodified.
"""

import datasets
import sympy

from third_party.prm800k_grading import grader
from third_party.prm800k_grading.grader import grade_answer


# 1e-6 accepts computed floats but rejects hand-rounded decimals (4.6667 vs 14/3 is off by 7e-6).
REL_TOL = 1e-6
_ABS_TOL = 1e-9


# Decimals defeat the sympy check (4.6667 is not 14/3), so ask for exact values.
MATH500_ANSWER_INSTRUCTION = (
    "\n\nPass only the final answer to final_answer, with no explanation, as an exact value "
    "(e.g. \"\\frac{14}{3}\", \"2\\sqrt{2}\", \"(3, -1)\") rather than a decimal approximation."
)


def load_math500_dataset(split: str = "test") -> datasets.Dataset:
    raw = datasets.load_dataset("HuggingFaceH4/MATH-500", split=split)
    examples = [
        {
            "task_id": row["unique_id"],
            "question": row["problem"],
            "true_answer": row["answer"],
            "task": f"math500-L{row['level']}",
            "file_name": None,
            "subject": row["subject"],
        }
        for row in raw
    ]
    return datasets.Dataset.from_list(examples)


def _numeric_value(elem: str) -> complex | None:
    if not grader.should_allow_eval(elem):
        return None
    try:
        return complex(sympy.N(grader._sympy_parse(elem)))
    except Exception:
        return None


def _close_within_tolerance(prediction: str, true_answer: str, rel_tol: float) -> bool:
    gold = grader._normalize(true_answer)
    pred = grader._normalize(prediction)
    if not gold or not pred:
        return False
    gold_elems, pred_elems = grader.split_tuple(gold), grader.split_tuple(pred)
    if len(gold_elems) != len(pred_elems):
        return False
    # Same bracket rule as grade_answer: (a, b) and [a, b] are different intervals.
    if len(gold_elems) > 1 and (gold[0] != pred[0] or gold[-1] != pred[-1]):
        return False
    for g_elem, p_elem in zip(gold_elems, pred_elems):
        if grader._is_frac(g_elem) and grader._is_frac(p_elem):
            if g_elem != p_elem:
                return False
            continue
        g_val, p_val = _numeric_value(g_elem), _numeric_value(p_elem)
        if g_val is None or p_val is None:
            return False
        if abs(g_val - p_val) > max(rel_tol * abs(g_val), _ABS_TOL):
            return False
    return True


class Math500Scorer:
    def __init__(self, rel_tol: float = REL_TOL):
        self.rel_tol = rel_tol
        self.last_label = None

    def __call__(self, prediction: str, true_answer: str) -> bool:
        if not prediction:
            self.last_label = "INCORRECT"
        elif grade_answer(str(prediction), true_answer):
            self.last_label = "CORRECT_STRICT"
        elif _close_within_tolerance(str(prediction), true_answer, self.rel_tol):
            self.last_label = "CORRECT_TOLERANCE"
        else:
            self.last_label = "INCORRECT"
        return self.last_label != "INCORRECT"


math500_scorer = Math500Scorer()
