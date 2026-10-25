# PRM800K answer grader (vendored)

Scorer for the MATH-500 runs (`baseline/math500_setup.py`).

- **Source:** https://github.com/openai/prm800k/tree/main/prm800k/grading
- **Upstream commit:** `7ecc794703b2877f63226f2477a49b34f9b25163` (2023-06-01; the repo is archived)
- **Vendored:** 2026-10-25
- **License:** MIT, Copyright (c) 2023 OpenAI (see `LICENSE`)
- **Paper:** Lightman et al., *Let's Verify Step by Step*, 2023 — https://arxiv.org/abs/2305.20050

## Why this grader

MATH-500 (`HuggingFaceH4/MATH-500`) is the 500-problem test subset OpenAI built for the paper above, so this is the grader that matches the dataset's own origin.

`grade_answer(given, gold)` normalizes both answers and tries an exact match. It then handles tuples and intervals. If that's still inconclusive, it calls an answer correct when `sympy.simplify(gold - given) == 0`. That catches equivalent forms such as `14/3` vs `\frac{14}{3}`, which the original Hendrycks et al. MATH grader treats as a string-normalized exact match and can miss.

## Modifications

There is exactly one change: an import in `grader.py`, made so the code works as a package here.

```diff
- from grading import math_normalize
+ from . import math_normalize
```

`grader.py` and `math_normalize.py` are otherwise byte-for-byte upstream. The grader needs `sympy` and `pylatexenc`.

## Tolerance extension (in `math500_setup.py`, not in this directory)

These files are left untouched. `Math500Scorer` in `baseline/math500_setup.py` runs `grade_answer` first. Only if that fails does it accept answers whose every tuple element evaluates numerically within `REL_TOL = 1e-6` of the gold. This is for CodeAgent returning full-precision Python floats, e.g. `1.5707963267948966` for `\frac{\pi}{2}`.

- It reuses this grader's `_normalize`, `split_tuple`, bracket matching, and `should_allow_eval` guard. It also keeps the grader's refusal of unreduced fractions.
- The 1e-6 threshold accepts computed floats but rejects hand-rounded decimals (`4.6667` vs `14/3` is off by 7e-6).
- Every verdict is labeled `CORRECT_STRICT` / `CORRECT_TOLERANCE` / `INCORRECT` and saved as `grade_label`. **To report literature-comparable accuracy, count only `CORRECT_STRICT`.**

## Sanity checks (2026-10-25)

- All 500 MATH-500 golds grade correct against themselves.
- Equivalent forms match: `14/3` = `\frac{14}{3}`, `\dfrac` = `\frac`, `2\sqrt2` = `2\sqrt{2}`, `0.5` = `\frac{1}{2}`, `(3,-1)` = `(3, -1)`.
- Decimal approximations are rejected: `4.6667` is not `\frac{14}{3}`. `MATH500_ANSWER_INSTRUCTION` asks the agent for exact values for this reason.
