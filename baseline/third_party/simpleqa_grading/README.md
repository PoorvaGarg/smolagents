# SimpleQA grader (vendored)

Scorer for the SimpleQA runs (`baseline/simpleqa_setup.py`).

- **Source:** https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
- **Upstream commit:** `652c89d0ca9df547706735883097e9537d40dc47` (2026-04-22)
- **Vendored:** 2026-10-25
- **License:** MIT, Copyright (c) 2024 OpenAI (see `LICENSE`)
- **Paper:** Wei et al., *Measuring short-form factuality in large language models* (SimpleQA), 2024 — https://cdn.openai.com/papers/simpleqa.pdf
- **Dataset:** the official CSV that upstream itself loads, https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv (4,326 questions, cached at `baseline/data/simpleqa/`).

## Files

- `simpleqa_eval_upstream.py`: the upstream file, byte-for-byte, kept for provenance. It isn't importable here because it depends on simple-evals' `common`/`types` framework.
- `grader_template.py`: `GRADER_TEMPLATE` and the choice-letter constants, extracted from the upstream file with an AST parse (not retyped). It is checked byte-identical to upstream's value. Don't edit it by hand.

## How it's called (matching upstream)

Upstream's `simple_evals.py` builds its SimpleQA grader as `ChatCompletionSampler(model="gpt-4.1-2025-04-14", system_message="You are a helpful assistant.", max_tokens=2048)`, with the sampler's default `temperature=0.5`. `SimpleQAGrader` uses exactly these settings.

The reply is parsed the way upstream does it: the first `A`/`B`/`C` anywhere in the text, defaulting to `C`. A maps to CORRECT, B to INCORRECT, and C to NOT_ATTEMPTED.

Note that the paper's original grader was gpt-4o; the current upstream code uses gpt-4.1. We follow the current code.

## Deviations from upstream

- **Subsetting:** upstream draws subsets with `random.Random(0).sample(examples, n)`. We take the first N rows in CSV order, the way GAIA and Fermi are sliced in this repo. The CSV is already well mixed by topic (its first 20 rows span 7 topics), so the prefix isn't biased toward one topic, but it is a **different subset** from upstream's `num_examples=N`.
- **What gets graded:** upstream grades the model's full chat response. We grade whatever the agent passes to `final_answer(...)`, with `SIMPLEQA_ANSWER_INSTRUCTION` asking for a short answer.

## Sanity checks (2026-10-25)

- The grader returns the expected label on all 8 labeled examples from its own prompt: 5 Obama-children cases plus 3 cases of the "120k citations" numeric rule. These examples are in the prompt itself, so this checks the wiring (model access, formatting, parsing), not grader accuracy. The paper reports the grader's own validation.
