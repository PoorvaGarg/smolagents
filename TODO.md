# TODO

Working list for the `agents_as_prob_progs` branch. Ask me to "check the TODO" and I'll read
this back and update it. Last updated 2026-09-07.

## Runs in flight

As of 2026-09-07 17:14. All launched detached with `start_new_session`, so they survive a session
ending; logs in the session scratchpad under `v6n3/` and `v6n5/`.

| run | dir | pid | state |
|---|---|---|---|
| Qwen3.5-9B n=3 | `tracelet_direct_tp_v6n3_react_Qwen/Qwen3.5-9B` | 59581 | 120/165, 2 errors |
| Qwen3.7-Plus n=3 | `tracelet_direct_tp_v5n3_react_Qwen/Qwen3.7-Plus` | 59587 | 143/165, 5 errors (~6 are content-filter 400s that reproduce) |
| gpt-4o n=5 | `tracelet_direct_n5_tp_v6n5_react_gpt-4o` | 59856 | 67/165, 0 errors |
| gpt-5.4-mini n=5 | `tracelet_direct_n5_tp_v6n5_react_gpt-5.4-mini` | 59873 | 76/165, 0 errors |
| Qwen3.5-9B n=5 | `tracelet_direct_n5_tp_v6n5_react_Qwen/Qwen3.5-9B` | — | not started; watcher pid 60131 fires it when both n=3 sweeps exit |

**Health check:** pickle-count growth and newest-write time, *not* CPU (these are I/O bound —
a working kernel and a hung one both sit near 0.14s CPU per 25s). Recovery from a provider burst
is: move errored pickles to a `_backup_errors_<date>/` subdir inside the run dir, then relaunch
with the same `SMOLAGENTS_RUN_TAG` so cached results are reused.

**n=5 cost concern is resolved.** On 42-45 paired questions n=5 costs about the same as n=3, not
5x: gpt-4o 1.22x median / 0.92x total, mini 0.77x / 0.78x. My earlier 99M-token projection came
from extrapolating a single outlier question and was wrong. The interesting result is that n=5
uses **fewer steps** than n=3 (gpt-4o 10.6 -> 9.9, mini 8.2 -> 6.9), the opposite of n=3 vs n=1 —
so the step-reduction effect may need more than three candidates to outweigh protocol overhead.
Accuracy differences remain within noise on both models.

## Decisions waiting on me

- **Keep Qwen3.7-Plus at all, or swap in a different fourth model?** The immediate trigger is that
  n=5 is impossible there — the provider caps `n` at 4 (`Range of n should be [1, 4]`) — so the
  options are n=4, skip it, or drop every model to n=4. But the recurring friction is worth
  weighing before spending more on it:

  | against keeping it | for keeping it |
  |---|---|
  | streaming-only, so **no fill-in logprobs at all** — refuses `n>1` together with `logprobs` | the only model near the top of the GAIA range (66.1% naive, 69.1% at n=3) |
  | `n` capped at 4, so it can never match the other models' n | genuinely different family from the OpenAI pair |
  | 6 content-filter `400`s that reproduce, plus naive idx 164 permanently unrecoverable | already has full n=1 and n=3 results, so its history is sunk cost |
  | slowest per question (~5.6 q/h) and the main source of connection bursts | |

  A replacement needs: non-streaming support, `n>1` **with** `logprobs`, and ideally a different
  model family from gpt-4o / gpt-5.4-mini so the comparison still spans providers. Qwen3.5-9B
  already satisfies all three, which is why it is the only open-weight model carrying probability
  data. Decide before launching any further Qwen3.7-Plus sweep.
- **Retry inside `evaluate_agent`** — declined twice. Without it, an overnight provider burst
  permanently scores affected questions wrong (cost so far: 92 questions on one Qwen sweep, 44 on
  another). Recovery is manual quarantine + relaunch.
- **Local vLLM server, and a record-replay web cache.** vLLM removes every provider restriction
  hit today (n caps, logprobs-with-n>1, `-9999` placeholders, connection bursts) and its prefix
  caching attacks the dominant cost directly — memory resends are 53.9% of tokens. It serves an
  OpenAI-compatible API so `api_base` is the only change; do *not* use the `VLLMModel` class,
  which is the offline-batch path and loses prefix caching. It will **not** fix the noise floor,
  because live web results are the other half. For that, a record-replay HTTP cache for the
  browser would do more for less effort, and would also remove DuckDuckGo rate-limiting as the
  throughput ceiling that forced 0.15 q/s.

## Questions to walk through with me

- **"How did you conclude the n=3 null is the judge's fault?"** Worth pushing on, because it is an
  inference from converging evidence, not a controlled result. The chain was: (a) candidates are
  genuinely diverse — only 2% of steps on gpt-4o and 0.2% on mini produce identical fill-ins, so
  the mechanism is not collapsing; (b) n=1 vs n=3 changes 20-22% of answers but almost perfectly
  symmetrically (17/19, 13/19, 19/17; p≈0.87, 0.38, 0.87), which is what an uninformative selector
  looks like; (c) a probe on hand-built candidate sets showed the judge ranks correctly when there
  is a real quality gap ([10,3,0]) but returns [10,10,0] or [10,10,10] whenever candidates are
  merely plausible — a reliable error detector, an uninformative ranker.
  **The decisive control has not been run**: random selection among viable candidates instead of
  judge selection (item 1 below). And a competing explanation is not separated from it — per-step
  argument quality may simply not be the binding constraint, which the ceiling result supports
  (over half of GAIA is never solved in any draw). Ask me to lay out both and say which the
  evidence actually favours.

## Experiments worth running, highest value first

1. **Random-selection control at n=3.** Replace judge selection with a random pick among viable
   candidates. n=1 vs n=3 was p≈0.87 with ~20% of answers flipping symmetrically, so if random
   matches the judge, the judge contributes nothing. One-line change, one fewer LLM call per step.
2. **Trajectory-level marginalisation + a verifier.** Oracle union over two n=3 draws is
   **+12.5pp** (gpt-4o 27.5%→40.0%) and **+11.2pp** (mini 36.1%→47.3%). That is the measured
   headroom; the open question is whether a verifier can capture a third of it.
3. **Move the sentinels off tool arguments.** Diversity currently lives only in argument values,
   never in the plan, which is the leading explanation for the null result. On Fermi the natural
   sites are the decomposition factors.
4. **Fermi instead of GAIA for probability work.** 60-75% of questions score strictly between 0
   and 1 (continuous outcome, far more power than binary), 2-4x cheaper, answers are magnitudes so
   a posterior over log10(answer) is meaningful. Caveat: mini makes almost no tool calls there, so
   the current sentinel mechanism barely fires — which is an argument for item 3.
5. **Repeat the v5 n=1 arm.** Never run twice; every n=1 conclusion rests on a single draw.
6. **Context pruning at scale.** Implemented since June, 44 tests pass, never evaluated — there
   are zero pruning run dirs in `baseline/`.

## Goal-satisfaction context pruning (built 2026-09-07)

Implements the `notes.ipynb` pseudocode: the model declares an expectation per step, a judge
checks whether the observation satisfied it, satisfied steps close the subtask and drop the
failed attempts before them.

| file | what it is |
|---|---|
| `src/smolagents/prompts/context_pruning_agent.yaml` | custom system prompt, Expectation line in all 12 in-context examples, incl. a retry example that restates the same expectation |
| `src/smolagents/context_pruning/completion_judge.py` | one small call per step, sees only that step, returns **P(satisfied)** so the threshold is tunable after a run |
| `src/smolagents/context_pruning/goal_agent.py` | `GoalPruningCodeAgent`, `parse_expectation` |
| `tests/context_pruning/test_goal_agent.py` | 21 tests; suite is 65 passing |

Verified end to end on three shaped runs: all-fail (force-close path, pruned to 22%), all-succeed
(judge path, correctly prunes nothing), and **fail/fail/succeed — dropped both failures, kept the
successful step, 45% of content**. Judge probes: correct on 5 adversarial negatives including two
"related but not the thing" cases, and on 4 positives; commit rate on a sample of real trajectory
steps was ~40-50%.

**`</code>` is a stop sequence** (`agents.py:1659`), so nothing after the code block can ever be
generated. The Expectation line therefore goes **before** the code, which is also stronger — the
model commits to it before writing code and cannot retrofit it.

Still to do here:
- Measure the commit-rate distribution on a real GAIA sweep. If it is near 0% or 100%, nothing
  else about this design matters. Sweep `completion_threshold` on the recorded probabilities.
- Compare against two free baselines: fixed recent window of *k* steps, and the existing
  tool-group `ContextPruningCodeAgent`. If the judge cannot beat "keep the last 10 steps", the
  call is not earning its keep.
- Commit a one-line summary of dropped attempts instead of nothing, so a later subtask does not
  retry the same dead ends.
- Consider a `recall(step_number)` tool so pruning is reversible and mistakes are observable.
- Evaluate on **tokens** (measurable in one run); accuracy is a non-inferiority check only,
  given the ±2.4pp noise floor.

## Known defects and debt

- **`python_executor.state` is never cleared between questions.** `run(reset=True)` resets memory
  and the monitor but not executor state, so variables leak across all 165 questions. Unfixed.
- **`extract_primary_tool` iterates a `set`.** With randomised string hashing the "primary" tool
  on a line with two tool calls varies between processes, so pruning is not reproducible.
- **Context pruning prunes the wrong group hardest.** 54-56% of dropped steps are `browse` chains,
  where consecutive views hold *different* content; only 20-33% are the redundant repeated
  searches its docstring targets. It also keeps the *last* step of a segment, not the best.
- **`-9999` placeholder contamination.** OpenAI sends `-9999.0` for unscored tokens; 87 of 3,338
  gpt-4o candidate records in `_v6n3` are affected. Fixed going forward (`n_placeholder`), but the
  existing pickles are not repaired.
- **`tests/test_agents.py::...stream_logs_multiple_tool_calls_observations`** fails on `MockChoice`
  lacking `.index` — pre-existing baseline, not caused by this branch.
- **naive Qwen3.7-Plus idx 164** hits the Together content filter reproducibly; report as an error.

## Uncommitted

- `src/smolagents/tracelet_agent.py`, `src/smolagents/memory.py` — judge strategies
  (`scores` / `score_probs` / `token_probs`), `ActionStep.candidates`, placeholder fix
- `baseline/traceletReAct_trajectory.ipynb` — new, untracked
- `baseline/RUNS.md` — `_v6n5` runs not yet recorded

## Measurement rules established this session

- **Noise floor.** ~±2.4pp between identical runs, with 32.9% of questions flipping across three
  naive draws. Single-run differences under ~3pp are not interpretable, for either arm.
- **Ceiling.** Over half of GAIA is never solved in any draw (56.7% naive gpt-4o, 60.0% n=3).
  Algorithmic work only ever operates on the flippable quarter to third.
- **Liveness.** CPU delta does *not* distinguish a working run from a hung one (both ~0.14s/25s,
  the work is I/O bound). Use pickle-count growth and newest-write time.
- Watch the parent `nbconvert` PID's *child* kernel, not the parent, when inspecting a run.