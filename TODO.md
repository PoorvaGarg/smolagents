# TODO

Working list for the `agents_as_prob_progs` branch. Ask me to "check the TODO" and I'll read
this back and update it. Last updated 2026-09-12.

Commit `484598a` ("Finalizing experimental results for further analysis") landed; working tree
clean at that point. Full test suite: 1113 passed, 66 skipped, 2 failed -- both confirmed
pre-existing/environmental (the documented `MockChoice.index` baseline failure, and
`test_import_smolagents_without_extras` shelling out to a `uv` binary not installed on this
machine; reproduced identically by stashing to HEAD before this session's commit).

## Runs in flight

As of 2026-09-12, ~09:40 into these three. All from `launch_random_control.py` (see below),
detached with `start_new_session`:

| run | dir | pid | state |
|---|---|---|---|
| n=5 random-select, gpt-4o | `tracelet_direct_n5_tp_rand5_react_gpt-4o` | 79740 | alive, 0/165 (heavier per-question, first pkl not landed yet) |
| n=5 random-select, gpt-5.4-mini | `tracelet_direct_n5_tp_rand5_react_gpt-5.4-mini` | 79741 | alive, 7/165 |
| n=5 random-select, Qwen3.5-9B | `tracelet_direct_n5_tp_rand5_react_Qwen/Qwen3.5-9B` | 79742 | alive, 0/165 (first pkl not landed yet) |

Logs in `scratchpad/random_control/`. Everything else — naive/n=1/n=3/n=5 filter-on baselines for
all four models, goal-pruning for all four models, n=1 Qwen3.5-9B — is complete at 165/165.
Residual, not-yet-cleaned-up error counts on old runs: naive Qwen3.7-Plus 3, n=1 Qwen3.7-Plus 1,
n=3 Qwen3.7-Plus 1, n=5 Qwen3.5-9B 1, goal-pruning Qwen3.7-Plus 5 — likely the model's known
reproducible content-filter 400s rather than transient bursts; check before quarantining.

## Random-selection control at n=5 (launched 2026-09-12)

This is experiment 1 from the backlog below, finally run — decisive test of whether the judge
beats chance. `judge_strategy="random"` added to `TraceletCodeAgent` (`tracelet_agent.py`): no
judge call at all, uniform pick among viable candidates, `TokenUsage(0, 0)`. Verified standalone
(2000 draws, uniform distribution) and against all 77 existing tracelet tests before running a
live 2-question smoke test (correct `judge_kind="random"`, uniform scores, zero notebook errors)
prior to launching the full sweeps. `traceletReAct.ipynb` now exposes `SMOLAGENTS_JUDGE_STRATEGY`
(default `"scores"`, so every existing sweep's behaviour is unchanged).

Directly comparable to the existing `tracelet_direct_n5_tp_v6n5_react_<model>` judge-selection
sweeps (same model, same n, same prompt/filter/max-steps -- only `RUN_TAG` differs, `_rand5` vs
`_v6n5`, so they don't collide). **Qwen3.7-Plus excluded** — the provider hard-caps `n` at 4
regardless of logprobs (`Range of n should be [1, 4]`), so it cannot run n=5 at all; user chose
"3 models at true n=5" over the n=4 or skip-entirely alternatives.

**When these finish**: pair each against its `_v6n5` counterpart (accuracy McNemar + token/step
comparison). If random matches the judge within noise, the judge selection step is contributing
nothing at n=5 — directly answering the open question below.

## Judge-effectiveness diagnostic (measured 2026-09-12, offline, no new runs)

Two ground-truthable checks over already-recorded candidate judge scores, across all four models'
n=3/n=5 sweeps (4,867 judged steps with >=2 candidates):

| metric | value |
|---|---|
| mean ESS/n (1.0 = judge shows no preference at all) | **0.90-0.94**, consistent across every model |
| steps with ESS/n >= 0.95 (near-uniform, i.e. uninformative) | **74.3%** |
| steps with real discrimination (ESS/n < 0.5) | 3.9% |
| judge picked an **errored** candidate over a clean alternative | 26 times / 4,867 (0.53%) -- objectively wrong, no counterfactual needed |

Quantifies what TODO already suspected qualitatively (flat scores except on rare clear-cut cases).
This is diagnostic, not decisive -- it shows the judge often doesn't discriminate, not that doing
so costs anything; that's exactly what the random-selection control above is for.

## Trajectory-viewing notebooks rebuilt (2026-09-12)

- `baseline/traceletReAct_trajectory.ipynb` — was 11 cells (a live-agent path plus a saved-sweep
  path bolted on); rebuilt to **4 cells**: intro, renderer, config+loader, ranking. Drops the
  entire live-agent path (model/tools/agent/dataset setup, running one live question) -- this
  notebook now only reads completed sweep pickles, no API calls, no keys needed. The loader's
  `RUNS` dict covers all 11 completed (model, arm) combinations, not just the three n=5 sweeps it
  originally shipped with, and flags which arms carry residual errors. Restart the kernel before
  using it (autoreload doesn't survive the cell-numbering/state change cleanly).
- `baseline/goalPruningTrajectory.ipynb` — replays a goal-pruning run's buffer offline (no model
  calls) showing exactly what was committed/open/dropped at each step; used to diagnose the q0
  arXiv-ID-loss bug below.
- `baseline/logprob_nondeterminism.ipynb` — demonstrates identical fill-in text getting different
  logprobs within one `n>1` request; established cause is lack of batch invariance in serving
  kernels (confirmed to reproduce on local vLLM too, not an API-only artefact), not the MoE-routing
  guess floated earlier in this file's history.

**Qwen3.7-Plus n=5 is impossible** — verified live 2026-09-07, not just from notes:

| request | result |
|---|---|
| `n=4` | OK |
| `n=4` + logprobs | `The n parameter must be 1 when logprobs is true` |
| `n=5` | `Range of n should be [1, 4]` |

Qwen3.5-9B accepts `n=5` **with** logprobs (5 choices returned). So the n=5 arm can only ever span
three models unless Qwen3.7-Plus is replaced, or every model drops to n=4.

### Laptop-sleep outage, 2026-09-07

Closing the laptop to travel killed the network mid-sweep. Result: **79 `Connection error` pickles
in contiguous tails** (gpt-4o n=5 idx 97-134, mini n=5 idx 118-164, Qwen3.7-Plus idx 161-164) and
three processes left alive but wedged with no writes for 170-240 minutes. Recovery: kill the
wedged pids, move errored pickles to `_backup_errors_20260907/` inside each run dir (not deleted),
relaunch with the same `RUN_TAG` so the good results stay cached.

**15 of the 79 quarantined errors are deterministic and will re-error**: 7 context-overflow on
gpt-4o n=5 (idx 79-85), 2 on Qwen3.5-9B n=3 (idx 81-82), and 6 Qwen3.7-Plus content-filter 400s.
They re-run cheaply because they die fast. Fixing the context ones means truncating the judge
prompt (below), which changes the method mid-run — so it was deliberately *not* done here.

**Accuracy read off a sweep with many errored pickles is deflated**, since an errored question is
scored wrong. mini n=5 read 23.9% with 45 errors but 33.1% on its 118 non-errored questions. Check
the error count before quoting any accuracy.

### Judge prompt is unbounded — the cause of the context overflows

`tracelet_agent.py:671` and `:696` interpolate `{t.observation}` **raw** for every candidate, and
`code_output.logs` is never truncated either (`:642`; only the "Last output" tail is, at `:643`).
Only the pickle *record* is bounded, to 2000 chars (`:584`). So the judge prompt scales as
**n x full observation**: requests of 132k-599k tokens against gpt-4o's 128k limit. n=3 saw 0
errors across all 165 on gpt-4o — three copies fit, five do not. One-line fix, and it would cut
judge cost at every n; do it between runs, not during one.

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

1. **Random-selection control.** ~~At n=3~~ — **launched at n=5 2026-09-12**, see the section
   above; results pending. n=1 vs n=3 was p≈0.87 with ~20% of answers flipping symmetrically, so
   if random matches the judge, the judge contributes nothing.
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
| `baseline/goalPruningReAct.ipynb` | GAIA sweep notebook, env-parameterised like `traceletReAct.ipynb` (`SMOLAGENTS_MODEL`, `_RUN_TAG`, `_JUDGE_ON`, `_COMPLETION_THRESHOLD`); `JUDGE_ON=0` is the prompt-only control arm |

**Judge works on all four models** (verified live 2026-09-07). Qwen3.5-9B returns graded logprobs
(0.997 / 0.0003 on a yes/no pair). Qwen3.7-Plus is streaming-only and `agglomerate_stream_deltas`
builds a fresh `ChatMessage` with no `raw`, so `_answer_probability` finds nothing and
`judge_completion` falls back to reading the word — correct verdicts, but binary 1.0/0.0 instead of
graded. Since the judge is saturated in practice anyway (144/187 gpt-4o probabilities below 0.01),
that costs almost nothing.

**No `max_tokens` cap on the judge call.** gpt-5.4-mini rejects the parameter outright
(`Unsupported parameter: 'max_tokens'... use 'max_completion_tokens'`) and smolagents does not
normalise between the two, so capping it would hard-fail every mini run. The prompt already pins
the answer to one word and gpt-4o returns exactly 1 output token, so the cap bought nothing.
| `tests/context_pruning/test_goal_agent.py` | 22 tests; suite is 66 passing |

**`</code>` is a stop sequence** (`agents.py:1659`), so nothing after the code block can ever be
generated. The Expectation line therefore goes **before** the code, which is also stronger — the
model commits to it before writing code and cannot retrofit it.

### Force-close removed 2026-09-07

The design originally closed a subtask after `max_open_steps=4` unsatisfied steps, keeping the
highest-scoring one. **Removed**: a judged success is now the only thing that ever prunes, so the
worst case is that nothing is pruned and the agent runs as plain ReAct — the right floor.

Why it had to go, from the aborted gpt-4o sweep:
- The keep rule ranked by `completion_prob`, but that quantity answers *"did this step deliver the
  expected content"* — which systematically scores **navigation** steps near zero. Measured on q1:
  a `web_search` returning Wikipedia links scored 0.356 and was kept, while
  `visit_page(FactSheet.aspx?speciesID=3243)` — the correct page — scored 0.018 and was dropped.
  The next step re-fetched the identical URL, and the browser said so: *"You previously visited
  this page 3 seconds ago."*
- Most probabilities were exactly 0.000, so the `(prob, step_number)` key degenerated to "keep the
  most recent" — the same defect already logged against the tool-group pruner.
- **4 of 6 closes were force-closes**, so the path with the weak rule dominated.

### Measured on the 7-question aborted sweep (gpt-4o, force-close still in)

| | goal-pruning | naive gpt-4o |
|---|---|---|
| commit rate | **3.9%** (3/76 steps) | — |
| steps, q0 / q1 | 32 / 46 | 12 / 6 |
| tokens, q0 / q1 | 296k / 594k | 131k / 33k |
| last-step input tokens, q0 / q1 | 15.2k / 18.7k | 13.5k / 7.8k |

Prompt compliance was fine (88-100% of steps stated an Expectation) and the judge itself is only
~9% of the spend. The problem is the **step count**: 8.2 → 39 mean. Pruning does reduce context
relative to its own step count (~11 committed of 46) but the explosion swamps it, so absolute
per-step context ended up *higher* than naive. Judge probabilities are saturated — 70/76 below
0.01, none above 0.99 — so retention is identical at threshold 0.5 and 0.9 and post-hoc threshold
sweeping buys nothing.

Still to do here:
- **Isolate the step explosion.** Two candidate causes, not separated: the Expectation prompt
  changed the model's behaviour, or pruning destroys context and the agent redoes work.
  `JUDGE_ON=False` (same prompt, no judge, no pruning) answers it directly. Run both arms on
  ~15 questions before committing to any 165 sweep — the full run projected to **73M tokens**.
- Re-measure the commit rate now that force-close is gone. At 3.9% the agent is plain ReAct paying
  for a judge call every step, which is a strictly-worse configuration.
- Compare against two free baselines: fixed recent window of *k* steps, and the existing
  tool-group `ContextPruningCodeAgent`. If the judge cannot beat "keep the last 10 steps", the
  call is not earning its keep.
- Commit a one-line summary of dropped attempts instead of nothing, so a later subtask does not
  retry the same dead ends.
- Consider a `recall(step_number)` tool so pruning is reversible and mistakes are observable.
- Evaluate on **tokens** (measurable in one run); accuracy is a non-inferiority check only,
  given the ±2.4pp noise floor.
- `SideContextBuffer.commit_subtask(keep=...)` now has no caller — the argument existed only for
  force-close. Left in place (tested, one line, shared with the other pruning agent); drop it if
  you want the dead parameter gone.

## Redundancy-based pruning judge (designed 2026-09-09, not implemented)

Parked to go look at ProbReAct results; pick this up when ready to build the "simple version."

**The bug that motivated it.** Replaying `goalprune_gp1_react_gpt-4o/0.pkl` in
`goalPruningTrajectory.ipynb` (gpt-4o, force-close already removed): step 12 finds the correct
arXiv paper (`2206.04615`); step 10 (a content-free "let's summarize and give up" non-action, no
real tool call) is separately judged satisfied and commits, wiping steps 1-9; step 23 (an unrelated
"academic database" tangent) is later judged satisfied and commits, wiping steps 11-22 — including
step 12. Confirmed `2206.04615` never appears in any step's code or thought again after step 23.
The agent re-searches variations of the same two query families up to 11 times afterward (verified
via the literal query strings) and hits the 50-step cap without answering. This is
`commit_subtask()`'s single-survivor design (already logged below) caught in the act, not a
one-off — and it shows the existing judge question ("did the observation satisfy the expectation")
is answered per-step with no protection for facts found along the way.

**Design arrived at, after several rejected shapes:**
- *Rejected: batch reconciliation every step* ("which of the currently-open steps are now
  redundant, given everything since") — correct in principle but O(n^2) over a trajectory, since
  the compared set grows every step it's invoked.
- *Rejected: useful/resolved/ruled-out three-way classification with a one-line summary for ruled-
  out attempts* — explicitly declined ("I don't want a summarizing step").
- **Chosen:** one `resolved` check per step ("would the agent need to repeat this action, with
  different phrasing, to get more or better information?"). If not resolved, the step joins
  `pending`. If resolved, commit the step to `memory`, then ask **one all-or-nothing question**:
  "does this step make every one of `pending` unnecessary to redo?" — if yes, wipe `pending`
  entirely; if no, leave it untouched. A boolean over the whole set is a much higher, safer bar
  than asking the judge to pick out a specific subset, which just gives it a new way to
  mis-select and wipe something valuable — the exact failure being fixed.

```python
system_prompt = ...
task = ...
is_final = False
memory = system_prompt + task
pending = []
while not is_final:
    thought, action = sample(LLM(. | memory + pending))
    is_final = check(action)
    if is_final:
        answer = extract_answer(action)
    else:
        result = execute(action)
        step = (thought, action, result)
        resolved = sample(LLM(. | task, step,
                                 "Would the agent need to repeat this action, with "
                                 "different phrasing, to get more or better information?")) == "no"
        if not resolved:
            pending += [step]
        else:
            memory += step
            supersedes_all = sample(LLM(. | task, step, pending,
                                            "Does this step make every one of these earlier "
                                            "steps unnecessary to redo?")) == "yes"
            pending = [] if supersedes_all else pending
return answer
```

No Expectation line needed — this judge doesn't check the step against a self-declared goal, so
the custom `context_pruning_agent.yaml` prompt could likely be dropped in favour of plain
`code_agent.yaml`, removing that confound rather than needing `JUDGE_ON=False` to isolate it.

**Known, measured limitation: doesn't compress interleaved multi-part questions.** The all-or-
nothing gate only fires cleanly when `pending` holds one line of inquiry at a time. Measured on
`naive_react_gpt-4o_False_fixedbrowser` (165 questions, unpruned, so this is the model's natural
work pattern): a crude `" and "`-split heuristic found 22/165 (13%) structurally two-part — an
undercount, since q0 itself is two-part but phrased without "and" and wasn't even caught. Of those
22, **7 (32%) showed genuine interleaving** (>=2 thread switches, not just "finish A then start
B") — q155 switched threads 9 times over 51 steps. So roughly 4-5% of GAIA overall gets zero
compression benefit from this design; accepted as a v1 limitation rather than adding thread-scoped
`pending` lists, which would reintroduce the subtask-detection complexity this is trying to avoid.

Next step when resumed: implement as a new judge module (two calls: `resolved`, then
`supersedes_all` only when resolved), wire into a `GoalPruningCodeAgent` variant with
`prompt_templates=None`, and replay it against `goalprune_gp1_react_gpt-4o/0.pkl`'s question
specifically to confirm `2206.04615` survives while the AI-regulation search family gets dropped
once step 11 succeeds.

## Known defects and debt

- **`python_executor.state` is never cleared between questions.** `run(reset=True)` resets memory
  and the monitor but not executor state, so variables leak across all 165 questions. Unfixed.
- **`extract_primary_tool` iterates a `set`.** With randomised string hashing the "primary" tool
  on a line with two tool calls varies between processes, so pruning is not reproducible.
- **Context pruning prunes the wrong group hardest.** 54-56% of dropped steps are `browse` chains,
  where consecutive views hold *different* content; only 20-33% are the redundant repeated
  searches its docstring targets. It also keeps the *last* step of a segment, not the best. The
  goal-pruning agent hit the same "keeps the wrong step" failure via its force-close path, now
  removed.
- **A judged success keeps the step that satisfied it, which may not be the step worth keeping.**
  `commit_subtask()` promotes `current_context[-1]`. That is right when the satisfying step also
  holds the content, but a step whose expectation was met by a *summary* still discards the
  navigation steps that produced it.
- **`-9999` placeholder contamination.** OpenAI sends `-9999.0` for unscored tokens; 87 of 3,338
  gpt-4o candidate records in `_v6n3` are affected. Fixed going forward (`n_placeholder`), but the
  existing pickles are not repaired.
- **`tests/test_agents.py::...stream_logs_multiple_tool_calls_observations`** fails on `MockChoice`
  lacking `.index` — pre-existing baseline, not caused by this branch.
- **naive Qwen3.7-Plus idx 164** hits the Together content filter reproducibly; report as an error.

## Uncommitted

- `src/smolagents/context_pruning/goal_agent.py` — force-close path removed
- `src/smolagents/context_pruning/completion_judge.py` — `max_tokens=4` cap on the judge call
- `tests/context_pruning/test_goal_agent.py` — force-close tests replaced by the ReAct-floor
  tests, plus a regression test that the monitor total includes the judge's tokens
- `baseline/goalPruningReAct.ipynb` — new, untracked

## Measurement rules established this session

- **Noise floor.** ~±2.4pp between identical runs, with 32.9% of questions flipping across three
  naive draws. Single-run differences under ~3pp are not interpretable, for either arm.
- **Ceiling.** Over half of GAIA is never solved in any draw (56.7% naive gpt-4o, 60.0% n=3).
  Algorithmic work only ever operates on the flippable quarter to third.
- **Liveness.** CPU delta does *not* distinguish a working run from a hung one (both ~0.14s/25s,
  the work is I/O bound). Use pickle-count growth and newest-write time.
- Watch the parent `nbconvert` PID's *child* kernel, not the parent, when inspecting a run.