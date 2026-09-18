# TODO

Working list for the `agents_as_prob_progs` branch. Ask me to "check the TODO" and I'll read
this back and update it. Last updated 2026-09-15.

## RESUME HERE: 3 of 4 full beam-search sweeps running/done, DSL compiler built, empty-code root cause found (fix deferred) (2026-09-17)

**Sweep status** (all via `baseline/run_beam_search_sweep.py`, `beam_width=2, branch_factor=2,
max_steps=20`, resumable -- rerun the same command to continue from cached pickles):
- **gpt-4o: DONE, 165/165, 49/165 = 29.7% correct.**
- gpt-5.4-mini: running, pid 3335, resumed at 131/165 (log:
  `baseline/beamsearch_bw2_bf2_react_gpt-5.4-mini_launch.log`).
- Qwen/Qwen3.5-9B: running, pid 3336, resumed at 50/165 (log:
  `baseline/beamsearch_bw2_bf2_react_Qwen_Qwen3.5-9B_launch.log`; note the `run_name` for this
  one literally contains a `/`, matching existing repo convention -- pickles are under
  `baseline/beamsearch_bw2_bf2_react_Qwen/Qwen3.5-9B/`, trees under
  `.../Qwen3.5-9B_trees/`).
- **Qwen/Qwen3.7-Plus: blocked, not running.** Together API rejects it with
  `third_party_data_sharing_blocked` ("This model requires third-party data sharing to be
  enabled for your organization") -- an account-level Together dashboard setting, not fixable in
  code. Earlier tracelet/naive runs in this repo used this model fine, so this is a newer
  restriction. Needs the user to enable it in their Together account, or skip this model.
- All three OpenAI/Together-routing goes through `models = {...}` + provider branch in
  `run_beam_search_sweep.py`'s `main()` -- **this branching was missing in the first version** of
  that script (it always called OpenAI's API regardless of `SMOLAGENTS_MODEL`), which silently
  produced 165/165 "invalid model ID" failures for both Qwen models on the first launch attempt;
  fixed, verified with a live single-question test for both Qwen models before relaunching.

**naiveReAct gpt-5.4-mini comparison run: DONE.** `naive_react_gpt-5.4-mini_False_rerun_emptycode`
(165/165, fresh run launched specifically to check reproducibility): 33.9% accuracy, 33.4% empty
code (319/954 steps) -- closely matches the pre-existing `_fixedbrowser` run (32.1%, 27.6%),
confirming ~28-33% empty-code is a stable gpt-5.4-mini/naiveReAct characteristic, not a fluke.

**Key finding -- BeamSearchCodeAgent makes gpt-5.4-mini's empty-code problem much worse, and the
mechanism is now confirmed, not just hypothesized.** Across the first 16 questions of the
gpt-5.4-mini beam-search sweep: **90.4% of candidates (756/836) had empty code**, accuracy only
1/16 = 6.25% (naiveReAct on the same model: ~30% empty code, ~33% accuracy -- both roughly 3x
worse under beam search). Root cause, verified against real data
(`beamsearch_bw2_bf2_react_gpt-5.4-mini/0.pkl`, step 0 codeless -> step 1's
`model_input_messages` is exactly `[SYSTEM, USER(task), ASSISTANT(bare codeless text)]`, nothing
after):
- `_expand_beam`'s codeless branch (`beam_search_agent.py`, the `else:` when
  `step.code_action is None`) never sets `step.observations` or `step.error`.
- `ActionStep.to_messages()` (`memory.py`) only appends an `Observation:` message
  `if self.observations is not None`, and an `Error:` message `if self.error is not None`. With
  both `None`, a codeless step contributes *only* the bare assistant text to history.
- Consequence: the message list built for the *next* round's `_sample_step` call ends on a
  dangling `ASSISTANT` turn with no `USER`/`TOOL_RESPONSE` in between -- a malformed multi-turn
  structure. naiveReAct never has this gap: a parse failure sets `step.error`, which
  `to_messages()` turns into an explicit `Error:...Now let's retry...` `TOOL_RESPONSE`, guaranteeing
  a fresh turn always follows every assistant turn. This gap compounds every round sampled from
  an affected beam's history, matching the 42.7% "both thought and code empty" rate seen deeper in
  gpt-5.4-mini's trajectories.
- **A fix was proposed** (set `step.observations` to a neutral, non-scolding string like
  `"Execution logs:\n(no code block was provided in this step)"` in that `else:` branch, so
  `to_messages()` always emits *something* -- keeping the score-only survival rule from decision 7
  completely untouched, just fixing the conversational structure) **but the user explicitly
  declined to apply it now** ("why do we need a fix, just restart the process") when asked to
  confirm before implementing. **Do not apply this fix without being asked again** -- the
  currently-running gpt-5.4-mini/Qwen3.5-9B sweeps are running WITHOUT it, by explicit choice.

**Pending accuracy comparison (asked for, not yet done on the full set)**: on a 71-question
matched subset, gpt-4o beam search beat naive gpt-4o (`_var1/2/3`, filter-on, mean 25.1%) by
~3-10pp (32.4% vs 22.5-29.6%). Now that gpt-4o's full sweep is done (165/165, 29.7%), **redo this
comparison against the full 165-question naive baselines** (still `_var1/2/3`, NOT the
filter-off `_fixedbrowser` run -- see CLAUDE.md's leak-filter warning) -- not done yet. Repeat for
gpt-5.4-mini and Qwen3.5-9B once their sweeps finish.

**Probabilistic-program notation, settled by hand then implemented and verified** (this is the
answer to the user's earlier step 4 "write a translation from trajectories to probabilistic
programs" -- superseding the original tracelet-oriented `Choice/Evidence/Prune/Trace` schema
sketch further down this file, which was designed before beam search existed):
```
theta_<parent> = exp(thought_logprob_i) / sum(exp(thought_logprob_j) for j in remaining siblings)
thought<beam_id_i> = flip(theta_<parent>)              # first candidate
thought<beam_id_j> = (not thought_i) and flip(theta')  # subsequent candidates, stick-breaking
thought<beam_id_last> = not (all previous)              # last candidate forced, no flip needed

code<beam_id> = if thought<beam_id> then <the actual code text> else Error   # Error also covers
                                                                              # genuine no-code-generated candidates
obs<beam_id> = execute(code<beam_id>)
factor(log judge(thought, code, obs))                   # log(max(p_yes, 1e-4))

{pool} = resample_top_k({pool}, k=beam_width)  # pool = this round's new children PLUS any
                                                # already-final beam carried forward unexpanded
```
Settled decisions along the way: `theta` normalizes only over `thought_logprob` (code was
explicitly decided to carry no probability of its own -- "eliminate probability associated with
code"); `flip`s are a proper stick-breaking sequence for `branch_factor > 2`, not independent
per-candidate flips (verified: two independent flips could both come out true/false, which breaks
"exactly one realized" -- caught by the user reviewing round 1 by hand).

**Implementation, in `baseline/compile_trace_to_program.py`** (built one function at a time, each
verified against real Q0 data before moving on): `theta(children)` (stick-breaking sequence, any
N), `render_branch_point(parent_id, children)`, `render_round(round_data, prev_kept)` (handles the
carried-forward-frozen-final case), `find_winner(rounds)` (replicates `run()`'s own winner rule),
`compile_trace(rounds, question, true_answer, prediction)`, plus a CLI (`main()`). Verified
byte-for-byte consistent with the hand-derived Q0 program (8 rounds, winner
`0.0.1.0.0.0.0.1.1`, answer `"egalitarian or hierarchical"`). Run it (must use the venv's
Python, and from inside `baseline/` since paths are relative):
```
cd baseline && /Users/poorvag/Work/smolagents/.venv/bin/python compile_trace_to_program.py <run_name> <index>
```
e.g. `compile_trace_to_program.py beamsearch_bw2_bf2_react_gpt-4o 0`.

**`baseline/beamSearchReAct_single.ipynb` improvements** (answers the user's step 1 "improve
looking at trajectories"): cell 8 now renders an actual box-drawing tree (ancestry via `beam_id`,
not a flat round list), tagging `★ WINNER` / `FINAL` / `ERRORED` / `CUT (pruned)` /
`UNFINISHED (search ended)` per node, plus `p_yes`/`thought_lp`/`code_lp`/`pruning_score`. Cell 6
supports `LOAD_FROM_FILE` (old single-file dry-run JSONs) or `LOAD_FROM_SWEEP = (run_name, index)`
(loads a sweep's `.pkl` + tree `.json` pair directly, no API cost) instead of re-running the
agent. Cell 9 (new, last) calls `compile_trace` on whatever's currently loaded. The old
"save trajectory to JSON" cell was dropped during the `LOAD_FROM_SWEEP` refactor and confirmed
by the user as not needed -- do not re-add it.

**Failure-case findings from before the full sweep, still valid** (see the untouched
"premature convergence" section immediately below this one for full detail on Q2/Q4/Q6): a
confidently-wrong finished beam beats a still-correctly-working unfinished one because the judge's
"did this achieve the thought's expectation" question is nearly tautological for a `final_answer`
step (the model said it would answer, then did -- of course that "succeeds"). This is a distinct
issue from the gpt-5.4-mini empty-code/message-gap finding above; neither has been fixed yet, both
are documented, and no design change should be made to either without the user asking first.

**Exact next steps when resuming**:
1. Check on gpt-5.4-mini (pid 3335) and Qwen3.5-9B (pid 3336) sweep progress; they were at
   131/165 and 50/165 respectively as of 2026-09-17.
2. Once each finishes, compute its empty-code rate (gpt-5.4-mini in particular) and accuracy, and
   the matched-baseline naive comparison, the same way it was done for gpt-4o.
3. Redo the gpt-4o vs naive full-165 accuracy comparison (currently only done on a 71-question
   subset).
4. Qwen3.7-Plus stays blocked until the user enables third-party data sharing on Together, or
   they say to skip it permanently.
5. The codeless-step message-gap fix and the premature-convergence/judge-circularity issue both
   remain proposed-but-not-applied by explicit user choice -- revisit only if asked.

---

**State**: `BeamSearchCodeAgent` is implemented, unit-tested, and smoke-tested (gpt-4o and
Qwen3.5-9B both got Q112 right, 3 rounds, via `baseline/beamSearchReAct_single.ipynb`). Dry run
extended to 15 GAIA questions total (gpt-4o, `beam_width=2, branch_factor=2, max_steps=20`):
**5/15 correct** (Q0, Q1, Q5, Q13, Q14 right; Q2, Q3, Q4, Q6, Q7, Q8, Q9, Q10, Q11, Q12 wrong).
Scripts: `beam_5q_dryrun.py` (Q0-Q4) and `beam_10q_dryrun.py` (Q5-Q14), both in
`/private/tmp/claude-502/-Users-poorvag-Work-smolagents/2a2cc3b2-ba33-45f3-9f0d-3e21f682d237/scratchpad/`
(a session-scoped scratchpad -- may not survive indefinitely; the per-question JSON trees it
wrote under `baseline/beam_dryrun_q*.json` and `beam_dryrun_summary*.json` are the durable
record, though those are gitignored `*.json` so they only exist on this machine). Full round-by
-round beam trees per question are in those JSON files if deeper re-inspection is needed later.

**Key finding -- premature convergence, now with 15-question support**: a "premature
convergence" heuristic (did the last round keep a finished beam while cutting an unfinished
sibling, on a question that came out wrong) fired on **4 of the 10 wrong answers in the second
batch** (Q6, Q7, Q11, Q12), on top of the two clear qualitative examples from the first batch
(Q2, Q4 -- see below). That's a real, recurring pattern, not a one-off:
- Q4 (Kipchoge/Moon, true `17`): two branches reached `final_answer("17000")` (off-by-1000-unit
  slip) with high judge confidence (p_yes 0.97, 0.79) and won pruning, while two siblings had
  *just computed the correct value* (`17`) that same round but hadn't called `final_answer` yet
  -- their lower pruning_score (not-yet-final beams score lower almost by construction) got them
  cut before they could finish.
- Q2 (Nature p-values): a branch verifying the real article count on Nature's site got pruned in
  favor of two branches that guessed an unverified number and finished confidently (wrong).
- Q9, Q10 hit `max_steps=20` without any beam ever finishing, so `run()`'s fallback
  (`max(beams, key=pruning_score)` among the unfinished) kicked in -- Q9 returned a raw scraped
  webpage as the "answer", Q10 returned `None`. Working as designed (per the earlier decision:
  "just return the best beam's last action_output"), but a visible symptom of the same underlying
  issue: many candidates never converge cleanly, and codeless/thin candidates crowd the pool
  (Q9: 35/78 candidates had no code at all; Q10: 44/78; Q13 -- one of the *correct* ones --
  still had 33/46).
- **Conclusion**: the per-step "did this achieve the thought's expectation" judge tends to reward
  *reaching* `final_answer` over *being right*, so a confidently-wrong finished beam beats a
  still-correctly-working unfinished one. This is now empirically supported across 15 questions,
  not just anecdotal -- **worth addressing before a full sweep**, per the user's own ordering
  (next step 3 below is "run a whole sweep," explicitly listed after understanding this).

Also generic, not new: codeless candidates (model replies in prose, no `<code>` block) are common
across all models tested, not just gpt-5.4-mini's earlier degenerate case -- rates ranged from
0/6 to 44/78 per question across all 15. Per the user's explicit
design call, these are NOT treated as errored; they go to the judge and their score alone decides
survival (see decision 7's sibling note in `beam_search_agent.py`'s `_sample_step`/`_expand_beam`).

**User's 4 next steps, in the order given (2026-09-15), none started yet**:
1. Improve looking at the trajectories in the Jupyter notebook (`baseline/beamSearchReAct_single.ipynb` exists as a first cut -- round-by-round beam tree printout; "improve" likely means better formatting/navigation, not rebuilding from scratch, but ask rather than assume).
2. Look at the failure cases for a few questions (Q2, Q3, Q4, Q6, Q7, Q8, Q9, Q10, Q11, Q12 are all wrong -- 10 candidates to pick from; Q2/Q4/Q9/Q10 above already have some analysis, the rest don't yet).
3. Run a whole sweep (only after the dry-run pattern above is understood/resolved -- don't jump straight here).
4. Write a translation from trajectories to probabilistic programs (this connects back to the session's earlier custom-DSL design discussion -- `Choice`/`Evidence`/`Prune`/`Trace` -- mentioned before beam search implementation began; no code for this exists yet).

---

## Previous state: beam search implementation -- code complete, live smoke test done (2026-09-15)

`BeamSearchCodeAgent` (`src/smolagents/beam_search_agent.py`, untracked, not yet committed) is
now fully implemented and unit-tested. Working style stays strictly incremental: one function at
a time, explained before/while writing, confirmed before moving to the next -- keep that up for
whatever comes next (the live smoke test, then the 5-question dry run).

**Checkpoint verified**: full suite green -- 1126 passed, 66 skipped, 2 failed (both the same
pre-existing baseline failures, unrelated to this work: `TestToolCallingAgent::test_toolcalling_
agent_stream_logs_multiple_tool_calls_observations` and `test_import_smolagents_without_extras`).
10 new tests in `tests/test_beam_search_agent.py`, all passing.

**What's in `beam_search_agent.py`**:
- `_sum_scored_logprobs` / `_split_token_logprobs`: split one completion's per-token logprobs into
  a "thought" half and a "code" half at the opening code-block tag (everything before is thought,
  the tag onward is code). Reuses `PLACEHOLDER_LOGPROB`/`_FillinLogprob` from `tracelet_agent.py`.
- `Beam` dataclass: `beam_id`/`parent_id` (lineage), `memory` (this beam's own `AgentMemory`,
  never `self.memory`), `executor_state`/`tool_state` (this beam's own snapshot),
  `expectation_scores`/`thought_logprobs`/`code_logprobs` (per-step, running), `errored_this_step`,
  `is_final`/`final_output`. `extend()` builds a child without mutating the parent.
  - `score` = mean(expectation_scores) -- judge-only, kept purely as a diagnostic/reporting value.
  - `pruning_score` = mean over steps of `log(p_yes) + thought_logprob + code_logprob` (log(p_yes)
    floored at `log(1e-4)` so a zero judge score doesn't send it to -inf) -- **this is what
    actually drives both pruning and winner selection now** (see decision 7 below).
- `BeamSearchCodeAgent(CodeAgent)`: `beam_width`/`branch_factor` params.
  - `_build_root_beam(task, images)` -- fresh `AgentMemory` (system prompt + `TaskStep`), empty
    executor/tool state, `beam_id="0"`.
  - `_beam_messages(memory)` -- `write_memory_to_messages` against an explicit `memory` arg.
  - `_sample_step(memory)` -- one independent full (thought, code) resample (no skeleton/fill-in
    split); returns an unexecuted `ActionStep` plus the thought text and the two split logprobs.
  - `_execute_step(step, executor_state, tool_state)` -- runs `execute_against_snapshot`, writes
    `.observations`/`.action_output`/`.is_final_answer`/`.timing.end_time`/`.error` onto `step` in
    place, returns `(post_executor_state, post_tool_state, errored: bool)`.
  - `_judge_step(thought, step)` -- one `judge_step_achieved_expectation` call, returns
    `(p_yes, token_usage)` (`p_yes` defaults to 0.0 if unparseable).
  - `_expand_beam(beam)` -- `branch_factor` children off one live beam: sample, execute, judge,
    fold judge token cost into the step's `token_usage`, `extend()`. Never called on `is_final`.
  - `_prune(beams)` -- keep top `beam_width`, `key=lambda b: (not b.errored_this_step,
    b.pruning_score)`.
  - `_stateful_backends()` -- delegates to the shared `find_stateful_backends`.
  - `run(task, reset=True, images=None, max_steps=None)` -- builds the root beam; loops (each
    round: `is_final` beams pass through untouched, others `_expand_beam`; combine and `_prune`)
    until every beam is `is_final` or `max_steps` hits; winner = highest-`pruning_score` finished
    beam if any finished, else highest-`pruning_score` beam overall; copies the winner's steps
    into `self.memory` (so `evaluate_agent`, which reads `agent.memory.steps` afterward, sees an
    ordinary-looking trajectory) and returns `winner.final_output` or, if nothing ever finished,
    the winning beam's last `action_output`.

**Refactor done alongside** (to avoid duplicating logic between `TraceletCodeAgent` and
`BeamSearchCodeAgent`, per explicit instruction not to duplicate code):
- `src/smolagents/utils.py` gained three shared functions: `safe_deepcopy` (moved from
  `tracelet_agent.py`'s `_safe_deepcopy`), `find_stateful_backends(tools)` (moved from
  `tracelet_agent.py`'s `_stateful_backends` method body, now takes `tools` as a plain arg),
  and `execute_against_snapshot(python_executor, tools, code, executor_state, tool_state)` (new
  -- installs a given state, runs one code string, returns `(observation, output,
  is_final_answer, error, post_executor_state, post_tool_state)`; does **not** restore prior
  live state itself, that's the caller's job).
- `tracelet_agent.py`: `_stateful_backends()` and `_execute_candidates()` both now delegate to
  the shared functions instead of having their own copies; `_safe_deepcopy` and the now-unused
  `import copy` were removed. All 80 tracelet tests still pass after this refactor.

**Design decisions settled this session** (don't re-litigate, just implement):
1. Errored candidate: still judged (never hard-zeroed to p=0.0), but ranks below every
   non-errored candidate in pruning regardless of score -- lexicographic tier, not a score
   penalty. Can still survive a prune if there aren't enough clean candidates to fill
   `beam_width` -- "the trajectory can still recover."
2. A beam that reaches `final_answer` freezes (carries its score forward unchanged, is not
   branched further each round) but is **not** exempt from pruning -- it still competes on the
   same `(errored, pruning_score)` key every round and can be cut before the search ends. Only at
   the very end does "did it finish" gate before "how well did it score": any finished beam beats
   any unfinished one regardless of score; ties among finished beams broken by `pruning_score`.
   If nothing ever finishes by `max_steps`, fall back to `max(beams, key=pruning_score)` among the
   unfinished ones.
3. First real test: `beam_width=2`, `branch_factor=2`, `max_steps=20`, on 5 questions run live
   and manually inspected -- before anything resembling a full 165-question sweep.
4. `branch_factor` samples are independent re-calls to the model at its default temperature
   (1.0) -- no forced diversity penalty, matches what was already validated (near-100%
   uniqueness across independent full-step resamples, see the naiveReAct-vs-tracelet experiments
   earlier in this file).
5. Thought/code split point: everything before the opening code-block tag is thought, the tag
   onward is code (settled explicitly, see `_split_token_logprobs`).
6. ~~`thought_logprobs`/`code_logprobs` are tracked but not yet used for pruning~~ -- **superseded
   by decision 7.**
7. (2026-09-15) `thought_logprobs`/`code_logprobs` now ARE folded into pruning and winner
   selection, via `Beam.pruning_score`: for each step, `log(p_yes) + thought_logprob +
   code_logprob` -- the log joint `P(thought) * P(code | thought) * P(judge=yes | thought, code)`
   by the chain rule (each term already conditioned on what came before it in the same step, not
   an independence assumption). Combined per step, then averaged across steps (not summed), so
   one bad step doesn't compound with the rest -- same reasoning as `score` averaging instead of
   summing. `p_yes` is floored at `1e-4` before the log so a zero-information judge fallback
   doesn't send the whole thing to -inf. `Beam.score` (judge-only mean) is kept as a separate,
   unused-for-ranking diagnostic property.

**Next step**: piece (e) from the original plan -- one live smoke test on a single GAIA question
at `beam_width=2, branch_factor=2, max_steps=20`, then the agreed 5-question dry run (manually
inspected), before anything resembling a full sweep.

Commit `484598a` ("Finalizing experimental results for further analysis") landed; working tree
clean at that point. Full test suite: 1113 passed, 66 skipped, 2 failed -- both confirmed
pre-existing/environmental (the documented `MockChoice.index` baseline failure, and
`test_import_smolagents_without_extras` shelling out to a `uv` binary not installed on this
machine; reproduced identically by stashing to HEAD before this session's commit).

## Runs in flight

**Launched 2026-09-14**: n=5 sweep of the new `judge_strategy="expectation"` (see the section
below) for the three models that support n=5, via `scratchpad/launch_expectation_judge.py`
(detached, `start_new_session`), `RUN_TAG="_exp5"`, logs in `scratchpad/expectation_judge/`:

| run | dir | pid |
|---|---|---:|
| gpt-4o | `tracelet_direct_n5_tp_exp5_react_gpt-4o` | 37892 |
| gpt-5.4-mini | `tracelet_direct_n5_tp_exp5_react_gpt-5.4-mini` | 37893 |
| Qwen3.5-9B | `tracelet_direct_n5_tp_exp5_react_Qwen/Qwen3.5-9B` | 37894 |

(Restarted 2026-09-14 at user request while all three were healthy and progressing — not a
recovery from a stall. Resumed cleanly from cache at 10/165, 15/165, 2/165.)

Directly comparable to `_v6n5` (scores judge) and `_rand5` (random control) — same model/n/
filter/max-steps, only the judge differs. Qwen3.7-Plus excluded (provider caps n at 4). When these
finish: pair each against `_v6n5` (accuracy McNemar + tokens/steps) the same way the random control
was compared — this is the real-accuracy test of whether the "expectation" judge's much better
ESS/n discrimination (see below) translates into better GAIA scores, not just a cleaner diagnostic.

Older: the three n=5 random-selection sweeps below finished 2026-09-12/13 (one needed a restart —
mini and Qwen3.5-9B had each accumulated a large contiguous block of cached `Connection error`
pickles from a long outage, quarantined to `_backup_errors_20260912/` and recomputed cleanly on
relaunch). Everything else — naive/n=1/n=3/n=5 filter-on baselines for all four models,
goal-pruning for all four models — is complete at 165/165. Residual, not-yet-cleaned-up error
counts on old runs: naive Qwen3.7-Plus 3, n=1 Qwen3.7-Plus 1, n=3 Qwen3.7-Plus 1, n=5 Qwen3.5-9B 1,
goal-pruning Qwen3.7-Plus 5 — likely the model's known reproducible content-filter 400s rather than
transient bursts; check before quarantining.

**Random-selection control result (the decisive test from below), n=5, vs `_v6n5`:**

| model | judge accuracy | random accuracy | McNemar p | judge tokens/q | random tokens/q |
|---|---|---|---:|---:|---:|
| gpt-4o | 55/165 = 33.3% | 40/165 = 24.2% | **0.014** | 267,970 | 274,973 |
| gpt-5.4-mini | 58/165 = 35.2% | 63/165 = 38.2% | 0.533 | 156,805 | 179,450 |
| Qwen3.5-9B | 79/165 = 47.9% | 80/165 = 48.5% | 1.000 | 601,707 | 607,643 |

Judge beats random significantly only for gpt-4o; for mini and Qwen3.5-9B, judge selection is
statistically indistinguishable from picking uniformly at random. Consistent across all three:
judge selection uses fewer tokens/steps than random regardless of the accuracy outcome, so it may
be doing something for efficiency even where it isn't for correctness.

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

## Beam search + probabilistic-program trace format (designed 2026-09-13/14, not implemented)

Prompted by the random-selection control above: since the judge barely beats chance for 2 of 3
models, the natural next question is whether *search* (not just per-step selection) is where the
value would come from. Sketched but deliberately not built yet — parked here per the established
"sideline it" pattern, same as the redundancy-pruning judge below.

**Why beam search, motivated by data, not just architecture appeal:** two uniqueness experiments
this session (`traceletReAct_single.ipynb`, 5 independent live re-runs of the same GAIA question)
showed diversity lives almost entirely *across* full trajectories, not *within* one step's n
fill-in candidates:
- n=5 fill-ins on a fixed skeleton: 45-85% of steps have >=1 duplicate candidate (measured
  earlier, `notes.ipynb` cells `9ce8fe10`/`fb9378d9`).
- 5 fully independent trajectories on Q112 (gpt-5.4-mini via TraceletCodeAgent): **0/70 and 0/50
  triplets repeated across two separate 5-run batches** — full-trajectory resampling gets
  essentially zero-repeat diversity, because a free-text "thought" at the API default temperature
  (1.0 — `_sample_skeleton_direct_prompt`/`_post_process` pass no explicit `temperature`) almost
  never reproduces verbatim, unlike a short constrained fill-in value.
- Dropping thought from the key (action+fill-in only) recovers a *few* genuine repeats — 3/50 and
  6/70 — driven by convergence at `final_answer(...)` calls (a small discrete space of plausible
  guesses), not by search-query convergence as originally guessed.
- **naiveReAct vs TraceletCodeAgent comparison, same question, same model:** naiveReAct showed
  *more* duplication (94.1%/70.9% unique on Q112/Q1) than tracelet's 100%/100% — but the mechanism
  was a genuine bug, not convergent reasoning: gpt-5.4-mini running plain `CodeAgent` (default
  `code_agent.yaml`, no tracelet protocol) sometimes returns a **completely empty completion**
  (`model_output==""`, `error=None`, `observation="...Last output...None"`) for several consecutive
  steps — happened in 4/5 independent Q1 runs, never in any tracelet run. Root cause unconfirmed
  (plausibly the model spending its turn on hidden reasoning tokens with nothing left for visible
  output); the tracelet protocol prompt appears to incidentally suppress this. Qwen3.5-9B showed no
  such artifact on the same question (4/5 correct vs mini's 2/5).

**Beam search sketch:** a `Beam` = partial trajectory (history, a *persisted* tool-state snapshot,
cumulative score). At each depth, every beam branches into `branch_factor` full step resamples
(thought+skeleton+fill-in, not just fill-in on one fixed skeleton — diversity needs the whole step
resampled, per the finding above), each is scored (`combine(prior_logprob, likelihood_score)`,
any of the 9 rules from `notes.ipynb` `e7abb006`), and only the top `beam_width` survive. This
generalizes today's per-step candidate-snapshot machinery (`TraceletCodeAgent._execute_candidates`)
to persist across *multiple* steps rather than being scored and discarded within one — the real
engineering cost is `beam_width` separate live tool states (browser position, etc.) carried
forward, not just `n_samples` transient ones.

**Framed as inference over a probabilistic program:** `TraceletCodeAgent` already defines the
generative model `∏_t P_prior(action_t) · P_likelihood(evidence_t)`; beam search is truncated
Viterbi over that joint (keep top-B partial-sequence scores per depth, discard the rest). Full
best-of-n (today's implementation) is the degenerate depth-1 case. Known gap: greedy top-B pruning
is not a properly-weighted particle filter — a beam cut at step 3 for scoring slightly lower might
have led to the best step 7, and there is no resampling-with-replacement to give it a chance the
way real SMC would.

**Custom-DSL trace format (user's choice over an existing PPL library):** compile the *search
tree*, not just the winning path, since that is the part a flat trajectory pickle can't already
represent:
```python
Choice(id, step, beam_id, parent_beam_id, thought, action, fillin, prior_logprob)
Evidence(choice_id, observation, likelihood_score)
Prune(step, kept_beam_ids, discarded_beam_ids, cumulative_scores)
Trace(question, choices, evidence, prunes, winner_beam_id)
```
Following `parent_beam_id` back to the root lets any beam's score be recomputed offline under a
*different* combination rule (any of the 9), purely by replaying `choices`/`evidence` — no LLM
calls, same replay-only pattern as `traceletReAct_trajectory.ipynb`. Worked a full 3-step toy
example by hand (Q112) showing prior-alone pruning would have kept a dead-end IMDb-error branch
over the branch that actually confirmed the correct answer via an independent path — concrete
evidence that judge-informed and prior-only search diverge on *which branch survives*, not just on
final accuracy. Hard limitation: a `Prune` record only holds the score *at the moment of
discarding* — it cannot answer "what would this branch have scored at step 10," since it was never
executed past step 3.

### Likelihood-source investigation (2026-09-14) — two self-assessment methods tested, both rejected

Before wiring any scoring rule into a beam, tested whether an LLM can usefully self-assess
multi-step trajectory success, offline, against 85 already-completed `naive_react_gpt-5.4-mini_False_filteron`
prefixes (30 questions, stratified 15 correct/15 incorrect, cut at ~25%/60%/90% of each
trajectory's length, ground truth never shown to the judge). Scripts in
`scratchpad/trajectory_success_calibration.py` (logprob) and `..._verbalized.py` (direct
verbalization); raw results in `baseline/trajectory_success_calibration.json` and
`..._verbalized.json`.

| method | corr(p, is_correct) | Brier (0.25 = always predict 0.5) |
|---|---:|---:|
| P(yes) from logprobs (`completion_judge.py`-style) | 0.085 | **0.407** — worse than the constant baseline |
| direct verbalized percentage | 0.151 | 0.283 — close to, still not beating, baseline |

Both share the decision-relevant failure: confidence rises monotonically with more of the
trajectory shown while real accuracy stays flat (~50%) or ticks down — e.g. logprob method's Brier
gets *worse* going from 25% (0.367) to 90% (0.452) shown. Reads as the model inferring "looks
complete/confident" rather than verifying correctness, which is exactly the regime beam-search
pruning lives in (comparing partial paths as they grow). **Both methods rejected as a likelihood
source; do not wire either into beam search.**

Side-finding, orthogonal to the calibration failure: logprob-extracted P(yes) values are not
continuous — every one of the 85 values decomposes exactly as `sigmoid(k * 0.25)` for integer k
(confirmed by inverting each value back to its logit and checking it lands on the 0.25 grid with no
exceptions). OpenAI's returned logprobs for this kind of binary read are quantized to ~0.25-nat
increments, capping resolution at a couple dozen distinguishable confidence levels regardless of
prompt/task quality — likely explains part of why the original per-step judge's 0-10 scores looked
clustered/discrete rather than smooth (see the score-distribution histograms in `notes.ipynb`).
Verbalized percentages have their own, different clustering problem instead: heavy weight on a
handful of "favorite" numbers (78 chosen 17/85 times, 62 chosen 11/85, 72 chosen 10/85) rather than
smooth variation — trading the API's quantization grid for the model's own round-number bias.

**Recommended likelihood source instead, not yet built:** cross-beam agreement (do independent
branches converge on the same conclusion — free once multiple beams exist, no extra LLM call, and
directly motivated by the worked toy example above where two independently-searched branches
corroborating each other was a stronger and cheaper signal than either one's judge score) combined
with a hard objective-execution floor (never let an errored candidate beat a clean one — closes the
one unambiguous failure already measured in the judge-effectiveness diagnostic, 0.53% of steps).

### A third self-assessment prompt tried after the two above — this one looks genuinely promising

Third variant, closer to `completion_judge.py`'s framing but reading the expectation out of the
free-form thought instead of requiring a structured `Expectation:` line: **"Does the output of
this step achieve what the model expected in the thought?"**, yes/no + logprob extraction, judged
per step (not a multi-step forecast like the two rejected methods above). Scripts in
`scratchpad/step_judge_thought_expectation.py` (per-step calibration, 213 steps from the same
30-question `naive_react_gpt-5.4-mini_False_filteron` sample) and `..._ess_headtohead.py` (applied
per-candidate to `tracelet_direct_n5_tp_v6n5_react_gpt-5.4-mini`, to compare ESS/n directly against
the production judge).

| check | result |
|---|---|
| corr(mean per-step P(yes) over a trajectory, trajectory `is_correct`) | **0.528** (vs 0.085 and 0.151 for the two rejected trajectory-forecast methods), Brier 0.265 |
| mean P(yes), errored steps vs clean steps | 0.028 vs 0.224 — correctly downgrades errors |
| mean P(yes), steps in eventually-correct vs -incorrect trajectories | 0.271 vs 0.131 |
| objective failures (errored step judged P(yes)>0.5) | 1/36 (2.8%) |

Caveat: it's a strict judge (139/213 = 65% of all judgments are exactly 0.0) — most individual
ReAct steps are exploratory and genuinely don't "achieve what was expected" in isolation, so the
raw score is a useful *relative ranking* signal, not a calibrated absolute probability.

**Head-to-head ESS/n vs the production judge** (`scratchpad/new_judge_ess_headtohead.py`, 30
multi-candidate steps from `tracelet_direct_n5_tp_v6n5_react_gpt-5.4-mini`, same per-step prompt
applied to each candidate against its shared thought, raw results in
`baseline/new_judge_ess_headtohead.json`):

| metric | production judge (n=4,867) | new judge (n=30) |
|---|---:|---:|
| mean ESS/n | 0.90-0.94 | **0.791** |
| near-uniform (ESS/n >= 0.95) | 74.3% | **63.3%** |
| real discrimination (ESS/n < 0.5) | 3.9% | **20.0%** (5x) |
| objective failures (errored candidate preferred) | 0.53% | **0.00%** (0/30) |

Wins on every metric. One nuance: many of this judge's "near-uniform" cases are literally all n
candidates scoring `p_yes=0` -- a defensible "none of these achieved it" verdict with no way to
rank equally-bad options under strict yes/no, not the production judge's failure mode of giving
everything a similar *mediocre* score. Sample is small (30 vs 4,867 steps) so exact percentages
would tighten with more data, but the direction agrees with the 0.528 trajectory-level correlation
above -- **this is the most promising judge design found this session; a real candidate to replace
the current 0-10 `judge_strategy="scores"`**, not just another rejected self-assessment attempt.

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

1. ~~**Random-selection control.**~~ Done — see "Runs in flight" above. Judge beats random only
   for gpt-4o (p=0.014); mini and Qwen3.5-9B show no detectable benefit from judge selection over
   chance at n=5.
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