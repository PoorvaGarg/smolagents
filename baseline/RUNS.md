# GAIA run manifest

What produced each result directory. Written because run directories only encode part of the
config: `MAX_STEPS`, question count, and the environment variables that gate the leak filter
and the search rate are not in the name, and were never in the notebooks either — they were set
on the command line. Runs whose numbers appear in `notes.ipynb` are listed here; older
exploratory directories are summarised at the bottom.

All runs below: GAIA `validation`, `max_steps=50`, `enable_thinking=False` for Together-served
models, launched headless with
`jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1`.

**Streaming changed for Qwen3.5-9B on 2026-09-03.** Every run in the tables below used
`stream_outputs=True` for both Together models, on the belief that Together rejects non-streaming
outright. That is true only of Qwen3.7-Plus; Qwen3.5-9B accepts non-streaming with `n=3` and
reports fill-in logprobs, which the streaming path drops. The tracelet notebooks now set
`stream_outputs = model_name in {"Qwen/Qwen3.7-Plus"}`, so **new Qwen3.5-9B runs take the
non-streaming code path and are not strictly config-identical to the ones recorded here.**

Environment variables (added to `common_setup.build_tools` on 2026-09-01/02):

| var | default | meaning |
|---|---|---|
| `SMOLAGENTS_FILTER_LEAKS` | `1` (on) | drop `web_search` results containing GAIA answer keys (`benchmark_leak_filter.py`) |
| `SMOLAGENTS_SEARCH_RPS` | `1.0` | DuckDuckGo queries/sec per run; lower it when sweeps run in parallel |
| `SMOLAGENTS_DOWNLOADS_DIR` | `downloads_folder` | per-run downloads dir, so parallel sweeps don't collide |
| `SMOLAGENTS_LEAK_LOG` | `<pickle_dir>_leakdrops.jsonl` | where the filter records dropped results; set it only to override the default |

From 2026-09-03 a filter-on run also writes `<pickle_dir>_leakdrops.jsonl`, one JSON record per search
that lost results: `question_idx`, `query`, `n_results`, `kept`, `all_dropped`, and the dropped
URLs with the pattern that matched. Runs before that date have no such file — the filter only
printed to stdout, which nbconvert never saved, so **how often it fired in `_var1/2/3` and the n=3
runs is unrecoverable**. Join `question_idx` against each pickle's `tool_usage['web_search']` for a
per-question drop rate.

## Current baselines — naive CodeAgent, 165 questions

| directory | model | leak filter | search rps | date |
|---|---|---|---|---|
| `naive_react_gpt-4o_False_fixedbrowser` | gpt-4o | **off** | 1.0 | 2026-08-30/31 |
| `naive_react_gpt-5.4-mini_False_fixedbrowser` | gpt-5.4-mini | **off** | 1.0 | 2026-08-27 |
| `naive_react_Qwen/Qwen3.5-9B_False_fixedbrowser` | Qwen/Qwen3.5-9B | **off** | 1.0 | 2026-08-31 |
| `naive_react_Qwen/Qwen3.7-Plus_False_fixedbrowser` | Qwen/Qwen3.7-Plus | **off** | 1.0 | 2026-09-01 |

All four are post-`mdconvert` fix (browser returns markdown, not raw HTML). Errors were recovered
by deleting the errored pickle and re-running; originals kept under `_backup_errors/` and
`_backup_errors_2/`. `naive_react_Qwen/Qwen3.7-Plus_False_fixedbrowser` idx 164 still errors
(Together content filter, reproduced three times) — report it as an error, not a wrong answer.

## Variance repeats — naive gpt-4o × 3, 165 questions

| directory | leak filter | search rps | date | accuracy |
|---|---|---|---|---|
| `naive_react_gpt-4o_False_var1` | **on** | 0.15 | 2026-09-02 | 39/165 = 23.6% |
| `naive_react_gpt-4o_False_var2` | **on** | 0.15 | 2026-09-02 | 46/165 = 27.9% |
| `naive_react_gpt-4o_False_var3` | **on** | 0.15 | 2026-09-02 | 39/165 = 23.6% |

Purpose: measure run-to-run noise. Mean 25.1%, sd 2.4pp. **Not config-identical to
`_fixedbrowser` above** (leak filter and search rate differ), so the gap to that run's 33.9% is
confounded and cannot be attributed to noise alone.

## Probabilistic ReAct (TraceletCodeAgent) — `direct_prompt`, tracelet prompt template

`RUN_TAG` values: `_v3` = pre-`mdconvert`-fix (broken browser); `_v5` = fixed browser plus the
fill-in-sees-thought change; `_v5n3` = `_v5` plus `N_SAMPLES=3`.

Note `N_TAG` is empty when `N_SAMPLES==3`, which is why the n=3 directories read
`tracelet_direct_tp_...` with no `n1`.

| directory | model | n | leak filter | search rps | questions | date |
|---|---|---|---|---|---|---|
| `tracelet_direct_n1_tp_v5_react_gpt-4o` | gpt-4o | 1 | off | 1.0 | 165 | 2026-08-30/31 |
| `tracelet_direct_n1_tp_v5_react_gpt-5.4-mini` | gpt-5.4-mini | 1 | off | 1.0 | 165 | 2026-08-27..31 |
| `tracelet_direct_n1_tp_v5_react_Qwen/Qwen3.5-9B` | Qwen3.5-9B | 1 | off | 1.0 | 165 | 2026-08-30 |
| `tracelet_direct_n1_tp_v5_react_Qwen/Qwen3.7-Plus` | Qwen3.7-Plus | 1 | off | 1.0 | 165 | 2026-09-01 |
| `tracelet_direct_tp_v5n3_react_gpt-4o` | gpt-4o | **3** | **on** | 0.15 | 165 | 2026-09-02/03 |
| `tracelet_direct_tp_v5n3_react_gpt-5.4-mini` | gpt-5.4-mini | **3** | **on** | 0.15 | 165 | 2026-09-02 |
| `tracelet_direct_tp_v5n3_react_Qwen/Qwen3.5-9B` | Qwen3.5-9B | **3** | **on** | 0.15 | **134/165, stopped early** | 2026-09-02/03 |
| `tracelet_direct_tp_v5n3_react_Qwen/Qwen3.7-Plus` | Qwen3.7-Plus | **3** | **on** | 0.15 | **60/165, stopped early** | 2026-09-03 |

The n=3 runs also carry two fixes the n=1 runs do not: the browser is snapshotted per trial
candidate (`SimpleTextBrowser.get_state`/`set_state`), and the leak filter is on. **So n=1 vs n=3
is not a clean single-variable comparison.**

`tracelet_direct_tp_v5n3_react_Qwen/Qwen3.5-9B` was first run to 50 questions and later extended;
its indices 47-49 errored transiently and were re-run (originals in `_backup_errors/`).

## One-off diagnostics

| directory | what it was for |
|---|---|
| `tracelet_direct_n1_tp_reasoningprobe_react_gpt-5.4-mini` | 10 questions with `Completions.create` wrapped to log `reasoning_tokens` per call — showed 0 on all 147 calls. Notebook: `traceletReAct_reasoningprobe.ipynb` |
| `tracelet_direct_n1_tp_v4_react_gpt-5.4-mini` | 50 questions, fill-in-sees-thought change measured alone |
| `tracelet_direct_n1_tp_try*`, `tracelet_direct_tp_try_*`, `*_rerun`, `*_try*` | short interactive shakedown runs, not results |
| `naive_react_gpt-5.4-mini_False_rule13` | 50 questions, rule 13 injected into naive as a control (14/50 vs naive's 17/50) |

## Older directories

Everything dated before 2026-08-24 (`tracelet_react_*`, `tracelet_direct_react_*`,
`tracelet_postprocess_*`, `naive_react_*_False`, `naive_react_Qwen/*_False`) predates this
manifest and the `mdconvert` fix, so those runs read raw HTML from every web page. Their exact
config is not recorded. Do not mix them with the runs above.

## Reproducing a run

The parameterised notebooks are `naiveReAct.ipynb` (set `model_name`) and `traceletReAct.ipynb`
(set the config block: `MODEL`, `SKELETON_STRATEGY`, `N_SAMPLES`, `USE_TRACELET_PROMPT`,
`N_QUESTIONS`, `MAX_STEPS`, `RUN_TAG`). Parallel sweeps need a distinct
`SMOLAGENTS_DOWNLOADS_DIR` each and a reduced `SMOLAGENTS_SEARCH_RPS` (~1.0 divided by the number
of concurrent runs) to avoid collectively tripping DuckDuckGo throttling.