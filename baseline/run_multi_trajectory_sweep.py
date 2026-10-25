"""Multi-trajectory sweep: N independent naive-ReAct (CodeAgent) trajectories per question, with
per-token logprobs, on one (model, dataset) pair per process.

Each of the N_TRAJ passes builds a fresh agent and tools and answers the first N_QUESTIONS
questions. evaluate_agent caches each pass in its own pickle dir, so a killed run resumes where it
stopped.

Independence and isolation from ground truth:
  * Executor state (Python variables) is reset before every question; run(reset=True) alone
    clears only memory.
  * The agent's working directory is an empty sandbox outside the repo, so no answer-bearing file
    (run outputs, dataset caches) is reachable by relative path. GAIA attachments are copied into
    the sandbox and passed as relative paths, so the agent never sees a repo path.
  * Leak filter v2 (search + visit + file inspection) is on via build_tools' default.

Run:
    SMOLAGENTS_MODEL=gpt-4o SMOLAGENTS_DATASET=gaia python run_multi_trajectory_sweep.py
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parent

MODELS = {
    "gpt-4o": "openai",
    "gpt-5.4-mini": "openai",
    "Qwen/Qwen3.5-9B": "together",
}
DATASETS = ("gaia", "fermi", "math500", "simpleqa")


def build_model(model_id: str, temperature: float):
    from smolagents import OpenAIModel

    # Model kwargs reach every API call; logprobs need a non-streaming model, hence no Qwen3.7-Plus.
    common = {"logprobs": True, "top_logprobs": 5, "temperature": temperature}
    if MODELS[model_id] == "openai":
        return OpenAIModel(model_id=model_id, api_key=os.environ["OPENAI_API_KEY"], **common)
    return OpenAIModel(
        model_id=model_id,
        api_base="https://api.together.ai/v1/",
        api_key=os.environ["TOGETHER_API_KEY"],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        client_kwargs={"timeout": 300.0},
        **common,
    )


def load_dataset_config(name: str):
    """(dataset, scorer, question_suffix) for one benchmark."""
    if name == "gaia":
        from common_setup import load_gaia_dataset, question_scorer

        return load_gaia_dataset(data_dir=str(BASELINE_DIR / "data" / "gaia")), question_scorer, None
    if name == "fermi":
        from fermi_setup import FERMI_ANSWER_INSTRUCTION, fermi_scorer, load_fermi_dataset

        return load_fermi_dataset(), fermi_scorer, FERMI_ANSWER_INSTRUCTION
    if name == "math500":
        from math500_setup import MATH500_ANSWER_INSTRUCTION, Math500Scorer, load_math500_dataset

        return load_math500_dataset(), Math500Scorer(), MATH500_ANSWER_INSTRUCTION
    if name == "simpleqa":
        from simpleqa_setup import SIMPLEQA_ANSWER_INSTRUCTION, SimpleQAGrader, load_simpleqa_dataset

        return load_simpleqa_dataset(), SimpleQAGrader(), SIMPLEQA_ANSWER_INSTRUCTION
    raise ValueError(f"unknown dataset {name!r}; pick one of {DATASETS}")


def copy_attachments_into_sandbox(ds, sandbox: Path):
    """Point file_name at a sandbox copy, relative to the sandbox, so no repo path is ever shown."""
    (sandbox / "attachments").mkdir(parents=True, exist_ok=True)

    def relocate(row):
        if row["file_name"]:
            src = Path(row["file_name"])
            rel = Path("attachments") / src.name
            if not (sandbox / rel).exists():
                shutil.copy2(src, sandbox / rel)
            row["file_name"] = str(rel)
        return row

    return ds.map(relocate)


def main():
    sys.path.insert(0, str(BASELINE_DIR))
    from dotenv import load_dotenv

    load_dotenv(BASELINE_DIR.parent / ".env")

    model_id = os.getenv("SMOLAGENTS_MODEL", "gpt-4o")
    dataset_name = os.getenv("SMOLAGENTS_DATASET", "gaia")
    n_questions = int(os.getenv("SMOLAGENTS_N_QUESTIONS", "20"))
    n_traj = int(os.getenv("SMOLAGENTS_N_TRAJ", "5"))
    max_steps = int(os.getenv("SMOLAGENTS_MAX_STEPS", "50"))
    temperature = float(os.getenv("SMOLAGENTS_TEMPERATURE", "1.0"))
    run_tag = os.getenv("SMOLAGENTS_RUN_TAG", "_mt2")
    # Search rate-limit sleeps run inside the agent's code, so the 30s smolagents default
    # times out ordinary multi-search code blocks when many sweeps share a low search rate.
    exec_timeout = int(os.getenv("SMOLAGENTS_EXEC_TIMEOUT", "300"))
    os.environ.setdefault("SMOLAGENTS_SEARCH_RPS", "0.1")
    assert model_id in MODELS, f"unknown model {model_id!r}; pick one of {list(MODELS)}"
    assert dataset_name in DATASETS, f"unknown dataset {dataset_name!r}; pick one of {DATASETS}"

    model_safe = model_id.replace("/", "_")
    out_dir = BASELINE_DIR / f"multitraj{run_tag}" / dataset_name / model_safe
    sandbox_root = Path(os.getenv("AGENT_SANDBOX_DIR", BASELINE_DIR.parents[1] / ".agent_sandbox"))
    sandbox = sandbox_root / f"multitraj{run_tag}" / dataset_name / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    (sandbox / "downloads").mkdir(parents=True, exist_ok=True)

    from benchmark_leak_filter import LEAK_FILTER_VERSION
    from common_setup import build_tools, evaluate_agent
    from smolagents import CodeAgent
    from smolagents.monitoring import LogLevel

    ds, scorer, suffix = load_dataset_config(dataset_name)
    ds = copy_attachments_into_sandbox(ds, sandbox)

    config = {
        "model": model_id, "dataset": dataset_name, "n_questions": n_questions, "n_traj": n_traj,
        "max_steps": max_steps, "temperature": temperature, "top_logprobs": 5,
        "search_rps": float(os.environ["SMOLAGENTS_SEARCH_RPS"]), "leak_filter_version": LEAK_FILTER_VERSION,
        "reset_executor_state": True, "exec_timeout_seconds": exec_timeout,
        "sandbox": str(sandbox), "launched": datetime.now().isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    print(f"run: {out_dir}\nsandbox (agent cwd): {sandbox}\n{json.dumps(config)}")

    # Everything below runs with the empty sandbox as cwd; all harness paths are absolute.
    os.chdir(sandbox)
    os.environ["SMOLAGENTS_DOWNLOADS_DIR"] = "downloads"
    model = build_model(model_id, temperature)

    for k in range(n_traj):
        print(f"\n===== trajectory {k + 1}/{n_traj} =====")
        tools, ti_tool, visualizer = build_tools(model)
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=max_steps,
            verbosity_level=LogLevel.ERROR,
            additional_authorized_imports=["pandas", "numpy", "PIL", "json", "io", "zipfile", "csv", "openpyxl"],
            executor_kwargs={"timeout_seconds": exec_timeout},
        )
        evaluate_agent(
            agent,
            ds,
            ti_tool,
            visualizer,
            n_samples=n_questions,
            output_file=str(out_dir / f"traj{k}.jsonl"),
            pickle_dir=str(out_dir / f"traj{k}"),
            scorer=scorer,
            question_suffix=suffix,
            capture_logprobs=True,
            reset_executor_state=True,
        )

    print(f"\nDONE: {n_traj} trajectories x {n_questions} questions in {out_dir}")


if __name__ == "__main__":
    main()
