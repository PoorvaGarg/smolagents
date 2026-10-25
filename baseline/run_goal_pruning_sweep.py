"""Full GAIA validation sweep for GoalPruningCodeAgent, standalone equivalent of
goalPruningReAct.ipynb's setup+eval cells (0-6) -- lets a sweep run detached in the background
instead of needing a live notebook kernel. Diagnostics/comparison cells (8-12) are intentionally
left out here; rerun those from the notebook against the resulting pickle_dir when this finishes.

Run:
    SMOLAGENTS_MODEL=gpt-4o SMOLAGENTS_RUN_TAG=_gp2 python run_goal_pruning_sweep.py
"""

import os
import sys

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(BASELINE_DIR)
    sys.path.insert(0, BASELINE_DIR)
    sys.path.insert(0, os.path.join(os.path.dirname(BASELINE_DIR), "examples", "open_deep_research"))

    MODEL = os.getenv("SMOLAGENTS_MODEL", "gpt-4o")
    N_QUESTIONS = int(os.getenv("SMOLAGENTS_N_QUESTIONS", "165"))
    MAX_STEPS = int(os.getenv("SMOLAGENTS_MAX_STEPS", "50"))
    COMPLETION_THRESHOLD = float(os.getenv("SMOLAGENTS_COMPLETION_THRESHOLD", "0.5"))
    JUDGE_ON = os.getenv("SMOLAGENTS_JUDGE_ON", "1") == "1"
    RUN_TAG = os.getenv("SMOLAGENTS_RUN_TAG", "_gp1")
    SEARCH_RPS = float(os.getenv("SMOLAGENTS_SEARCH_RPS", "0.3"))

    MODELS = {
        "gpt-4o": "openai",
        "gpt-5.4-mini": "openai",
        "Qwen/Qwen3.5-9B": "together",
        "Qwen/Qwen3.7-Plus": "together",
    }
    assert MODEL in MODELS, f"unknown MODEL {MODEL!r}; pick one of {list(MODELS)}"

    RUN_NAME = f"goalprune{RUN_TAG}{'' if JUDGE_ON else '_nojudge'}_react_{MODEL}"
    os.environ["SMOLAGENTS_DOWNLOADS_DIR"] = f"downloads_{RUN_NAME.replace('/', '_')}"
    os.environ["SMOLAGENTS_SEARCH_RPS"] = str(SEARCH_RPS)

    print(f"run   : {RUN_NAME}")
    print(f"judge : {'on' if JUDGE_ON else 'OFF (control)'} | threshold={COMPLETION_THRESHOLD}")

    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(BASELINE_DIR), ".env"))

    from smolagents import OpenAIModel
    from smolagents.context_pruning.goal_agent import GoalPruningCodeAgent
    from smolagents.monitoring import LogLevel

    from common_setup import build_tools, evaluate_agent, load_gaia_dataset

    STREAM_ONLY = {"Qwen/Qwen3.7-Plus"}
    if MODELS[MODEL] == "openai":
        model = OpenAIModel(model_id=MODEL, api_key=os.environ["OPENAI_API_KEY"])
    else:
        model = OpenAIModel(
            model_id=MODEL,
            api_base="https://api.together.ai/v1/",
            api_key=os.environ["TOGETHER_API_KEY"],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            client_kwargs={"timeout": 300.0},
        )
    stream_outputs = MODEL in STREAM_ONLY
    print(f"{MODEL} via {MODELS[MODEL]} | stream_outputs={stream_outputs}")

    tools, ti_tool, visualizer = build_tools(model)

    agent = GoalPruningCodeAgent(
        tools=tools,
        model=model,
        max_steps=MAX_STEPS,
        verbosity_level=LogLevel.ERROR,
        additional_authorized_imports=["pandas", "numpy", "PIL", "json", "io", "zipfile", "csv", "openpyxl"],
        completion_threshold=COMPLETION_THRESHOLD,
        judge_completion_calls=JUDGE_ON,
        stream_outputs=stream_outputs,
    )

    assert "Expectation:" in agent.system_prompt, "system prompt is missing the Expectation protocol"
    print(f"system prompt: {len(agent.system_prompt)} chars, {agent.system_prompt.count('Expectation:')} Expectation mentions")

    eval_ds = load_gaia_dataset(set_to_run="validation")
    print(f"Loaded {len(eval_ds)} examples")

    results = evaluate_agent(
        agent,
        eval_ds,
        ti_tool,
        visualizer,
        n_samples=N_QUESTIONS,
        output_file=f"{RUN_NAME}.jsonl",
        pickle_dir=RUN_NAME,
    )

    n_correct = sum(r["is_correct"] for r in results)
    print(f"\nDONE: {n_correct}/{len(results)} correct (see {RUN_NAME}.jsonl and {RUN_NAME}/)")


if __name__ == "__main__":
    main()
