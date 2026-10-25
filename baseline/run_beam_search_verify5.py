"""Ad-hoc verification run: BeamSearchCodeAgent against the first 5 GAIA validation questions,
using gpt-4o. Mirrors run_beam_search_sweep.py's setup exactly, but bounded to n_samples=5 and
written to a separate run_tag ("_verify5") so it never touches the existing full-sweep cache
(beamsearch_bw2_bf2_react_gpt-4o*) already present in this directory.

Not part of the tracked sweep scripts -- a one-off smoke test.

Run:
    python run_beam_search_verify5.py
"""

import os
import sys

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(BASELINE_DIR)
    sys.path.insert(0, BASELINE_DIR)

    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(BASELINE_DIR), ".env"))

    from smolagents import OpenAIModel
    from smolagents.beam_search_agent import BeamSearchCodeAgent
    from smolagents.monitoring import LogLevel

    from common_setup import build_tools, evaluate_agent, load_gaia_dataset

    model_name = "gpt-4o"
    max_steps = int(os.getenv("SMOLAGENTS_MAX_STEPS", "20"))
    beam_width = int(os.getenv("SMOLAGENTS_BEAM_WIDTH", "2"))
    branch_factor = int(os.getenv("SMOLAGENTS_BRANCH_FACTOR", "2"))
    run_tag = "_verify5"

    run_name = f"beamsearch_bw{beam_width}_bf{branch_factor}{run_tag}_react_{model_name}"

    model = OpenAIModel(model_id=model_name, api_key=os.environ["OPENAI_API_KEY"])
    print(f"model: {model_name} via openai")
    tools, ti_tool, visualizer = build_tools(model)

    agent = BeamSearchCodeAgent(
        tools=tools,
        model=model,
        max_steps=max_steps,
        beam_width=beam_width,
        branch_factor=branch_factor,
        verbosity_level=LogLevel.INFO,
        additional_authorized_imports=["pandas", "numpy", "PIL", "json", "io", "zipfile", "csv", "openpyxl"],
    )

    eval_ds = load_gaia_dataset(set_to_run="validation")
    print(f"run: {run_name}  (5 of {len(eval_ds)} questions, beam_width={beam_width}, branch_factor={branch_factor})")

    results = evaluate_agent(
        agent,
        eval_ds,
        ti_tool,
        visualizer,
        n_samples=5,
        output_file=f"{run_name}.jsonl",
        pickle_dir=run_name,
    )

    n_correct = sum(r["is_correct"] for r in results)
    print(f"\nDONE: {n_correct}/{len(results)} correct (see {run_name}.jsonl)")


if __name__ == "__main__":
    main()
