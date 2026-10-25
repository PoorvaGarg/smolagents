"""Full GAIA validation sweep for BeamSearchCodeAgent, using the same evaluate_agent harness
(caching, resumability, leak-filter drop logging) as the tracelet/naive sweeps.

Accuracy is not the point of this run -- it's to produce full trajectories (winning-path steps
via evaluate_agent's pickles, plus every candidate including pruned ones via the round-by-round
JSON trees this script also saves) for compiling into probabilistic programs later.

Everything with a side effect (model/tool construction, the dataset load, the sweep itself) is
gated behind `if __name__ == "__main__"` -- importing this module (e.g. to reuse
_TreeSavingWrapper in a test) must not start a sweep.

Run:
    python run_beam_search_sweep.py
"""

import json
import os
import sys

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def _record_expand(orig_expand_beam, current_round_expansions):
    def _wrapped(self, beam):
        children = orig_expand_beam(self, beam)
        child_records = []
        for c in children:
            step = c.memory.steps[-1]
            thought = (step.model_output or "").split(self.code_block_tags[0], 1)[0].strip()
            child_records.append(
                {
                    "beam_id": c.beam_id,
                    "errored": c.errored_this_step,
                    "is_final": c.is_final,
                    "p_yes": c.expectation_scores[-1] if c.expectation_scores else None,
                    "thought_logprob": c.thought_logprobs[-1] if c.thought_logprobs else None,
                    "code_logprob": c.code_logprobs[-1] if c.code_logprobs else None,
                    "pruning_score": c.pruning_score,
                    "thought": thought,
                    "code": step.code_action or "",
                    "observation": step.observations or "",
                }
            )
        current_round_expansions.append(
            {"parent_id": beam.beam_id, "parent_pruning_score": beam.pruning_score, "children": child_records}
        )
        return children

    return _wrapped


def _record_prune(orig_prune, current_round_expansions, current_rounds):
    def _wrapped(self, beams):
        survivors = orig_prune(self, beams)
        current_rounds.append(
            {
                "n": len(current_rounds) + 1,
                "expansions": list(current_round_expansions),
                "kept": [b.beam_id for b in survivors],
            }
        )
        current_round_expansions.clear()
        return survivors

    return _wrapped


class _TreeSavingWrapper:
    """Wraps agent.run so each question's tree is saved right after it. evaluate_agent skips
    calling agent.run() entirely for a question whose {pickle_dir}/{i}.pkl already exists, so a
    plain call counter here would drift out of sync with the real question index after any
    resumed run -- instead, replicate evaluate_agent's own skip logic against the same
    pickle_dir to recover the true index before each call."""

    def __init__(self, agent, pickle_dir, trees_dir, current_rounds):
        self.agent = agent
        self.pickle_dir = pickle_dir
        self.trees_dir = trees_dir
        self.current_rounds = current_rounds
        self.next_index = 0

    def run(self, *args, **kwargs):
        while os.path.exists(os.path.join(self.pickle_dir, f"{self.next_index}.pkl")):
            self.next_index += 1
        index = self.next_index
        self.next_index += 1

        self.current_rounds.clear()
        result = self.agent.run(*args, **kwargs)
        with open(os.path.join(self.trees_dir, f"{index}.json"), "w") as f:
            json.dump(self.current_rounds, f, indent=1)
        return result

    def __getattr__(self, name):
        return getattr(self.agent, name)


def main():
    os.chdir(BASELINE_DIR)
    sys.path.insert(0, BASELINE_DIR)

    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(BASELINE_DIR), ".env"))

    from smolagents import OpenAIModel
    from smolagents.beam_search_agent import BeamSearchCodeAgent
    from smolagents.monitoring import LogLevel

    from common_setup import build_tools, evaluate_agent, load_gaia_dataset

    model_name = os.getenv("SMOLAGENTS_MODEL", "gpt-4o")
    max_steps = int(os.getenv("SMOLAGENTS_MAX_STEPS", "20"))
    beam_width = int(os.getenv("SMOLAGENTS_BEAM_WIDTH", "2"))
    branch_factor = int(os.getenv("SMOLAGENTS_BRANCH_FACTOR", "2"))
    run_tag = os.getenv("SMOLAGENTS_RUN_TAG", "")

    run_name = f"beamsearch_bw{beam_width}_bf{branch_factor}{run_tag}_react_{model_name}"
    trees_dir = f"{run_name}_trees"
    os.makedirs(trees_dir, exist_ok=True)

    models = {
        "gpt-4o": "openai",
        "gpt-5.4-mini": "openai",
        "Qwen/Qwen3.5-9B": "together",
        "Qwen/Qwen3.7-Plus": "together",
    }
    provider = models.get(model_name, "openai")
    if provider == "openai":
        model = OpenAIModel(model_id=model_name, api_key=os.environ["OPENAI_API_KEY"])
    else:
        model = OpenAIModel(
            model_id=model_name,
            api_base="https://api.together.ai/v1/",
            api_key=os.environ["TOGETHER_API_KEY"],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            client_kwargs={"timeout": 300.0},
        )
    print(f"model: {model_name} via {provider}")
    tools, ti_tool, visualizer = build_tools(model)

    agent = BeamSearchCodeAgent(
        tools=tools,
        model=model,
        max_steps=max_steps,
        beam_width=beam_width,
        branch_factor=branch_factor,
        verbosity_level=LogLevel.ERROR,
        additional_authorized_imports=["pandas", "numpy", "PIL", "json", "io", "zipfile", "csv", "openpyxl"],
    )

    # Record every candidate (including pruned ones), per question, as a round-by-round tree --
    # evaluate_agent's own pickle only keeps the winning beam's steps. Same instrumentation as
    # the dry-run scripts and beamSearchReAct_single.ipynb.
    current_round_expansions = []
    current_rounds = []
    BeamSearchCodeAgent._expand_beam = _record_expand(BeamSearchCodeAgent._expand_beam, current_round_expansions)
    BeamSearchCodeAgent._prune = _record_prune(BeamSearchCodeAgent._prune, current_round_expansions, current_rounds)

    eval_ds = load_gaia_dataset(set_to_run="validation")
    print(f"run: {run_name}  ({len(eval_ds)} questions, beam_width={beam_width}, branch_factor={branch_factor})")

    wrapped_agent = _TreeSavingWrapper(agent, pickle_dir=run_name, trees_dir=trees_dir, current_rounds=current_rounds)

    results = evaluate_agent(
        wrapped_agent,
        eval_ds,
        ti_tool,
        visualizer,
        n_samples=None,
        output_file=f"{run_name}.jsonl",
        pickle_dir=run_name,
    )

    n_correct = sum(r["is_correct"] for r in results)
    print(f"\nDONE: {n_correct}/{len(results)} correct (accuracy incidental -- see {run_name}.jsonl and {trees_dir}/)")


if __name__ == "__main__":
    main()
