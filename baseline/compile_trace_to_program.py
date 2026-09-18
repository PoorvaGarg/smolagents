"""Compile a BeamSearchCodeAgent trace (the round-by-round tree JSON saved by
run_beam_search_sweep.py / the dry-run scripts) into the minimal flip/if-then-else/factor/
resample_top_k probabilistic-program notation worked out by hand in TODO.md.
"""

import argparse
import json
import math
import pickle

MIN_PROB = 1e-4


def theta(children):
    """Stick-breaking sequence of normalized flip probabilities, for any number of children:
    theta[i] = weight of children[i] renormalized over children[i:] (whichever candidates
    haven't been "claimed" by an earlier flip yet). One fewer value than len(children) -- the
    last candidate needs no flip, it's forced by elimination. For branch_factor=2 this is a
    single value, the same P(children[0]) vs children[1] as before."""
    weights = [math.exp(c["thought_logprob"]) for c in children]
    return [weights[i] / sum(weights[i:]) for i in range(len(children) - 1)]


def render_branch_point(parent_id, children):
    """One parent's expansion into any number of children, as program text: a stick-breaking
    sequence of flips so exactly one child ends up realized, then each child's
    code/obs/factor lines. For branch_factor=2 this is exactly flip/not."""
    thetas = theta(children)
    thought_vars = [f"thought{c['beam_id']}" for c in children]

    lines = []
    prior_vars = []
    for i, t in enumerate(thetas):
        theta_name = f"theta_{parent_id}_{i}"
        lines.append(f"{theta_name} = {t:.3f}")
        if not prior_vars:
            lines.append(f"{thought_vars[i]} = flip({theta_name})")
        else:
            not_prior = " and ".join(f"not {v}" for v in prior_vars)
            lines.append(f"{thought_vars[i]} = ({not_prior}) and flip({theta_name})")
        prior_vars.append(thought_vars[i])
    lines.append(f"{thought_vars[-1]} = " + " and ".join(f"not {v}" for v in prior_vars))

    for child, thought_var in zip(children, thought_vars):
        code_var = f"code{child['beam_id']}"
        if child["code"]:
            lines.append(f"{code_var} = if {thought_var} then {child['code']!r} else Error")
        else:
            lines.append(f"{code_var} = if {thought_var} then Error else Error")
        lines.append(f"obs{child['beam_id']} = execute({code_var})")
        p_yes = max(child["p_yes"], MIN_PROB)
        lines.append(f"factor({math.log(p_yes):.3f})   # log({child['p_yes']:.6f})")
    return "\n".join(lines)


def render_round(round_data, prev_kept):
    """One round: every parent's branch point, then the resample_top_k line covering the
    full competing pool -- new children plus any beam carried forward unexpanded because it
    had already reached final_answer in an earlier round (run() skips _expand_beam for those,
    but they still compete in this round's prune)."""
    expanded_parents = {exp["parent_id"] for exp in round_data["expansions"]}
    carried_forward = [b for b in prev_kept if b not in expanded_parents]

    lines = []
    pool = list(carried_forward)
    for exp in round_data["expansions"]:
        lines.append(render_branch_point(exp["parent_id"], exp["children"]))
        pool.extend(c["beam_id"] for c in exp["children"])

    pool_str = "{" + ", ".join(pool) + "}"
    k = len(round_data["kept"])
    lines.append(f"{pool_str} = resample_top_k({pool_str}, k={k})   # kept: {round_data['kept']}")
    return "\n\n".join(lines)


def find_winner(rounds):
    """Replicates run()'s own winner-selection rule (finished beats unfinished, tiebreak by
    pruning_score) against the trace's actual last-round survivors -- same logic as the
    tree-view cell in beamSearchReAct_single.ipynb."""
    nodes = {}
    for r in rounds:
        for exp in r["expansions"]:
            for c in exp["children"]:
                nodes[c["beam_id"]] = c
    frontier = [nodes[b] for b in rounds[-1]["kept"]]
    finished = [n for n in frontier if n["is_final"]]
    return max(finished if finished else frontier, key=lambda n: n["pruning_score"])


def compile_trace(rounds, question=None, true_answer=None, prediction=None):
    """Top-level entry point: the full program text for one question's trace, all rounds
    joined, with a header (if question/true_answer given) and a winner summary footer."""
    lines = []
    if question is not None:
        lines.append(f"# question: {question}")
    if true_answer is not None:
        lines.append(f"# true_answer: {true_answer}")
    if lines:
        lines.append("")

    prev_kept = ["0"]
    for r in rounds:
        lines.append(f"# ============ ROUND {r['n']} ============")
        lines.append(render_round(r, prev_kept))
        lines.append("")
        prev_kept = r["kept"]

    winner = find_winner(rounds)
    lines.append(f"winner = {winner['beam_id']}   # pruning_score={winner['pruning_score']:.3f}")
    if prediction is not None:
        lines.append(f"answer = {prediction!r}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", help='e.g. "beamsearch_bw2_bf2_react_gpt-4o"')
    parser.add_argument("index", type=int, help="question index within that run")
    args = parser.parse_args()

    with open(f"{args.run_name}_trees/{args.index}.json") as f:
        rounds = json.load(f)

    question = true_answer = prediction = None
    pkl_path = f"{args.run_name}/{args.index}.pkl"
    try:
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
        question, true_answer, prediction = result["question"], result["true_answer"], result["prediction"]
    except FileNotFoundError:
        pass  # tree-only data (e.g. a dry-run file) -- compile without the header/answer

    print(compile_trace(rounds, question, true_answer, prediction))


if __name__ == "__main__":
    main()
