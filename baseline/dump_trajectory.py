"""Dump a readable per-step trajectory from a run's pickles.

Usage:
    python dump_trajectory.py <run_dir> [indices ...]      # default: every pkl present
    python dump_trajectory.py tracelet_direct_tp_v5n3_react_Qwen/Qwen3.5-9B 0 1 2

Writes <run_dir>/q<i>_trajectory.txt per index and prints a one-line summary each.
For n>1 runs, `cands=3` marks a step where the fill-in sampled candidates and the judge
chose; only the winner is stored in the pickle, so losing candidates are not shown.
"""

import pickle
import re
import sys
import textwrap
from pathlib import Path

POSITIONAL = ("find_on_page_ctrl_f", "page_down", "page_up", "find_next")


def dump(pkl_dir: Path, index: int) -> str:
    with open(pkl_dir / f"{index}.pkl", "rb") as f:
        r = pickle.load(f)
    steps = r.get("steps") or []
    tps = r.get("tokens_per_step") or []
    out = [
        "=" * 100,
        f"QUESTION {index}   {pkl_dir}",
        "=" * 100,
        "TASK:\n" + textwrap.fill(str(r.get("question"))[:1500], 100),
        f"\ntrue answer : {r.get('true_answer')!r}",
        f"prediction  : {str(r.get('prediction'))[:200]!r}",
        f"correct={r.get('is_correct')}  steps={r.get('num_steps')}  "
        f"tokens={r['token_counts']['total_tokens']:,}  error={str(r.get('error'))[:80]}",
        "",
    ]
    first_seen: dict[str, int] = {}
    for k, st in enumerate(steps):
        code = str(st.get("code_action") or "").strip()
        sig = re.sub(r"\s+", " ", code)
        dup = f"   <-- IDENTICAL CODE TO STEP {first_seen[sig]}" if code and sig in first_seen else ""
        if code:
            first_seen.setdefault(sig, k)
        sc = st.get("sentinel_count")
        cands = f"cands={'3' if sc else '1 (direct, no judge)'}" if sc is not None else ""
        tu = tps[k] if k < len(tps) else None
        m = re.search(r"Thought:(.*?)(?:<code>|$)", str(st.get("model_output") or ""), re.S)
        # Steps reading implicit browser position are the ones n>1 used to corrupt.
        pos = "  [uses browser position]" if any(re.search(rf"\b{t}\s*\(", code) for t in POSITIONAL) else ""
        out.append("-" * 100)
        head = f"STEP {k}"
        if tu:
            head += f"   in={tu['input_tokens']:,} out={tu['output_tokens']:,}"
        out.append(f"{head}   {cands}{pos}{dup}")
        if st.get("error"):
            out.append(f"  ERROR {st['error'].get('type')}: {str(st['error'].get('message'))[:300]}")
        out.append(f"  THOUGHT: {(m.group(1).strip()[:300] if m else '') or '(none)'}")
        out.append(f"  CODE   : {code[:500] or '(none)'}")
        out.append(f"  OBS    : {str(st.get('observations') or '')[:500].strip() or '(empty)'}")
    text = "\n".join(out)
    path = pkl_dir / f"q{index}_trajectory.txt"
    path.write_text(text)
    dups = sum(1 for k, st in enumerate(steps) if st.get("code_action") and k != first_seen.get(re.sub(r"\s+", " ", str(st["code_action"]).strip()), k))
    print(
        f"  q{index}: {r.get('num_steps')} steps, {r['token_counts']['total_tokens']:,} tok, "
        f"correct={r.get('is_correct')}, {dups} repeated steps -> {path}"
    )
    return text


if __name__ == "__main__":
    d = Path(sys.argv[1])
    idxs = [int(a) for a in sys.argv[2:]] or sorted(int(p.stem) for p in d.glob("[0-9]*.pkl"))
    for i in idxs:
        dump(d, i)