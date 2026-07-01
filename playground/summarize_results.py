"""
Parse raw output from compare_agents.py and print per-agent totals.
Usage: python summarize_results.py <output_file>
       cat output.txt | python summarize_results.py
"""
import re
import sys
from collections import defaultdict

# Matches lines like:
#        CodeAgent             6     40,900        ✅  → Egalitarian
RESULT_RE = re.compile(
    r"^\s+(CodeAgent|PruningAgent|ProbAgent\(n=\d+\))\s+(\d+)\s+([\d,]+)\s+(✅|❌)"
)

def parse(lines):
    totals = defaultdict(lambda: {"steps": 0, "tokens": 0, "correct": 0, "total": 0})
    for line in lines:
        m = RESULT_RE.match(line)
        if not m:
            continue
        agent, steps, tokens, correct = m.groups()
        totals[agent]["steps"] += int(steps)
        totals[agent]["tokens"] += int(tokens.replace(",", ""))
        totals[agent]["correct"] += int(correct == "✅")
        totals[agent]["total"] += 1
    return totals

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    totals = parse(lines)

    if not totals:
        print("No result lines found.")
        return

    print(f"\n{'Agent':<20} {'Correct':>8} {'Steps':>8} {'Tokens':>14}")
    print("-" * 54)
    for agent, d in totals.items():
        correct_str = f"{d['correct']}/{d['total']}"
        print(f"{agent:<20} {correct_str:>8} {d['steps']:>8} {d['tokens']:>14,}")
    print()

if __name__ == "__main__":
    main()
