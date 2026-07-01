import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../examples/open_deep_research"))

from dotenv import load_dotenv
load_dotenv()

from smolagents import CodeAgent, OpenAIModel
from smolagents.context_pruning.agent import ContextPruningCodeAgent
from smolagents.default_tools import DuckDuckGoSearchTool
from smolagents.memory import ActionStep
from smolagents.monitoring import LogLevel
from smolagents.probabilistic_agent import ProbabilisticCodeAgent
from scripts.text_web_browser import (
    ArchiveSearchTool, FinderTool, FindNextTool,
    PageDownTool, PageUpTool, SimpleTextBrowser, VisitTool,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.visual_qa import visualizer

model = OpenAIModel(model_id="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
browser = SimpleTextBrowser(
    viewport_size=1024 * 5,
    downloads_folder=os.path.join(os.path.dirname(__file__), "downloads_folder"),
    serpapi_key=os.getenv("SERPAPI_API_KEY"),
    request_kwargs={"headers": {"User-Agent": user_agent}, "timeout": 300},
)
os.makedirs(os.path.join(os.path.dirname(__file__), "downloads_folder"), exist_ok=True)

tools = [
    DuckDuckGoSearchTool(),
    VisitTool(browser),
    PageUpTool(browser),
    PageDownTool(browser),
    FinderTool(browser),
    FindNextTool(browser),
    ArchiveSearchTool(browser),
    TextInspectorTool(model, text_limit=100000),
    visualizer,
]

agent_kwargs = dict(
    tools=tools,
    model=model,
    max_steps=50,
    verbosity_level=LogLevel.ERROR,
    additional_authorized_imports=["pandas"],
)

prob_agent = ProbabilisticCodeAgent(n_samples=3, **agent_kwargs)
base_agent = CodeAgent(**agent_kwargs)
pruning_agent = ContextPruningCodeAgent(**agent_kwargs)

def collect_metrics(agent):
    steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    total_tokens = sum(s.token_usage.total_tokens for s in steps if s.token_usage)
    return len(steps), total_tokens

import argparse


class Tee:
    """Writes to both the original stream and a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


parser = argparse.ArgumentParser()
parser.add_argument("--tasks", choices=["simple", "gaia"], default="gaia")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--output", type=str, default=None, help="Path to also save output to")
args = parser.parse_args()

if args.output:
    output_file = open(args.output, "w")
    sys.stdout = Tee(sys.stdout, output_file)

simple_tasks = [
    ("What is the current population of Tokyo?", None),
    ("Who won the most recent FIFA World Cup?", None),
    ("What is the latest version of Python?", None),
]

gaia_data = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "gaia_results.jsonl"))]
gaia_tasks = [(d["question"], d["true_answer"]) for d in gaia_data[:10]]

tasks = simple_tasks if args.tasks == "simple" else gaia_tasks
if args.limit is not None:
    tasks = tasks[:args.limit]

print(f"\n{'='*70}")
print(f"{'':6} {'Agent':<16} {'Steps':>6} {'Tokens':>10} {'Correct':>8}")
print(f"{'='*70}")

for idx, (question, true_answer) in enumerate(tasks):
    print(f"\nQ{idx+1}: {question[:80]}...")
    for agent, label in [(base_agent, "CodeAgent"), (pruning_agent, "PruningAgent"), (prob_agent, "ProbAgent(n=3)")]:
        agent.memory.reset()
        result = agent.run(question)
        steps, tokens = collect_metrics(agent)
        if true_answer is not None:
            correct = str(result).strip().lower() == str(true_answer).strip().lower()
            correctness = '✅' if correct else '❌'
        else:
            correctness = "  "
        print(f"  {'':4} {label:<16} {steps:>6} {tokens:>10,} {correctness:>8}  → {str(result)[:60]}")

print(f"\n{'='*70}")
