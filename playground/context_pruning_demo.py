import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../examples/open_deep_research"))

from dotenv import load_dotenv

load_dotenv()

import argparse
import json

from smolagents import OpenAIModel
from smolagents.context_pruning.agent import ContextPruningCodeAgent
from smolagents.context_pruning.subtask_detector import get_tool_group
from smolagents.default_tools import DuckDuckGoSearchTool
from smolagents.memory import ActionStep
from smolagents.monitoring import LogLevel
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

agent = ContextPruningCodeAgent(
    tools=tools,
    model=model,
    max_steps=50,
    verbosity_level=LogLevel.ERROR,
    additional_authorized_imports=["pandas"],
)

# --- instrumentation ---

_original_generate = model.generate

def _debug_generate(messages, **kwargs):
    print(f"\n{'~'*70}")
    print(f"PROMPT TO LLM ({len(messages)} messages):")
    for i, msg in enumerate(messages):
        role = msg.role if hasattr(msg, "role") else msg.get("role", "?")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            )
        snippet = str(content)[:500].replace("\n", "↵ ")
        print(f"  [{i}] {role}: {snippet}")
    print(f"{'~'*70}\n")
    response = _original_generate(messages, **kwargs)
    print(f"\nLLM RESPONSE:\n{str(response.content)[:1000]}\n")
    return response

model.generate = _debug_generate

import smolagents.context_pruning.agent as _cpa

_original_finalize = _cpa.ContextPruningCodeAgent._finalize_step
_step_log: list[dict] = []


def _instrumented_finalize(self, memory_step):
    prev_committed = len(self._ctx_buffer.committed_steps)
    prev_active = len(self._ctx_buffer.current_context)
    prev_code = self._prev_code

    _original_finalize(self, memory_step)

    if not isinstance(memory_step, ActionStep):
        return

    code = memory_step.code_action or ""
    tool = self._subtask_detector.primary_tool(code)
    group = get_tool_group(tool)

    boundary = len(self._ctx_buffer.committed_steps) > prev_committed

    _step_log.append({
        "step": memory_step.step_number,
        "tool": tool or "(none)",
        "group": group,
        "boundary": boundary,
        "pruned_count": prev_active - 1 if boundary else 0,
    })

    marker = " ← subtask boundary" if boundary else ""
    print(
        f"  step {memory_step.step_number:>2} | group={group:<12} tool={tool or '(none)':<28}{marker}"
    )


_cpa.ContextPruningCodeAgent._finalize_step = _instrumented_finalize

# --- CLI ---


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
parser.add_argument("--question", type=int, default=None, help="1-based index")
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

if args.question is not None:
    tasks = [tasks[args.question - 1]]

for idx, (question, true_answer) in enumerate(tasks):
    q_num = args.question if args.question else idx + 1
    print(f"\n{'='*70}")
    print(f"Q{q_num}: {question[:100]}")
    if true_answer:
        print(f"Expected: {true_answer}")
    print(f"{'='*70}")

    _step_log.clear()
    agent.memory.reset()
    result = agent.run(question)

    # summary table
    all_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    kept_steps = {id(s) for s in agent._ctx_buffer.get_pruned_steps()}
    total_pruned = sum(e["pruned_count"] for e in _step_log)

    print(f"\n--- pruning summary ---")
    print(f"  total steps: {len(all_steps)}")
    print(f"  steps pruned from context: {total_pruned}")
    print(f"  steps kept in final context: {len(agent._ctx_buffer.get_pruned_steps())}")
    print(f"\n  step | kept | group        | tool")
    print(f"  -----|------|--------------|-------------------------------")
    for s in all_steps:
        kept = "YES " if id(s) in kept_steps else "no  "
        code = s.code_action or ""
        tool = agent._subtask_detector.primary_tool(code) or "(none)"
        group = get_tool_group(tool if tool != "(none)" else None)
        print(f"  {s.step_number:>4} | {kept} | {group:<12} | {tool}")

    print(f"\nFinal answer: {result}")
    if true_answer:
        correct = str(result).strip().lower() == str(true_answer).strip().lower()
        print(f"Correct: {'YES' if correct else 'NO'}  (expected: {true_answer})")
