import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from smolagents import OpenAIModel
from smolagents.default_tools import DuckDuckGoSearchTool
from smolagents.monitoring import LogLevel
from smolagents.probabilistic_agent import ProbabilisticCodeAgent

model = OpenAIModel(
    model_id="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
)

agent = ProbabilisticCodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    n_samples=3,
    max_steps=50,
    verbosity_level=LogLevel.DEBUG,
)

import smolagents.probabilistic_agent as _pa

_original_completions = _pa.ProbabilisticCodeAgent._generate_completions
_original_trajectories = _pa.ProbabilisticCodeAgent._generate_trajectories

def _debug_completions(self, memory_messages, n):
    completions, token_usage = _original_completions(self, memory_messages, n)
    print(f"\n=== {len(completions)} completions ===")
    for i, c in enumerate(completions):
        print(f"\n  [completion {i}]:\n{c}")
    return completions, token_usage

def _debug_trajectories(self, memory_messages, n):
    trajectories, token_usage = _original_trajectories(self, memory_messages, n)
    print(f"\n=== {len(trajectories)} trajectories (tokens: {token_usage.total_tokens}) ===")
    for i, (first_output, obs, next_output, is_final) in enumerate(trajectories):
        print(f"\n  [Trajectory {i}] is_final={is_final}")
        print(f"    observation: {repr(obs[:120])}")
    return trajectories, token_usage

_pa.ProbabilisticCodeAgent._generate_completions = _debug_completions
_pa.ProbabilisticCodeAgent._generate_trajectories = _debug_trajectories

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--tasks", choices=["simple", "gaia"], default="gaia")
parser.add_argument("--question", type=int, default=None, help="1-based index of a single question to run")
args = parser.parse_args()

simple_tasks = [
    "What is the current population of Tokyo?",
    "Who won the most recent FIFA World Cup?",
    "What is the latest version of Python?",
]

gaia_data = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "gaia_results.jsonl"))]
gaia_tasks = [(d["question"], d["true_answer"]) for d in gaia_data[:3]]

if args.tasks == "simple":
    tasks = [(q, None) for q in simple_tasks]
else:
    tasks = gaia_tasks

if args.question is not None:
    tasks = [tasks[args.question - 1]]

for question, true_answer in tasks:
    print(f"\n{'='*60}\nTask: {question}")
    if true_answer is not None:
        print(f"Expected: {true_answer}")
    print('='*60)
    agent.memory.reset()
    result = agent.run(question)
    print(f"\nFinal answer: {result}")
    if true_answer is not None:
        correct = str(result).strip().lower() == str(true_answer).strip().lower()
        print(f"Expected:     {true_answer}")
        print(f"Correct:      {correct}")
