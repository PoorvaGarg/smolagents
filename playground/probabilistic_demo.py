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
    max_steps=1,
    verbosity_level=LogLevel.DEBUG,
)

import smolagents.probabilistic_agent as _pa

_original_completions = _pa.ProbabilisticCodeAgent._generate_completions
_original_trajectories = _pa.ProbabilisticCodeAgent._generate_trajectories

def _debug_completions(self, memory_messages, output_prefix, n):
    print("\n=== output_prefix sent to model (the shared prefix before divergence) ===")
    print(repr(output_prefix))
    completions = _original_completions(self, memory_messages, output_prefix, n)
    print(f"\n=== {len(completions)} completions returned ===")
    for i, c in enumerate(completions):
        suffix = c[len(output_prefix):]
        print(f"  completion[{i}] suffix: {repr(suffix)}")
    return completions

def _debug_trajectories(self, memory_messages, output_prefix, n):
    trajectories = _original_trajectories(self, memory_messages, output_prefix, n)
    print(f"\n=== {len(trajectories)} trajectories ===")
    for i, (first_output, obs, next_output, is_final) in enumerate(trajectories):
        print(f"\n  [Trajectory {i}] is_final={is_final}")
        print(f"    observation: {repr(obs[:120])}")
    return trajectories

_pa.ProbabilisticCodeAgent._generate_completions = _debug_completions
_pa.ProbabilisticCodeAgent._generate_trajectories = _debug_trajectories

result = agent.run("What is the current population of Tokyo?")
print("\nFinal answer:", result)
