"""Shared GAIA baseline setup: tool construction, dataset loading, scoring, and eval loop.

Used by both naiveReAct.ipynb (CodeAgent) and markovReAct/markovReAct.ipynb
(MarkovReActCodeAgent) so the two stay directly comparable: same tools, same GAIA
split, same scorer, same evaluate_agent loop.
"""

import json
import os
import pickle
import re
import string
import sys
import warnings
from datetime import datetime
from pathlib import Path

import datasets
import pandas as pd  # noqa: F401  (kept for notebook callers that do `from common_setup import *`)
from huggingface_hub import login, snapshot_download

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ODR_DIR = _REPO_ROOT / "examples" / "open_deep_research"
if str(_ODR_DIR) not in sys.path:
    sys.path.insert(0, str(_ODR_DIR))

from smolagents import DuckDuckGoSearchTool
from smolagents.memory import ActionStep
from scripts.run_agents import get_single_file_description, get_zip_description
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer


def assert_no_reasoning(memory_step, agent=None):
    """Step callback: fail loudly the moment any step returns reasoning despite enable_thinking=False,
    rather than only noticing it later by eyeballing console output."""
    reasoning = memory_step.model_output_message.reasoning if memory_step.model_output_message else None
    assert not reasoning, f"Expected no reasoning (enable_thinking=False) but got: {reasoning!r}"


def build_tools(model):
    """Construct the standard GAIA tool stack (search, browser, file inspector, visualizer)."""
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    )
    browser = SimpleTextBrowser(
        viewport_size=1024 * 5,
        downloads_folder="downloads_folder",
        serpapi_key=os.getenv("SERPAPI_API_KEY"),
        request_kwargs={"headers": {"User-Agent": user_agent}, "timeout": 300},
    )
    os.makedirs("downloads_folder", exist_ok=True)

    ti_tool = TextInspectorTool(model, text_limit=100000)

    tools = [
        DuckDuckGoSearchTool(),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        ti_tool,
        visualizer,
    ]
    return tools, ti_tool, visualizer


def load_gaia_dataset(set_to_run: str = "validation", data_dir: str = "data/gaia"):
    """Load the GAIA benchmark split, downloading attachment files if not already present."""
    login(os.environ.get("HF_TOKEN"))

    if not os.path.exists(data_dir):
        snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            local_dir=data_dir,
            ignore_patterns=[".gitattributes", "README.md"],
        )

    eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all", split=set_to_run)
    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})

    def resolve_file_path(row):
        # file_path (e.g. "2023/validation/<uuid>.xlsx") already matches where snapshot_download
        # placed the real file on disk, unlike file_name which is just the bare filename.
        if row["file_path"]:
            row["file_name"] = f"{data_dir}/{row['file_path']}"
        return row

    eval_ds = eval_ds.map(resolve_file_path)
    return eval_ds


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def normalize_str(input_str, remove_punct=True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def split_string(s: str, char_list=[",", ";"]) -> list:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    if is_float(ground_truth):
        return normalize_number_str(str(model_answer)) == float(ground_truth)
    elif any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(str(model_answer))
        if len(gt_elems) != len(ma_elems):
            warnings.warn("Answer lists have different lengths, returning False.")
            return False
        return all(
            normalize_number_str(ma) == float(gt) if is_float(gt)
            else normalize_str(ma, remove_punct=False) == normalize_str(gt, remove_punct=False)
            for ma, gt in zip(ma_elems, gt_elems)
        )
    else:
        return normalize_str(str(model_answer)) == normalize_str(ground_truth)


def evaluate_agent(
    agent,
    dataset,
    ti_tool,
    visualizer,
    n_samples=None,
    output_file="gaia_results.jsonl",
    pickle_dir="gaia_results",
):
    examples = dataset.to_list()
    if n_samples:
        examples = examples[:n_samples]

    output_path = Path(output_file)
    pickle_path = Path(pickle_dir)
    pickle_path.mkdir(parents=True, exist_ok=True)
    results = []

    for i, example in enumerate(examples):
        cache_file = pickle_path / f"{i}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                result = pickle.load(f)
            results.append(result)
            print(f"\n[{i + 1}/{len(examples)}] (cached) {example['question'][:100]}...")
            is_correct = result.get("is_correct", False)
            print(f"  {'✓' if is_correct else '✗'} | cached")
            continue

        print(f"\n[{i + 1}/{len(examples)}] {example['question'][:100]}...")

        augmented_question = example["question"]
        if example["file_name"]:
            if ".zip" in example["file_name"]:
                prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
                prompt_use_files += get_zip_description(
                    example["file_name"], example["question"], visualizer, ti_tool
                )
            else:
                prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
                prompt_use_files += get_single_file_description(
                    example["file_name"], example["question"], visualizer, ti_tool
                )
            augmented_question += prompt_use_files

        start_time = datetime.now()
        try:
            prediction = str(agent.run(augmented_question, reset=True))
            error = None
        except Exception as e:
            prediction = None
            error = str(e)
        time_taken = (datetime.now() - start_time).total_seconds()

        is_correct = question_scorer(prediction, example["true_answer"]) if prediction else False

        action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
        num_steps = len(action_steps)

        tool_names = list(agent.tools.keys())
        tool_usage = {name: 0 for name in tool_names}
        for step in action_steps:
            code = step.code_action or ""
            for name in tool_names:
                tool_usage[name] += code.count(f"{name}(")

        tokens_per_step = []
        for step in action_steps:
            tokens_per_step.append(step.token_usage.dict() if step.token_usage else None)
        token_counts = agent.monitor.get_total_token_counts()

        assert token_counts.total_tokens == sum(
            step.token_usage.total_tokens for step in action_steps if step.token_usage
        ), "Token counts mismatch!"

        result = {
            "task_id": example.get("task_id", ""),
            "question": example["question"],
            "augmented_question": augmented_question,
            "true_answer": example["true_answer"],
            "prediction": prediction,
            "is_correct": is_correct,
            "task": example["task"],
            "time_taken_seconds": round(time_taken, 2),
            "num_steps": num_steps,
            "tool_usage": tool_usage,
            "token_counts": token_counts.dict(),
            "tokens_per_step": tokens_per_step,
            "error": error,
            "steps": [step.dict() for step in action_steps],
        }
        results.append(result)

        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        with open(pickle_path / f"{i}.pkl", "wb") as f:
            pickle.dump(result, f)

        print(f"  {'✓' if is_correct else '✗'} | {time_taken:.1f}s | {num_steps} steps | tokens: {token_counts}")
        if not is_correct:
            print(f"    Expected: {example['true_answer']}")
            print(f"    Got:      {prediction}")

    return results
