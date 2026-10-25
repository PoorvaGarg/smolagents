"""SimpleQA baseline setup: dataset loading and LLM-judge scoring.

Dataset: OpenAI's official SimpleQA test set (4,326 questions), Wei et al., "Measuring short-form
factuality in large language models", 2024 (https://cdn.openai.com/papers/simpleqa.pdf), from the
public CSV that openai/simple-evals itself loads.

Scorer: the official SimpleQA grader. The prompt is extracted verbatim from openai/simple-evals'
simpleqa_eval.py (see third_party/simpleqa_grading/README.md), called with the same settings as
upstream: gpt-4.1-2025-04-14, system "You are a helpful assistant.", temperature 0.5, max_tokens 2048.
Each answer is graded CORRECT / INCORRECT / NOT_ATTEMPTED. Only CORRECT counts as correct; the label
itself is kept so not-attempted rates and the paper's F1 can be computed later.

Returns a Dataset shaped like load_gaia_dataset's output, so evaluate_agent works unmodified.
"""

import os
import re
from pathlib import Path

import datasets
import pandas as pd
import requests

from smolagents import OpenAIModel
from smolagents.models import ChatMessage, MessageRole
from third_party.simpleqa_grading.grader_template import CHOICE_LETTER_TO_STRING, GRADER_TEMPLATE


SIMPLEQA_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
# Kept outside the repo so the agent's Python can't read answers via a relative path.
_DEFAULT_DATA_DIR = Path(
    os.getenv("BENCHMARK_DATA_DIR", Path(__file__).resolve().parents[2] / ".benchmark_data")
) / "simpleqa"

GRADER_MODEL_ID = "gpt-4.1-2025-04-14"
GRADER_SYSTEM_MESSAGE = "You are a helpful assistant."

SIMPLEQA_ANSWER_INSTRUCTION = "\n\nPass only a short final answer to final_answer, with no explanation."


def load_simpleqa_dataset(data_dir: str | Path | None = None) -> datasets.Dataset:
    data_dir = Path(data_dir or _DEFAULT_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "simple_qa_test_set.csv"
    if not csv_path.exists():
        response = requests.get(SIMPLEQA_CSV_URL)
        response.raise_for_status()
        csv_path.write_bytes(response.content)

    df = pd.read_csv(csv_path)
    examples = [
        {
            "task_id": str(i),
            "question": row["problem"],
            "true_answer": row["answer"],
            "task": "simpleqa",
            "file_name": None,
        }
        for i, row in df.iterrows()
    ]
    return datasets.Dataset.from_list(examples)


class SimpleQAGrader:
    """Callable scorer for evaluate_agent; it needs the question, unlike the string-matching scorers."""

    takes_question = True

    def __init__(self, api_key: str | None = None):
        self.model = OpenAIModel(
            model_id=GRADER_MODEL_ID,
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            temperature=0.5,
            max_tokens=2048,
        )
        self.last_label = None

    def grade(self, question: str, true_answer: str, prediction: str) -> str:
        prompt = GRADER_TEMPLATE.format(question=question, target=true_answer, predicted_answer=prediction)
        response = self.model.generate([
            ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": GRADER_SYSTEM_MESSAGE}]),
            ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}]),
        ])
        # Upstream's parse: first A/B/C anywhere in the reply, defaulting to NOT_ATTEMPTED.
        match = re.search(r"(A|B|C)", response.content or "")
        return CHOICE_LETTER_TO_STRING[match.group(0) if match else "C"]

    def __call__(self, prediction: str, true_answer: str, question: str) -> bool:
        self.last_label = self.grade(question, true_answer, str(prediction))
        return self.last_label == "CORRECT"
