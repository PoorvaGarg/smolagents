"""Fermi Problems baseline setup: dataset loading and scoring.

Uses the RealFP split from https://github.com/allenai/fermi (Kalyan et al., 2021).
Reuses build_tools/evaluate_agent from common_setup.py so naiveReAct stays directly
comparable across GAIA and Fermi -- only the dataset and scorer differ.
"""

import json
import math
import os
import re
from pathlib import Path

import datasets
import pint
import requests

FERMI_RAW_BASE = "https://raw.githubusercontent.com/allenai/fermi/main"
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "fermi"

FERMI_ANSWER_INSTRUCTION = (
    "\n\nGive your final answer as a number with units matching the question (e.g. \"22.3 km\", "
    "\"79 g\"; use a bare number like \"33000\" only if the question has no natural unit). "
    "Do not include filler words like \"about\" or \"roughly\", or thousands separators. "
    "Write the full number of digits (e.g. \"256000000000\") rather than word multipliers like "
    "\"million\" or \"billion\"."
)


def load_fermi_dataset(data_dir: str | Path | None = None, split: str = "val") -> datasets.Dataset:
    """Download (once) and load the RealFP split as a Dataset shaped like load_gaia_dataset's
    output: question/true_answer/task/file_name/task_id columns, so evaluate_agent works unmodified."""
    data_dir = data_dir or _DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    json_path = Path(data_dir) / f"{split}_realfp.json"
    if not json_path.exists():
        response = requests.get(f"{FERMI_RAW_BASE}/data/realFP/{split}_realfp.json")
        response.raise_for_status()
        json_path.write_bytes(response.content)

    with open(json_path) as f:
        records = json.load(f)

    examples = [
        {
            "task_id": str(i),
            "question": record["question"],
            "true_answer": record["answer"],
            "task": "fermi",
            "file_name": None,
        }
        for i, record in enumerate(records)
    ]
    return datasets.Dataset.from_list(examples)


def _load_unit_registry(data_dir: str | Path | None = None) -> pint.UnitRegistry:
    """Fermi's ground-truth answers use a few non-standard unit names (e.g. "person", "pa" for
    pascal) defined in the repo's units.txt, on top of pint's default mks registry."""
    data_dir = data_dir or _DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    units_path = Path(data_dir) / "units.txt"
    if not units_path.exists():
        response = requests.get(f"{FERMI_RAW_BASE}/units.txt")
        response.raise_for_status()
        units_path.write_bytes(response.content)

    ureg = pint.UnitRegistry(system="mks", autoconvert_offset_to_baseunit=True)
    ureg.load_definitions(str(units_path))
    return ureg


_UREG = _load_unit_registry()

_FILLER_WORDS_RE = re.compile(
    r"\b(about|approximately|approx\.?|roughly|around|nearly|estimated|estimate|circa)\b", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def _clean_answer_text(answer: str) -> str:
    """Strip filler words and thousands-separator commas an agent's free-text final answer
    tends to include but pint can't parse (e.g. "roughly 100,000 people" -> "100000 people")."""
    text = _FILLER_WORDS_RE.sub("", str(answer))
    text = text.replace(",", "")
    return text.strip()


def _extract_leading_number(answer: str) -> float | None:
    match = _NUMBER_RE.search(_clean_answer_text(answer))
    return float(match.group().replace(",", "")) if match else None


def _parse_quantity(answer: str) -> tuple[float, str] | None:
    """Parse a "<number> <unit>" (or bare number) string to (magnitude, base_unit_str)."""
    try:
        quantity = _UREG(_clean_answer_text(answer))
    except Exception:
        return None
    if isinstance(quantity, (int, float)):
        return float(quantity), ""
    try:
        base = quantity.to_base_units()
        unit_str = str(base.units)
        return float(base.magnitude), "" if unit_str == "dimensionless" else unit_str
    except Exception:
        return None


def _accuracy_metric(true_value: float, predicted_value: float) -> float:
    """Order-of-magnitude accuracy metric from allenai/fermi's eval_utils.py: 1.0 for an exact
    match, decaying to 0 once the two values are 3 orders of magnitude apart."""
    if true_value < 0 or predicted_value < 0:
        return 0.0
    if true_value == 0 and predicted_value == 0:
        return 1.0
    if true_value == 0 or predicted_value == 0:
        return max(0.0, 1 - abs(math.log10(abs(true_value - predicted_value))))
    return max(0.0, 3 - abs(math.log10(true_value / predicted_value))) / 3


def fermi_scorer(prediction: str, true_answer: str) -> float:
    """Score a predicted answer against the ground truth on a continuous 0-1 scale, matching
    units (e.g. km vs. miles) before comparing magnitudes.

    If the prediction has no units at all -- a common failure mode where the agent's estimate is
    numerically right but the units got dropped (e.g. "12.2" for a "22.3 km" answer) -- fall back
    to comparing bare numeric literals instead of scoring 0 outright. A prediction with units that
    are simply incompatible with the ground truth (e.g. seconds vs. km) still scores 0: that's a
    real error, not a formatting slip.
    """
    if not prediction:
        return 0.0
    truth = _parse_quantity(true_answer)
    if truth is None:
        return 0.0
    true_value, true_unit = truth

    predicted = _parse_quantity(prediction)
    if predicted is not None:
        predicted_value, predicted_unit = predicted
        if predicted_unit == true_unit:
            return _accuracy_metric(true_value, predicted_value)
        if predicted_unit != "":
            return 0.0

    predicted_number = _extract_leading_number(prediction)
    true_number = _extract_leading_number(true_answer)
    if predicted_number is None or true_number is None:
        return 0.0
    return _accuracy_metric(true_number, predicted_number)
