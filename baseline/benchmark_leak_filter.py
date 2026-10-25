"""Block benchmark answer keys (GAIA, Fermi, MATH-500, SimpleQA) from reaching the agent.

v1 (GAIA only) filtered search results. Leak vectors confirmed in run trajectories:
  * huggingface.co Spaces holding questions.json / gaia_*.jsonl with "Final answer" fields
  * GitHub issues where someone pasted an eval log containing "true_answer"
v2 (2026-10-25) adds the other three datasets and blocks page *visits* too. An audit of past
trajectories found agents calling visit_page() directly on HF dataset mirrors
(e.g. huggingface.co/datasets/zyang39/GAIA) in filter-on runs. v1 never saw visits, and its HF
pattern only checked the org name, not the repo name.

URL patterns cover hosts and paths that only ever serve benchmark data. Content markers catch dumps
on hosts like github.com, which also serve legitimate material.
"""

import json
import os
import re
import sys
import time
from pathlib import Path


_ODR_DIR = Path(__file__).resolve().parent.parent / "examples" / "open_deep_research"
if str(_ODR_DIR) not in sys.path:
    sys.path.insert(0, str(_ODR_DIR))

from scripts.text_inspector_tool import TextInspectorTool  # noqa: E402
from scripts.text_web_browser import SimpleTextBrowser  # noqa: E402
from smolagents import DuckDuckGoSearchTool


LEAK_FILTER_VERSION = 2

_BENCH = r"(gaia|fermi|realfp|math|prm800k|simple[_-]?qa)"

# Hosts/paths that serve benchmark dumps and nothing an agent legitimately needs.
LEAK_URL_PATTERNS = (
    r"(huggingface|hf)\.co/spaces/",
    rf"(huggingface|hf)\.co/(api/)?datasets/[^\s?#]*{_BENCH}",
    rf"datasets-server\.huggingface\.co/\S*{_BENCH}",
    rf"kaggle\.com/datasets/\S*{_BENCH}",
    r"/questions\.json",
    # JSON only: ESA's Gaia star catalogs ship as CSV/FITS and are legitimate sources.
    r"gaia[^/\s]*\.jsonl?",
    r"metadata(\.level\d)?\.(jsonl|parquet)",
    # Fermi
    r"(github\.com|raw\.githubusercontent\.com)/allenai/fermi",
    r"realfp",
    # MATH / MATH-500
    r"(github\.com|raw\.githubusercontent\.com)/(hendrycks/math|openai/prm800k)",
    r"people\.eecs\.berkeley\.edu/~hendrycks",
    r"math[_-]?500",
    # SimpleQA
    r"(github\.com|raw\.githubusercontent\.com)/openai/simple-evals",
    r"openaipublic\.blob\.core\.windows\.net/simple-evals",
    r"simple[_-]?qa",
)

# Field names and markers distinctive to dataset records or eval harnesses, not to prose.
LEAK_CONTENT_PATTERNS = (
    r"true_answer",
    r"intermediate_steps",
    r"iteration_limit_exceeded",
    r"augmented_question",
    r"parsing_error",
    r'"Final answer"\s*:',
    r'"task_id"\s*:',
    # Fermi records
    r"PROGRAM:=Q1:",
    r"CONTEXT:=F1:",
    # MATH records
    r'"unique_id"\s*:\s*"test/',
    r'"level"\s*:\s*"Level \d',
    # SimpleQA CSV header and its metadata dicts
    r"metadata,problem,answer",
    r"['\"]answer_type['\"]\s*:",
)

_URL_RE = re.compile("|".join(LEAK_URL_PATTERNS), re.I)
_CONTENT_RE = re.compile("|".join(LEAK_CONTENT_PATTERNS), re.I)


def leak_reason(url: str = "", text: str = "") -> str | None:
    """Return a short reason if this result looks like a benchmark dump, else None."""
    m = _URL_RE.search(url or "")
    if m:
        return f"url matches {m.group(0)!r}"
    m = _CONTENT_RE.search(text or "")
    if m:
        return f"content contains {m.group(0)!r}"
    return None


BLOCKED_PAGE_MESSAGE = (
    "This page was blocked: it holds benchmark dataset records or answer keys, not source "
    "material. Look for the underlying subject matter elsewhere."
)


class _BlockLogger:
    """Shared by the guards below; evaluate_agent fills in log_path and question_idx via logs_drops."""

    logs_drops = True
    log_path = None
    question_idx = None

    def _log_block(self, kind: str, target: str, reason: str) -> None:
        print(f"  [leak filter] blocked {kind} {target[:90]}  ({reason})")
        if not self.log_path:
            return
        record = {"ts": round(time.time(), 3), "question_idx": self.question_idx,
                  "kind": kind, "url": target, "reason": reason}
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            print(f"  [leak filter] could not write {self.log_path}: {e}")


class LeakFilteredTextBrowser(_BlockLogger, SimpleTextBrowser):
    """Refuses answer-key pages for every browsing tool, by URL before fetching and by content after."""

    def __init__(self, *args, log_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path or os.getenv("SMOLAGENTS_LEAK_LOG")

    def _fetch_page(self, url: str) -> None:
        # Every navigation path lands here: visit_page, relative links, downloads, archive snapshots.
        reason = leak_reason(url=url)
        if reason is None:
            super()._fetch_page(url)
            reason = leak_reason(text=self.page_content)
        if reason is None:
            return
        self._log_block("visit", url, reason)
        self.page_title = "Blocked"
        self._set_page_content(BLOCKED_PAGE_MESSAGE)


class LeakFilteredTextInspectorTool(_BlockLogger, TextInspectorTool):
    """Refuses answer-key files the agent asks to inspect, by path and by content.

    Only forward() is guarded: forward_initial_exam_mode is the harness describing GAIA
    attachments to the agent, not the agent choosing what to read.
    """

    def __init__(self, *args, log_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path or os.getenv("SMOLAGENTS_LEAK_LOG")

    def forward(self, file_path, question: str | None = None) -> str:
        reason = leak_reason(url=str(file_path))
        if reason is None:
            try:
                reason = leak_reason(text=self.md_converter.convert(file_path).text_content)
            except Exception:
                reason = None  # let the real forward() surface the conversion error
        if reason is not None:
            self._log_block("file", str(file_path), reason)
            return BLOCKED_PAGE_MESSAGE
        return super().forward(file_path, question)


class LeakFilteredDuckDuckGoSearchTool(DuckDuckGoSearchTool):
    """DuckDuckGoSearchTool that drops answer-key results and says so, rather than silently."""

    # Marker for the eval loop: this tool wants a log path and the current question index.
    logs_drops = True

    def __init__(self, *args, log_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path or os.getenv("SMOLAGENTS_LEAK_LOG")
        self.question_idx = None

    def _log_drops(self, query: str, n_results: int, kept: int, dropped: list) -> None:
        """Record every search that lost results; the stdout lines below were never captured by nbconvert."""
        if not self.log_path:
            return
        record = {
            "ts": round(time.time(), 3),
            "question_idx": self.question_idx,
            "query": query,
            "n_results": n_results,
            "kept": kept,
            "all_dropped": kept == 0,
            "dropped": [{"url": r.get("href", ""), "reason": reason} for r, reason in dropped],
        }
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            # A logging failure must not abort a 165-question sweep.
            print(f"  [leak filter] could not write {self.log_path}: {e}")

    def forward(self, query: str) -> str:
        results = self._search_with_retry(query)

        kept, dropped = [], []
        for r in results:
            reason = leak_reason(r.get("href", ""), f"{r.get('title', '')}\n{r.get('body', '')}")
            (dropped if reason else kept).append((r, reason))

        if dropped:
            print(f"  [leak filter] dropped {len(dropped)}/{len(results)} result(s) for '{query[:60]}':")
            for r, reason in dropped:
                print(f"      {r.get('href', '')[:90]}  ({reason})")
            self._log_drops(query, len(results), len(kept), dropped)

        if not kept:
            raise Exception(
                "All results for this query were excluded as benchmark answer-key pages. "
                "Rephrase the query to search for the underlying subject matter instead."
            )
        body = [f"[{r['title']}]({r['href']})\n{r['body']}" for r, _ in kept]
        return "## Search Results\n\n" + "\n\n".join(body)
