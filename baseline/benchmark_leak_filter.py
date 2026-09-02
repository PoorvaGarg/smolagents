"""Drop search results that leak GAIA answer keys.

Two leak vectors were confirmed in run trajectories:
  * huggingface.co Spaces holding questions.json / gaia_*.jsonl with "Final answer" fields
  * GitHub issues where someone pasted an eval log containing "true_answer"

Both reached the agent as *search snippets*, so filtering search results matters more than
blocking page fetches. URL patterns cover hosts that only ever serve benchmark dumps; the
content markers catch dumps on hosts (like github.com) that also serve legitimate material.
"""

import re

from smolagents import DuckDuckGoSearchTool

# Hosts/paths that serve benchmark dumps and nothing an agent legitimately needs.
LEAK_URL_PATTERNS = (
    r"huggingface\.co/spaces/",
    r"huggingface\.co/datasets/[^/]*gaia",
    r"hf\.co/spaces/",
    r"/questions\.json",
    r"gaia[_-]?\d*\.jsonl",
)

# Field names distinctive to an eval-harness record, not to prose about a topic.
LEAK_CONTENT_PATTERNS = (
    r"true_answer",
    r"intermediate_steps",
    r"iteration_limit_exceeded",
    r"augmented_question",
    r"parsing_error",
    r'"Final answer"\s*:',
    r'"task_id"\s*:',
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


class LeakFilteredDuckDuckGoSearchTool(DuckDuckGoSearchTool):
    """DuckDuckGoSearchTool that drops answer-key results and says so, rather than silently."""

    def forward(self, query: str) -> str:
        self._enforce_rate_limit()
        results = self.ddgs.text(query, max_results=self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")

        kept, dropped = [], []
        for r in results:
            reason = leak_reason(r.get("href", ""), f"{r.get('title', '')}\n{r.get('body', '')}")
            (dropped if reason else kept).append((r, reason))

        if dropped:
            print(f"  [leak filter] dropped {len(dropped)}/{len(results)} result(s) for '{query[:60]}':")
            for r, reason in dropped:
                print(f"      {r.get('href', '')[:90]}  ({reason})")

        if not kept:
            raise Exception(
                "All results for this query were excluded as benchmark answer-key pages. "
                "Rephrase the query to search for the underlying subject matter instead."
            )
        body = [f"[{r['title']}]({r['href']})\n{r['body']}" for r, _ in kept]
        return "## Search Results\n\n" + "\n\n".join(body)
