# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install for development:**
```bash
pip install -e ".[dev]"
# or with uv:
uv pip install -e "smolagents[dev] @ ."
```

**Lint / format:**
```bash
make quality   # check only
make style     # auto-fix
```

**Run all tests:**
```bash
make test
# or directly:
pytest ./tests/
```

**Run a single test file or test:**
```bash
pytest tests/test_agents.py
pytest tests/test_agents.py::TestCodeAgent::test_fake_code_agent
```

**CLI entry points** (installed by pyproject.toml):
```bash
smolagent   # maps to smolagents.cli:main
webagent    # maps to smolagents.vision_web_browser:main
```

## Architecture

`smolagents` is a minimal agentic library (~1,000 LOC for core agent logic). All public symbols are re-exported through `src/smolagents/__init__.py` via star-imports from the modules below.

### Core modules

| File | Purpose |
|---|---|
| `agents.py` | Agent classes and the ReAct loop |
| `models.py` | LLM wrappers (`Model` base + all provider subclasses) |
| `tools.py` | `Tool` / `BaseTool` / `ToolCollection` base classes |
| `default_tools.py` | Built-in tools (search, web, python interpreter, final answer) |
| `memory.py` | Memory dataclasses (`AgentMemory`, `MemoryStep` subclasses) |
| `local_python_executor.py` | Safe Python sandbox for `CodeAgent` |
| `remote_executors.py` | Remote sandbox wrappers (E2B, Docker, Modal, Blaxel) |
| `monitoring.py` | `AgentLogger`, `Monitor`, `TokenUsage` |
| `gradio_ui.py` | Gradio chat UI wrapper |
| `mcp_client.py` | MCP server tool integration |
| `serialization.py` | Agent/tool save/load logic |
| `prompts/` | Jinja2/YAML prompt templates for each agent type |

### Agent hierarchy

```
MultiStepAgent (ABC)  ← base ReAct loop in agents.py
├── CodeAgent         ← LLM writes Python; executed by PythonExecutor
└── ToolCallingAgent  ← LLM emits structured tool calls (JSON)
```

`MultiStepAgent.run()` drives the loop: it calls `step()` repeatedly until a `FinalAnswerTool` call or `max_steps` is hit. Each iteration appends an `ActionStep` to `AgentMemory`.

**Multi-agent / managed agents:** Any `MultiStepAgent` can be registered as a tool inside another agent via `managed_agents=`. The inner agent gets a `name` and `description` and is callable like a tool.

### Model hierarchy

```
Model (base)
├── ApiModel          ← any HTTP-based provider
│   ├── InferenceClientModel   ← HF Inference API / hub providers
│   ├── LiteLLMModel           ← 100+ providers via litellm
│   ├── LiteLLMRouterModel
│   ├── OpenAIModel / AzureOpenAIModel
│   └── AmazonBedrockModel
├── TransformersModel ← local HF transformers
├── VLLMModel         ← local vLLM server
└── MLXModel          ← Apple MLX
```

All models expose a `__call__(messages, stop_sequences, ...)` interface returning `ChatMessage`. `MODEL_REGISTRY` (end of `models.py`) maps string names to classes, used for serialization.

### Tool authoring

Subclass `Tool` and implement `forward()`. Type annotations on `forward` are used to auto-generate the JSON schema the LLM sees. Inputs/outputs declared in class attributes (`name`, `description`, `inputs`, `output_type`) override the inferred schema.

Tools can be shared to/loaded from the HF Hub via `Tool.push_to_hub()` / `Tool.from_hub()`.

### Code execution

`CodeAgent` runs LLM-generated Python through `LocalPythonExecutor` by default, which is a restricted AST-level interpreter (not `exec`). Only modules in `BASE_BUILTIN_MODULES` plus `additional_authorized_imports` are allowed. Remote sandboxes (`E2BExecutor`, `DockerExecutor`, `ModalExecutor`, `BlaxelExecutor`) all subclass `RemotePythonExecutor` and expose the same interface.

### Prompt templates

Each agent type loads a YAML file from `src/smolagents/prompts/` at instantiation:
- `code_agent.yaml` — default `CodeAgent`
- `structured_code_agent.yaml` — `CodeAgent` with `use_structured_outputs_internally=True`
- `toolcalling_agent.yaml` — `ToolCallingAgent`

Templates are Jinja2 strings; override via the `prompt_templates` constructor argument.

### Memory model

`AgentMemory` holds an ordered list of `MemoryStep` subclasses:
- `SystemPromptStep` — system prompt (set once)
- `TaskStep` — the user task
- `PlanningStep` — optional planning output
- `ActionStep` — one ReAct cycle (LLM output + tool result/observation)
- `FinalAnswerStep` — terminal step

Steps are converted to chat messages for each LLM call via `AgentMemory.get_messages_dict()`.

### Logging and thoughts

`CodeAgent` generates a thought + code block each step. By default (`LogLevel.INFO`), only the extracted code is displayed (`log_code` at INFO in `agents.py:1724`). The full model output including the thought is logged at `LogLevel.DEBUG` only. To surface thoughts:

```python
from smolagents.monitoring import LogLevel
agent = CodeAgent(..., verbosity_level=LogLevel.DEBUG)  # prints full LLM output
# or:
agent = CodeAgent(..., stream_outputs=True)  # streams full output live
```

The raw thought text is always stored in `step.model_output` (and `step.model_output_message`) on each `ActionStep` regardless of log level.

### Token usage accounting

`step.token_usage` (an `ActionStep`'s `TokenUsage`) is set from that single LLM call's API response (`response.usage.prompt_tokens`/`completion_tokens`, e.g. `models.py:1789`) — it is per-call, not cumulative. Because `write_memory_to_messages()` resends the full step history every call (`agents.py:758`), `input_tokens` still grows step over step; it just isn't a running total, it's the true size of that call's prompt.

The console trajectory line (`[Step N: ... | Input tokens: X | Output tokens: Y]`) and `agent.monitor.get_total_token_counts()` instead show a *cumulative sum* of each step's `token_usage`, computed in `Monitor.update_metrics` (`monitoring.py:100`). Since each step's own `input_tokens` already includes the resent history, this cumulative sum overcounts — it is not the size of the final context window. `step.token_usage` and the Monitor's cumulative total are answering different questions and will only agree on step 1.

Tools that call `self.model(...)` internally (e.g. `TextInspectorTool` in `examples/open_deep_research`) bypass this accounting entirely — their token usage is discarded and never reaches `step.token_usage` or `Monitor`, so true API spend can exceed both numbers.

### Context pruning (`context_pruning/`)

An experimental subpackage that reduces context bloat from repeated failed tool calls (e.g. many web searches for the same thing). Lives in `src/smolagents/context_pruning/` with a matching test directory `tests/context_pruning/`.

| File | Purpose |
|---|---|
| `subtask_detector.py` | Detects subtask boundaries by comparing the primary tool called in consecutive steps. Tools are grouped (`search`, `browse`, `file`); a group change signals a new subtask. |
| `context_buffer.py` | `SideContextBuffer` — accumulates steps per subtask. On subtask change, only the *last* step of the completed subtask is promoted to committed memory; all prior attempts are dropped. |
| `pruned_memory.py` | `PrunedMemoryMixin` — overrides `write_memory_to_messages()` to build the LLM prompt from the buffer's pruned view instead of the full `memory.steps` list. |
| `agent.py` | `ContextPruningCodeAgent` — wires the three above into a `CodeAgent` subclass. Hooks into `_finalize_step` to update the buffer after each step. |

```
MultiStepAgent
└── CodeAgent
    └── ContextPruningCodeAgent  ← PrunedMemoryMixin + CodeAgent
```

### Testing conventions

Tests live in `tests/`. Fixtures are in `tests/fixtures/` (imported as pytest plugins via `conftest.py`). `MultiStepAgent.__init__` is monkeypatched in `conftest.py` to suppress logging by default. Use `shared_datadir` (from `pytest-datadir`) for test data files. Most agent tests mock the model with a `MagicMock` rather than hitting real APIs.

**Known pre-existing failure** (as of 2026-08-11, branch `agents_as_prob_progs`): `tests/test_agents.py` gives `94 passed, 2 skipped, 1 failed`. The one failure is `TestToolCallingAgent::test_toolcalling_agent_stream_logs_multiple_tool_calls_observations`, with `AttributeError: 'MockChoice' object has no attribute 'index'` surfacing as `AgentGenerationError`.

Cause: the `models.py` n>1 streaming work (committed in `562da82`) made `OpenAIModel.generate_stream` loop over all choices and read `choice.index` unconditionally, but the test's `MockChoice` fixture defines no `index`. Verified by stashing only `models.py` before it was committed — the test passed without it. Not caused by `tracelet_agent.py`, which `ToolCallingAgent` never imports. `getattr(choice, "index", None)` would fix it; until then, treat this single failure as the expected baseline and don't re-investigate it. Any *other* failure in that file is new.

## Playground scripts

| Script | Purpose |
|---|---|
| `playground/compare_agents.py` | Runs `CodeAgent`, `ContextPruningCodeAgent`, and `ProbabilisticCodeAgent` side-by-side on GAIA or simple tasks and prints per-question steps/tokens/correctness. Requires `OPENAI_API_KEY`. |
| `playground/summarize_results.py` | Parses raw stdout from `compare_agents.py` and prints per-agent totals (correct count, total steps, total tokens). |

**Run a comparison:**
```bash
.venv/bin/python playground/compare_agents.py --tasks gaia --limit 10
```

**Summarize saved output:**
```bash
.venv/bin/python playground/summarize_results.py output.txt
# or pipe directly:
.venv/bin/python playground/compare_agents.py --tasks simple | .venv/bin/python playground/summarize_results.py
```

## Collaboration preferences

- **Implement incrementally**: propose and write one small building block at a time so the user can read and review it before moving on. Do not write large monolithic files in a single step.
- Ask which piece to implement next rather than proceeding automatically.
- **Comments: one short line, max.** Never write multi-line or paragraph-length code comments, no matter how subtle the reasoning. Put the longer explanation in the response or the commit message, not in the code.
