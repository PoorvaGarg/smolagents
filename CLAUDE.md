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

### Testing conventions

Tests live in `tests/`. Fixtures are in `tests/fixtures/` (imported as pytest plugins via `conftest.py`). `MultiStepAgent.__init__` is monkeypatched in `conftest.py` to suppress logging by default. Use `shared_datadir` (from `pytest-datadir`) for test data files. Most agent tests mock the model with a `MagicMock` rather than hitting real APIs.
