# AutoNeuro

## Code Structure

<a id="code-structure"></a>

```
├── .cache/                    - Local storage for data shards, TRIBE weights, and tokenizers
├── agents/                    - The Agentic Swarm Logic
│   ├── coding_agent.md        - System prompts/tools for modifying code
│   ├── research_agent.md      - System prompts for analyzing papers and providing context
│   └── contact_agent.md       - Logic for Human-in-the-Loop pings and structured options
├── research_docs/             - Knowledge base for the Research Agent
├── results.tsv                - Experiment log (git commit, metric, VRAM, status, description)
├── state.json                 - Runtime state (error_counter, iteration, current_best)
├── ERROR.txt                  - Cleared each run; written on crash by wrapper.sh
├── HUMAN_INSTRUCTION.txt      - Written by the dashboard; polled by the orchestrator
├── orchestrator.py            - The "Loop Forever" execution engine
├── wrapper.sh                 - Runs train.py, handles ERROR.txt and results.tsv writes
├── prepare.py                 - Data loading, parcellation, and HRF correction (EDITABLE)
├── train.py                   - Model architecture, training loop, and optimizer (EDITABLE)
├── evaluate.py                - Centralized metric calculation (R², ERP recovery, val_bpb)
├── program.md                 - Swarm instructions and high-level research goals
├── pyproject.toml             - Dependencies (PyTorch, MNE-Python, Nilearn, etc.)
└── .python-version            - Pinned Python version (3.10+)
```

---

## Table of Contents

Jump to a file's description:

- [orchestrator.py](#orchestrator-py)
- [wrapper.sh](#wrapper-sh)
- [program.md](#program-md)
- [prepare.py](#prepare-py)
- [train.py](#train-py)
- [research_docs/](#research-docs)
- [evaluate.py](#evaluate-py)
- [coding_agent.md](#coding-agent-md)
- [research_agent.md](#research-agent-md)
- [contact_agent.md](#contact-agent-md)
- [results.tsv](#results-tsv)
- [state.json](#state-json)
- [ERROR.txt](#error-txt)
- [HUMAN_INSTRUCTION.txt](#human-instruction-txt)
- [pyproject.toml](#pyproject-toml)
- [.cache/](#cache-dir)
- [.python-version](#python-version)

Also: [Code structure (tree)](#code-structure)

---

## Detailed file descriptions

### 1. The core execution engine

<a id="orchestrator-py"></a>

**orchestrator.py** — This script replaces the manual loop. It manages the git state (committing or reverting via `git checkout`), delegates runs to `wrapper.sh`, and implements the "LOOP FOREVER" logic. At the start of every iteration it reads `state.json` and `results.tsv` to decide whether to call the Coding Agent with an error, call it fresh for a new optimization pass, or escalate to the Contact Agent. It also polls `HUMAN_INSTRUCTION.txt` at the top of each loop to check whether a human has provided a manual override from the dashboard.

<a id="wrapper-sh"></a>

**wrapper.sh** — The execution boundary between the orchestrator and the training run. It clears `ERROR.txt` at the start of every invocation, runs `train.py`, and captures stdout and stderr. On success, it parses the target metric from the run log and appends a new row to `results.tsv`. On failure, it writes the stderr output to `ERROR.txt`, appends a `CRASH` row to `results.tsv`, and exits with a non-zero code. This is the only file that writes to `ERROR.txt` and the only file that directly invokes `train.py`. The metric parsing line is the single experiment-specific piece — swapping it is what makes the framework plug-and-play across different tasks.

<a id="program-md"></a>

**program.md** — The master instruction file. It defines the "Research Org" rules, the fixed time budget per experiment, and the "Simplicity Criterion" (simpler code is preferred). It is the only file the human edits to reprogram the behaviour of the entire swarm.

---

### 2. The editable pipeline scripts

<a id="prepare-py"></a>

**prepare.py** — Editable by the Coding Agent. In simple experiments, it loads CSVs (Iris, Diabetes). In neuroscience, it handles the TRIBE interface, converts voxels to parcels via parcellation algorithms (e.g., Glasser/HCP), and performs HRF alignment to correct for the 5–6 second hemodynamic delay before passing data to `train.py`.

<a id="train-py"></a>

**train.py** — Editable by the Coding Agent. It contains the model architecture (e.g., a 1D U-Net or Transformer for EEG synthesis) and the optimizer. The agent can modify hyperparameters, add or remove layers, or replace the entire model stack. This is the primary target for optimization across all experiments.

---

### 3. Knowledge & evaluation

<a id="research-docs"></a>

**research_docs/** — A folder containing PDFs and Markdown summaries that form the Research Agent's knowledge base. For neuroscience, this includes the TRIBE v2 paper, THINGS-EEG2 documentation, and MNE-Python guides. The Research Agent queries this folder to prevent the Coding Agent from proposing biologically impossible changes.

<a id="evaluate-py"></a>

**evaluate.py** — Provides the fixed ground-truth metric. It is **not editable by the Coding Agent** — locking this file ensures the agent cannot game the score by softening the evaluation criteria. For Iris, it computes Accuracy; for Neuroscience, it computes a Composite Score including Pearson R (target 0.2–0.6), ERP reconstruction (P300/N170 recovery), and Spatial Consistency Loss.

---

### 4. The agentic swarm (`agents/`)

<a id="coding-agent-md"></a>

**coding_agent.md** — System prompt and tool definitions for the agent that writes code. It is tasked with finding changes to `train.py` or `prepare.py` that improve the score reported by `evaluate.py`. When called with an error context, it attempts to fix the crash before triggering a new run. When called normally, it proposes a novel optimization. At any point during its inner loop it may call the Research Agent for domain-specific guidance before finalising a code change.

<a id="research-agent-md"></a>

**research_agent.md** — System prompt for the contextualizer agent. It does not write code. It reads `research_docs/` and responds to single-turn questions from the Coding Agent with direct, actionable technical insights — for example, "Use a double-gamma HRF kernel for better temporal alignment with EEG." It is designed to be stateless: each call is self-contained and receives all necessary context from the Coding Agent in the prompt.

<a id="contact-agent-md"></a>

**contact_agent.md** — The Human-in-the-Loop interface, triggered by the orchestrator when `state.json`'s `error_counter` exceeds 5 consecutive crashes, or when the model plateaus. It writes a structured flag to the SQLite database (consumed by the dashboard), and optionally sends an email to the team. The flag includes a summary of recent run states and a set of explicit options for the human to choose from (e.g., "A: Increase parcellation granularity, B: Change loss function"). The agent does not modify any code.

---

### 5. Runtime state files

<a id="results-tsv"></a>

**results.tsv** — A tab-separated file that tracks the outcome of every experiment: git commit hash, metric value, VRAM usage, status (`KEEP` / `DISCARD` / `CRASH`), and a short description of the change attempted. This is the primary signal the orchestrator uses to decide whether to commit or revert. It is also the data source for the dashboard's experiment runs page.

<a id="state-json"></a>

**state.json** — A lightweight JSON file storing the orchestrator's runtime state across iterations: `error_counter` (number of consecutive crashes), `iteration` (total loop count), and `current_best` (the best metric value seen so far). The `error_counter` is the trigger for Contact Agent escalation and is incremented by the orchestrator whenever `wrapper.sh` exits with a crash, and reset to zero on any successful run.

<a id="error-txt"></a>

**ERROR.txt** — Cleared at the start of every run by `wrapper.sh`. If `train.py` exits with a non-zero code, `wrapper.sh` writes the captured stderr here. The orchestrator reads this file at the top of each loop: if it is non-empty, the Coding Agent is called with the error as context rather than being asked for a new optimization.

<a id="human-instruction-txt"></a>

**HUMAN_INSTRUCTION.txt** — Written by the dashboard when a human responds to a Contact Agent flag. The orchestrator checks for this file at the top of every loop. If present and non-empty, its contents are injected into the Coding Agent's prompt as a high-priority instruction, overriding the agent's default optimization behaviour for that iteration. It is cleared after being consumed.

---

### 6. Environment & dependencies

<a id="pyproject-toml"></a>

**pyproject.toml** — Manages the full tech stack, including `torch`, `transformers`, and specialised neurotech libraries such as `mne`, `nilearn`, and `scikit-learn`.

<a id="cache-dir"></a>

**`.cache/`** — Local storage for data shards, TRIBE model weights, and tokenizers. Nothing in this directory is tracked by git. See [Code structure](#code-structure) above.

<a id="python-version"></a>

**`.python-version`** — Pinned Python version (3.10+). See [Code structure](#code-structure) above.