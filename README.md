# AutoNeuro

## Code Structure

<a id="code-structure"></a>

```
├── .cache/                    - Local storage for data shards, TRIBE weights, and tokenizers 
├── agents/                     - The Agentic Swarm Logic
│   ├── coding_agent.md        - System prompts/tools for modifying code
│   ├── research_agent.md      - System prompts for analyzing papers and providing context
│   └── contact_agent.md       - Logic for Human-in-the-Loop pings and structured options
├── research_docs/              - Knowledge base for the Research Agent
├── results.tsv                 - Experiment log (git commit, metric, VRAM, status, description)
├── orchestrator.py             - The "Loop Forever" execution engine
├── prepare.py                  - Data loading, parcellation, and HRF correction (EDITABLE)
├── train.py                    - Model architecture, training loop, and optimizer (EDITABLE)
├── evaluate.py                 - Centralized metric calculation (R², ERP recovery, val_bpb)
├── program.md                  - Swarm instructions and high-level research goals
├── pyproject.toml              - Dependencies (PyTorch, MNE-Python, Nilearn, etc.)
└── .python-version             - Pinned Python version (3.10+)
```

---

## Table of Contents

Jump to a file’s description:

- [orchestrator.py](#orchestrator-py)
- [program.md](#program-md)
- [prepare.py](#prepare-py)
- [train.py](#train-py)
- [research_docs/](#research-docs)
- [evaluate.py](#evaluate-py)
- [coding_agent.md](#coding-agent-md)
- [research_agent.md](#research-agent-md)
- [contact_agent.md](#contact-agent-md)
- [results.tsv](#results-tsv)
- [pyproject.toml](#pyproject-toml)
- [.cache/](#cache-dir)
- [.python-version](#python-version)

Also: [Code structure (tree)](#code-structure)

---

## Detailed file descriptions

### 1. The core execution engine

<a id="orchestrator-py"></a>

**orchestrator.py** — This script replaces the manual loop. It manages the git state (branching/committing), launches `prepare.py` and `train.py`, and implements the “LOOP FOREVER” logic. It monitors for crashes and decides whether to “Advance” or “Reset” the branch based on the metrics in `results.tsv`.

<a id="program-md"></a>

**program.md** — The master instruction file. It defines the “Research Org” rules, the 5-minute time budget per experiment, and the “Simplicity Criterion” (simpler code is better).

### 2. The editable pipeline scripts

<a id="prepare-py"></a>

**prepare.py** — Editable by the Coding Agent. In simple experiments, it loads CSVs (Iris). In neuroscience, it handles the TRIBE interface, converts voxels to parcels, and performs HRF alignment to correct for the 5-second hemodynamic delay.

<a id="train-py"></a>

**train.py** — Editable by the Coding Agent. It contains the model architecture (e.g., 1D U-Net or Transformer for EEG) and the optimizer. The agent can modify hyperparameters, add layers, or change the entire model stack.

### 3. Knowledge & evaluation

<a id="research-docs"></a>

**research_docs/** — A folder containing PDFs and Markdown summaries. For neuroscience, this includes the TRIBE v2 paper, THINGS-EEG2 documentation, and MNE-Python guides. The Research Agent queries this folder to prevent the Coding Agent from suggesting biologically impossible changes.

<a id="evaluate-py"></a>

**evaluate.py** — Provides the fixed ground truth metric. For Iris, it’s Accuracy; for Neuroscience, it’s a Composite Score including Pearson R (target 0.2–0.6), ERP reconstruction (P300/N170), and Spatial Consistency Loss.

### 4. The agentic swarm (`agents/`)

<a id="coding-agent-md"></a>

**coding_agent.md** — Instructions for the agent that writes code. It is tasked with generating the lowest possible error in `evaluate.py` by modifying `train.py` or `prepare.py`.

<a id="research-agent-md"></a>

**research_agent.md** — Contextualizer. It doesn’t write code but analyzes `research_docs/` to suggest “Ideas” (e.g., “Use a double-gamma HRF for better temporal alignment”).

<a id="contact-agent-md"></a>

**contact_agent.md** — The Human-in-the-Loop interface. If the model plateaus or hits a “circular logic” error, it messages the team with options (e.g., “A: Increase parcellation, B: Change Loss Function”).

### 5. Tracking & environment

<a id="results-tsv"></a>

**results.tsv** — A tab-separated file used to track the progress of every experiment. This is the only file the agents use to “see” if their changes worked.

<a id="pyproject-toml"></a>

**pyproject.toml** — Manages the tech stack, including torch, transformers, and specialized neurotech libraries like mne and nilearn.

<a id="cache-dir"></a>

**`.cache/`** — Local storage for data shards, TRIBE weights, and tokenizers (see [Code structure](#code-structure) above).

<a id="python-version"></a>

**`.python-version`** — Pinned Python version (3.10+); see [Code structure](#code-structure) above.
