# AutoNeuro — Complete Project Context

## What is this project?

AutoNeuro is a **generalised autonomous ML research pipeline** inspired by Andrej Karpathy's
AutoResearch tool. It runs a continuous loop in which Claude AI agents iteratively
modify a machine learning training script, execute it, observe the result, and decide
whether to keep or discard the change — all without human intervention, except in
escalation scenarios.

The system is designed to be **plug-and-play**: swap three files (`train.py`, `prepare.py`,
`program.md`) and the entire infrastructure adapts to any ML experiment.

The ultimate scientific target is a **Brain-Conditioned Generative Model of EEG signals**
for a Neuroscience Hackathon. The agentic pipeline is being developed and validated
on progressively harder proxy experiments first (Iris → Diabetes → ECG) before
being applied to the neuroscience problem.

---

## The three experiments (in increasing difficulty)

| # | Name | Task | Metric | Difficulty |
|---|------|------|--------|------------|
| 1 | Iris Classification | 3-class tabular | Accuracy | Easy — validates the loop |
| 2 | Diabetes Regression | Continuous regression | R² | Medium — noisy target, regularisation matters |
| 3 | ECG Arrhythmia | 5-class time-series (1D CNN) | Macro-F1 | Hard — class imbalance, temporal structure |
| 4 | EEG Synthesis (final) | Conditional EEG generation | Pearson R + ERP | Research-grade |

---

## The neuroscience end goal

The final experiment models the conditional distribution **P(EEG | stimulus, brain_state_prior)**
using a 7-stage pipeline:

1. **Stimulus** (video / audio / text)
2. **TRIBE v2** — tri-modal foundation model predicts ~70,000 fMRI voxel activations
3. **Voxel Processing** — high-dimensional brain state representation
4. **Parcellation** — Glasser/HCP atlas groups voxels into functional parcels
5. **Brain Latent Encoder** — compresses parcel space + learns HRF correction to align
   slow fMRI (1–3s resolution, 5–6s delay) with fast EEG (1ms resolution)
6. **Diffusion Model** — 1D U-Net or transformer synthesises multi-channel EEG
7. **Synthetic 64-channel EEG** — output evaluated on Pearson R, ERP recovery (P300, N170),
   and Spatial Consistency Loss

Dataset: **THINGS-EEG2** (1,854 object images, 50 subjects)

---

## Repository structure

```
autoneuro/
├── orchestrator.py            ← THE MAIN LOOP — do not modify
├── wrapper.sh                 ← runs train.py, writes results — do not modify
├── db.py                      ← SQLite schema init — do not modify
├── train.py                   ← EXPERIMENT-SPECIFIC — coding agent modifies this
├── prepare.py                 ← EXPERIMENT-SPECIFIC — coding agent modifies this
├── evaluate.py                ← LOCKED — coding agent must never touch this
├── program.md                 ← EXPERIMENT-SPECIFIC — human writes this
├── results.tsv                ← tab-separated experiment log (git-tracked)
├── state.json                 ← runtime state: error_counter, iteration, current_best
├── ERROR.txt                  ← written on crash by wrapper.sh, cleared each run
├── HUMAN_INSTRUCTION.txt      ← written by dashboard when human responds to a flag
├── agents/
│   ├── coding_agent.md        ← system prompt for the coding agent
│   ├── research_agent.md      ← system prompt for the research agent
│   ├── contact_agent.md       ← system prompt for the contact agent
│   └── contact_agent.py       ← Python: escalate(), write_flag_to_db(), generate_summary()
├── dashboard/
│   ├── server.py              ← Flask API + static file server
│   ├── index.html             ← Experiment runs page (KEEP/DISCARD/CRASH cards)
│   └── flags.html             ← Human-in-the-loop flags page
└── research_docs/             ← PDFs and markdown papers for the research agent
```

---

## The orchestration loop (exact logic)

```
STARTUP:
  init_db()   # create SQLite tables if not exist
  load state.json

LOOP FOREVER:

  [1] Check HUMAN_INSTRUCTION.txt
      → if non-empty: read it, clear it, inject into next coding agent call as high-priority override

  [2] Read results.tsv → get row count N

  [3] Decide context for coding agent:
      if N == 0:  mode = FIRST_RUN   (no changes yet, just read files)
      if N == 1:  mode = SECOND_RUN  (1 result, no comparison possible, make one change)
      if N >= 2:
        if ERROR.txt is non-empty:
          error_counter >= 5 → call contact_agent (ESCALATE), wait for human
          else               → mode = FIX_ERROR
        else:
          if results[-1].metric > results[-2].metric:
            git commit (KEEP)
            update current_best
          else:
            git checkout -- train.py prepare.py  (DISCARD)
          mode = OPTIMIZE

  [4] Call coding agent via Anthropic Python SDK
      → agent reads train.py, prepare.py, results.tsv, program.md
      → agent optionally calls research agent (implemented as a second API call)
      → agent returns modified file(s) wrapped in <file name="..."> tags
      → orchestrator parses tags and writes files

  [5] Run: bash wrapper.sh
      → wrapper clears ERROR.txt
      → runs python train.py
      → on success: parses METRIC= and VRAM_MB= from stdout → appends to results.tsv
      → on crash: writes stderr to ERROR.txt → appends CRASH row → exits 1

  [6] Sync latest run to database.db (for dashboard)

  [7] Update state.json:
      error_counter = error_counter+1 if CRASH else 0
      iteration += 1
      current_best = max(current_best, metric) if not CRASH
```

---

## The three agents

### Coding Agent
- **Called by**: orchestrator
- **Reads**: `train.py`, `prepare.py`, `results.tsv`, `program.md`, `agents/coding_agent.md`
- **Writes**: modified `train.py` and/or `prepare.py` (via `<file>` tag output)
- **May call**: research agent (second API call with a question + research_docs content)
- **Constraint**: one logical change per iteration; never touches locked files
- **Model**: `claude-sonnet-4-20250514`, max_tokens=8096

### Research Agent
- **Called by**: coding agent (via orchestrator making a sub-call)
- **Reads**: contents of `research_docs/` + question from coding agent
- **Writes**: nothing (read-only, single-turn, advisory)
- **Output**: direct technical insight — e.g. "Use double-gamma HRF kernel for better alignment"
- **Model**: `claude-sonnet-4-20250514`, max_tokens=2048

### Contact Agent
- **Called by**: orchestrator when `error_counter >= 5` or plateau detected
- **Reads**: recent run history, error logs
- **Writes**: flag row to `database.db` (via `agents/contact_agent.py`)
- **Output**: structured summary + A/B/C options for human
- **Loop unblocks when**: human types instruction into dashboard → writes `HUMAN_INSTRUCTION.txt`

---

## Data flow: what reads/writes what

| File | Written by | Read by | Purpose |
|------|-----------|---------|---------|
| `results.tsv` | `wrapper.sh` | `orchestrator.py`, coding agent | Decision log, git-tracked |
| `state.json` | `orchestrator.py` | `orchestrator.py` | Loop counters only |
| `ERROR.txt` | `wrapper.sh` | `orchestrator.py`, coding agent | Crash details |
| `HUMAN_INSTRUCTION.txt` | `dashboard/server.py` | `orchestrator.py` | HITL override |
| `database.db` / `runs` | `orchestrator.py` | `dashboard/server.py` | Dashboard UI |
| `database.db` / `flags` | `agents/contact_agent.py` | `dashboard/server.py` | Dashboard flags |
| `train.py` | coding agent | `wrapper.sh`, coding agent | Model code |
| `prepare.py` | coding agent | `train.py`, coding agent | Data pipeline |

**Critical rule**: `orchestrator.py` never reads `database.db`. `dashboard/server.py` never reads
`results.tsv` or `state.json` directly. These boundaries must not be crossed.

---

## The dashboard

Two pages, served by Flask on `localhost:5000`:

### Runs page (`index.html`)
- One card per experiment iteration
- Green border = KEEP (metric improved)
- Yellow border = DISCARD (metric did not improve, change was reverted)
- Red border = CRASH (train.py threw an error)
- Shows: iteration number, metric value, commit hash, code diff, timestamp
- Auto-refreshes every 10 seconds

### Flags page (`flags.html`)
- Appears when contact agent escalates (error_counter ≥ 5 or plateau)
- Shows: iteration, trigger reason (ERROR_LIMIT / PLATEAU), AI-generated summary
- Human can type an instruction into the text box → submits to `/api/flags/<id>/respond`
- This writes `HUMAN_INSTRUCTION.txt`, which unblocks the orchestrator loop
- "AI Summarize" button re-generates the summary via the contact agent

---

## How to adapt to a new experiment

Edit exactly these three files:

1. **`train.py`**: implement your model + training loop. End with:
   ```python
   print(f"METRIC={your_metric:.6f}", flush=True)
   print(f"VRAM_MB={vram_mb}", flush=True)
   ```

2. **`prepare.py`**: implement data loading/preprocessing. Save arrays to `.cache/`.

3. **`program.md`**: describe the experiment goal, metric name, metric direction
   (HIGHER IS BETTER or LOWER IS BETTER), what the agent may/may not change,
   and any domain notes for the research agent.

Then run:
```bash
python orchestrator.py       # starts the loop
python dashboard/server.py   # start dashboard in a second terminal
```

---

## API usage

All agent calls use the Anthropic Python SDK:

```python
import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8096,
    system=system_prompt,
    messages=[{"role": "user", "content": user_message}]
)
text = response.content[0].text
```

Required env var: `ANTHROPIC_API_KEY`

---

## Git strategy

- The working tree is always the "candidate" being tested.
- Files are only committed AFTER a successful run that improves the metric.
- If a run crashes or metric regresses: `git checkout -- train.py prepare.py` discards the changes.
- `results.tsv` is always committed regardless (it's the audit log).
- Commit message format: `"iteration {N}: {KEEP|DISCARD|CRASH} — {short description}"`

---

## Current implementation status

| Component | Status |
|-----------|--------|
| `wrapper.sh` | ✅ Complete |
| `agents/contact_agent.py` | ✅ Mostly complete (needs init_db wiring) |
| `dashboard/flags.html` | ✅ Complete |
| `dashboard/server.py` | ✅ Mostly complete |
| `orchestrator.py` | 🔲 Stub — needs full loop implementation |
| `db.py` | 🔲 Missing — needs to be created |
| `agents/coding_agent.md` | 🔲 Placeholder — needs real system prompt |
| `agents/research_agent.md` | 🔲 Placeholder — needs real system prompt |
| `agents/contact_agent.md` | 🔲 Placeholder — needs real system prompt |
| `dashboard/index.html` | 🔲 Needs `/api/runs` integration + colour-coded cards |
| `evaluate.py` | 🔲 Stub — experiment-specific, locked from agent |

---

## Dependencies

```toml
[project]
dependencies = [
    "flask",
    "anthropic>=0.40.0",
    "numpy",
    "scikit-learn",
    "torch",          # experiment 3+
]
```

---

## What "plug and play" means in practice

Once the orchestrator is fully implemented, a researcher should be able to:

1. Git clone the repo
2. Set `ANTHROPIC_API_KEY` env var
3. Write their `train.py`, `prepare.py`, `program.md`
4. Run `python orchestrator.py`
5. Open `localhost:5000` in a browser
6. Come back hours later to a log of experiments, a best model, and any human flags to review

No other files need to be touched. The infrastructure is completely generic.