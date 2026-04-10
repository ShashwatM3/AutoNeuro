## Code Structure

```
.
├── .cache/                 
│     └── (Local storage for data shards, TRIBE weights, and tokenizers)
├── agents/                 
│     ├── coding_agent.md       (System prompts/tools for modifying code)
│     ├── research_agent.md     (System prompts for analyzing papers and providing context)
│     └── contact_agent.md      (Logic for Human-in-the-Loop pings and structured options)
├── research_docs/           (Knowledge base for the Research Agent)
├── results.tsv              (Experiment log: git commit, metric, VRAM, status, description)
├── orchestrator.py          ("Loop Forever" execution engine)
├── prepare.py               (Data loading, parcellation, and HRF correction — EDITABLE)
├── train.py                 (Model architecture, training loop, and optimizer — EDITABLE)
├── evaluate.py              (Centralized metric calculation: R², ERP recovery, val_bpb)
├── program.md               (Swarm instructions and high-level research goals)
├── pyproject.toml           (Dependencies: PyTorch, MNE-Python, Nilearn, etc.)
└── .python-version          (Pinned Python version: 3.10+)
```