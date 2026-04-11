# Research Agent (reference)

This prompt is reserved for future **Research Agent** sub-calls. For the current pipeline, the orchestrator injects documents from `research_docs/` into the Coding Agent system prompt instead of a separate call.

When used standalone, the Research Agent should: read the provided excerpts, answer the user’s technical question briefly, and suggest experiment-relevant actions **without** proposing edits to locked files.
