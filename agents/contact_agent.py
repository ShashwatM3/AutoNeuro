from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict

AGENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENTS_DIR.parent
DB_PATH = REPO_ROOT / "database.db"
SYSTEM_PROMPT_PATH = AGENTS_DIR / "contact_agent.md"



# LOAD PROMPT
def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return "You are the Contact Agent. Summarize the issue clearly and concisely."


# LLM SUMMARY (Claude)

def generate_summary(context: Dict[str, Any]) -> str:
    system_prompt = load_system_prompt()

    user_prompt = (
        "Summarize the following escalation context for a developer.\n\n"
        f"Context JSON:\n{json.dumps(context, indent=2, sort_keys=True, ensure_ascii=False)}\n\n"
        "Return a concise summary with:\n"
        "- what happened\n"
        "- likely cause\n"
        "- what the developer should check first\n"
    )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[SUMMARY_PENDING_CLAUDE_API] " + json.dumps(
            context, sort_keys=True, ensure_ascii=False
        )

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("Install anthropic with pip install anthropic") from exc

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text.strip())

    return "\n".join(parts).strip() or "[EMPTY_SUMMARY_FROM_CLAUDE]"


# WRITE RUNS TABLE (NEW)
def write_run_to_db(
    iteration: int,
    commit_hash: str,
    status: str,
    metric_value: float,
    diff: str,
    error_log: str | None = None,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO runs (
                iteration,
                commit_hash,
                status,
                metric_value,
                diff,
                error_log,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iteration,
                commit_hash,
                status,
                metric_value,
                diff,
                error_log,
                datetime.now(UTC).isoformat(),
            ),
        )

        conn.commit()
    finally:
        conn.close()



# WRITE FLAGS TABLE
def write_flag_to_db(
    iteration: int,
    trigger_reason: str,
    summary: str,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO flags (
                iteration,
                trigger_reason,
                context_summary,
                human_instruction,
                resolved,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                iteration,
                trigger_reason,
                summary,
                None,
                0,
                datetime.now(UTC).isoformat(),
            ),
        )

        conn.commit()
    finally:
        conn.close()
        print("DB WRITE SUCCESS")


# OPTIONAL EMAIL
def send_email(summary: str) -> None:
    return



# ESCALATION ENTRYPOINT
def escalate(
    trigger_reason: str,
    iteration: int,
    context: Dict[str, Any],
) -> str:
    summary = generate_summary(context)
    write_flag_to_db(iteration, trigger_reason, summary)
    send_email(summary)
    return summary


if __name__ == "__main__":

    print("\n=== TEST: WRITE RUN ===")
    write_run_to_db(
        iteration=1,
        commit_hash="test_commit_abc",
        status="KEEP",
        metric_value=0.92,
        diff="+ improved model layer",
        error_log=None
    )

    print("RUN INSERTED")

    print("\n=== TEST: WRITE RUN (CRASH) ===")
    write_run_to_db(
        iteration=2,
        commit_hash="test_commit_def",
        status="CRASH",
        metric_value=0.31,
        diff="+ broken training step",
        error_log="ValueError: NaN loss detected"
    )

    print("CRASH RUN INSERTED")

    print("\n=== TEST: ESCALATION ===")
    escalate(
        trigger_reason="ERROR_LIMIT",
        iteration=2,
        context={
            "metric": 0.31,
            "error": "NaN loss detected",
            "system": "training_loop_v1"
        }
    )

    print("\nALL TESTS COMPLETED")