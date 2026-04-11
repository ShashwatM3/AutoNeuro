from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from db import DB_PATH, init_db

AGENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENTS_DIR.parent
load_dotenv(REPO_ROOT / ".env")
SYSTEM_PROMPT_PATH = AGENTS_DIR / "contact_agent.md"


def load_system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return "You are the Contact Agent. Summarize the issue clearly and concisely."


def generate_summary(context: Dict[str, Any]) -> str:
    init_db()
    system_prompt = load_system_prompt()
    user_prompt = (
        "Summarize the following escalation context for a developer.\n\n"
        f"Context JSON:\n{json.dumps(context, indent=2, sort_keys=True, ensure_ascii=False)}\n\n"
        "Return a concise summary with:\n"
        "- what happened\n"
        "- likely cause\n"
        "- what the developer should check first\n"
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[SUMMARY_PENDING_OPENAI_API] " + json.dumps(
            context, sort_keys=True, ensure_ascii=False
        )

    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    return text or "[EMPTY_SUMMARY_FROM_OPENAI]"


def write_run_to_db(
    iteration: int,
    commit_hash: str,
    status: str,
    metric_value: float | None,
    diff: str,
    error_log: str | None = None,
) -> None:
    init_db()
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


def write_flag_to_db(
    iteration: int,
    trigger_reason: str,
    summary: str,
) -> None:
    init_db()
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


def send_email(summary: str) -> None:
    return


def escalate(
    trigger_reason: str,
    iteration: int,
    context: Dict[str, Any],
) -> str:
    init_db()
    summary = generate_summary(context)
    write_flag_to_db(iteration, trigger_reason, summary)
    send_email(summary)
    return summary
