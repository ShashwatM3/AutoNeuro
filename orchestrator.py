"""
Main loop for AutoNeuro: loads state, calls the coding agent (OpenAI), runs wrapper.sh,
syncs runs to SQLite for the dashboard, and manages git keep/discard.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents.contact_agent import escalate, write_run_to_db
from db import init_db

REPO_ROOT = Path(__file__).resolve().parent
STATE_PATH = REPO_ROOT / "state.json"
RESULTS_PATH = REPO_ROOT / "results.tsv"
ERROR_PATH = REPO_ROOT / "ERROR.txt"
HUMAN_INSTRUCTION_PATH = REPO_ROOT / "HUMAN_INSTRUCTION.txt"
WRAPPER_SH = REPO_ROOT / "wrapper.sh"
CODING_AGENT_MD = REPO_ROOT / "agents" / "coding_agent.md"
PROGRAM_MD = REPO_ROOT / "program.md"


def load_state() -> dict[str, Any]:
    data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "error_counter": int(data["error_counter"]),
        "iteration": int(data["iteration"]),
        "current_best": data["current_best"],
    }


def save_state(state: dict[str, Any]) -> None:
    out = {
        "error_counter": state["error_counter"],
        "iteration": state["iteration"],
        "current_best": state["current_best"],
    }
    STATE_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def parse_results_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < len(header):
            parts.append("")
        rows.append(dict(zip(header, parts)))
    return rows


def metric_higher_is_better() -> bool:
    if not PROGRAM_MD.exists():
        return True
    text = PROGRAM_MD.read_text(encoding="utf-8").lower()
    if "lower is better" in text or "lower metric is better" in text:
        return False
    return True


def metric_improved(last_m: str, prev_m: str, higher_better: bool) -> bool:
    try:
        la = float(last_m)
        pr = float(prev_m)
    except ValueError:
        return False
    if math.isnan(la) or math.isnan(pr):
        return False
    if higher_better:
        return la > pr
    return la < pr


def parse_metric_float(s: str) -> float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v


def update_last_row_status(new_status: str) -> None:
    lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return
    header = lines[0].split("\t")
    try:
        idx_status = header.index("status")
    except ValueError:
        idx_status = 3
    last_idx = len(lines) - 1
    parts = lines[last_idx].split("\t")
    while len(parts) <= idx_status:
        parts.append("")
    parts[idx_status] = new_status
    lines[last_idx] = "\t".join(parts)
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_human_instruction_clear() -> str | None:
    if not HUMAN_INSTRUCTION_PATH.exists():
        return None
    raw = HUMAN_INSTRUCTION_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    HUMAN_INSTRUCTION_PATH.write_text("", encoding="utf-8")
    return raw


def wait_for_human_instruction(poll_s: float = 30.0) -> str:
    while True:
        if HUMAN_INSTRUCTION_PATH.exists():
            txt = HUMAN_INSTRUCTION_PATH.read_text(encoding="utf-8").strip()
            if txt:
                HUMAN_INSTRUCTION_PATH.write_text("", encoding="utf-8")
                return txt
        time.sleep(poll_s)


def load_research_docs_section(max_chars: int = 120_000) -> str:
    rd = REPO_ROOT / "research_docs"
    if not rd.is_dir():
        return "(No research_docs directory.)"
    chunks: list[str] = []
    total = 0
    for fp in sorted(rd.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".md", ".txt", ".markdown"}:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        piece = f"\n\n### {fp.relative_to(REPO_ROOT)}\n\n{text}"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    if not chunks:
        return "(No readable .md/.txt files in research_docs.)"
    return "".join(chunks)


def build_coding_system_prompt() -> str:
    base = CODING_AGENT_MD.read_text(encoding="utf-8")
    injected = load_research_docs_section()
    return base + "\n\n## Reference Documents (injected)\n\n" + injected


def _read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def _human_suffix(human_instruction: str | None) -> str:
    if human_instruction:
        return f'\n\nHUMAN INSTRUCTION (high priority): {human_instruction}\n'
    return ""


def call_coding_agent(
    mode: str,
    error: str | None = None,
    human_instruction: str | None = None,
) -> None:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    system_prompt = build_coding_system_prompt()
    train_code = _read_text(REPO_ROOT / "train.py")
    prepare_code = _read_text(REPO_ROOT / "prepare.py")
    results = _read_text(RESULTS_PATH)
    program = _read_text(PROGRAM_MD, "(missing program.md)")

    if mode == "FIRST_RUN":
        user_msg = f"""This is the very first iteration. Here is the current codebase.

program.md:
{program}

train.py:
{train_code}

prepare.py:
{prepare_code}

Do not change anything yet. Just confirm you have read the files and are ready.{_human_suffix(human_instruction)}"""

    elif mode == "SECOND_RUN":
        user_msg = f"""This is only the second iteration. We have one result but no comparison yet.
You are free to make one change you think will improve the metric.

program.md:
{program}

train.py:
{train_code}

prepare.py:
{prepare_code}

results.tsv:
{results}

Make exactly one change. Write the full updated content of whichever file(s) you modify.{_human_suffix(human_instruction)}"""

    elif mode == "FIX_ERROR":
        user_msg = f"""The last run crashed. Fix the error.

ERROR:
{error}

train.py:
{train_code}

prepare.py:
{prepare_code}

Write the full corrected file(s).{_human_suffix(human_instruction)}"""

    elif mode == "OPTIMIZE":
        user_msg = f"""Propose and implement one optimization to improve the metric.

program.md:
{program}

train.py:
{train_code}

prepare.py:
{prepare_code}

results.tsv:
{results}

{_human_suffix(human_instruction).strip()}

Write the full updated content of whichever file(s) you modify."""

    else:
        raise ValueError(f"Unknown mode: {mode}")

    response = client.chat.completions.create(
        model=model,
        max_tokens=8096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()

    pattern = re.compile(
        r'<file name="(train\.py|prepare\.py)">(.*?)</file>',
        re.DOTALL,
    )
    for match in pattern.finditer(raw):
        fname, content = match.group(1), match.group(2).strip()
        (REPO_ROOT / fname).write_text(content + ("\n" if content and not content.endswith("\n") else ""), encoding="utf-8")
        print(f"[orchestrator] wrote {fname}")


def _run_git(args: list[str]) -> None:
    subprocess.run(["git", "-C", str(REPO_ROOT)] + args, check=True)


def git_commit_keep(iteration: int) -> None:
    _run_git(["add", "train.py", "prepare.py", "results.tsv"])
    _run_git(["commit", "-m", f"iteration {iteration}: KEEP"])


def git_checkout_discard() -> None:
    _run_git(["checkout", "--", "train.py", "prepare.py"])


def git_diff_train_prepare() -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "HEAD", "--", "train.py", "prepare.py"],
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "").strip()
    return out if out else "(no diff vs HEAD)"


def sync_latest_run_to_db(state: dict[str, Any], last_row: dict[str, str]) -> None:
    diff = git_diff_train_prepare()
    err_log: str | None = None
    if last_row.get("status") == "CRASH" and ERROR_PATH.exists():
        err_log = ERROR_PATH.read_text(encoding="utf-8", errors="replace")

    mv = parse_metric_float(last_row.get("metric", "nan"))
    write_run_to_db(
        iteration=state["iteration"] + 1,
        commit_hash=last_row.get("commit_hash", ""),
        status=last_row.get("status", ""),
        metric_value=mv,
        diff=diff,
        error_log=err_log,
    )


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    init_db()
    state = load_state()

    while True:
        human_instruction = read_human_instruction_clear()
        rows = parse_results_rows()
        n = len(rows)
        higher = metric_higher_is_better()

        if n == 0:
            call_coding_agent("FIRST_RUN", human_instruction=human_instruction)
        elif n == 1:
            call_coding_agent("SECOND_RUN", human_instruction=human_instruction)
        else:
            err_nonempty = (
                ERROR_PATH.exists()
                and ERROR_PATH.read_text(encoding="utf-8", errors="replace").strip() != ""
            )
            if err_nonempty:
                error_text = ERROR_PATH.read_text(encoding="utf-8", errors="replace")
                if state["error_counter"] >= 5:
                    escalate(
                        trigger_reason="ERROR_LIMIT",
                        iteration=state["iteration"],
                        context={
                            "error_counter": state["error_counter"],
                            "error_log": error_text,
                            "recent_results": rows[-5:],
                            "current_best": state["current_best"],
                        },
                    )
                    instr = wait_for_human_instruction(30.0)
                    state["error_counter"] = 0
                    call_coding_agent(
                        "FIX_ERROR",
                        error=error_text,
                        human_instruction=instr or human_instruction,
                    )
                else:
                    call_coding_agent(
                        "FIX_ERROR",
                        error=error_text,
                        human_instruction=human_instruction,
                    )
            else:
                last = rows[-1]
                prev = rows[-2]
                if metric_improved(last["metric"], prev["metric"], higher):
                    try:
                        git_commit_keep(state["iteration"])
                    except subprocess.CalledProcessError as exc:
                        print(f"[orchestrator] git commit failed: {exc}")
                    update_last_row_status("KEEP")
                    lm = parse_metric_float(last["metric"])
                    if lm is not None:
                        cb = state["current_best"]
                        if higher:
                            state["current_best"] = max(cb if cb is not None else lm, lm)
                        else:
                            state["current_best"] = min(cb if cb is not None else lm, lm)
                else:
                    try:
                        git_checkout_discard()
                    except subprocess.CalledProcessError as exc:
                        print(f"[orchestrator] git checkout failed: {exc}")
                    update_last_row_status("DISCARD")
                call_coding_agent("OPTIMIZE", human_instruction=human_instruction)

        subprocess.run(["bash", str(WRAPPER_SH)], cwd=str(REPO_ROOT), check=False)

        rows_after = parse_results_rows()
        if not rows_after:
            print("[orchestrator] WARNING: no data rows in results.tsv after wrapper")
        else:
            sync_latest_run_to_db(state, rows_after[-1])

        last_row = rows_after[-1] if rows_after else None
        if last_row and last_row.get("status") == "CRASH":
            state["error_counter"] += 1
        else:
            state["error_counter"] = 0

        state["iteration"] += 1
        save_state(state)


if __name__ == "__main__":
    main()
