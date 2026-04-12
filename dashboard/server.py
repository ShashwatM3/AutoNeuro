#!/usr/bin/env python3
"""
Flask backend server for AutoNeuro dashboard.
Serves the Runs and Flags pages plus JSON API routes for the same data.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add the parent directory to Python path to import contact_agent and db
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.contact_agent import generate_summary
from db import init_db

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Database path - one level up from dashboard directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

DB_PATH = _REPO_ROOT / "database.db"
HUMAN_INSTRUCTION_PATH = _REPO_ROOT / "HUMAN_INSTRUCTION.txt"
ORCHESTRATOR_PID_PATH = _REPO_ROOT / ".orchestrator.pid"
init_db()


def get_db_connection():
    """Get SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables dict-like access to rows
    return conn


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_matches_orchestrator(pid: int) -> bool:
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    cmd = (proc.stdout or "").strip()
    return "orchestrator.py" in cmd


def _stop_orchestrator_process() -> tuple[bool, str]:
    if not ORCHESTRATOR_PID_PATH.exists():
        return False, "No active orchestrator PID file found."
    raw = ORCHESTRATOR_PID_PATH.read_text(encoding="utf-8").strip()
    try:
        pid = int(raw)
    except ValueError:
        ORCHESTRATOR_PID_PATH.unlink(missing_ok=True)
        return False, "Invalid orchestrator PID file removed."

    if not _is_pid_alive(pid):
        ORCHESTRATOR_PID_PATH.unlink(missing_ok=True)
        return False, f"Orchestrator PID {pid} not running; removed stale PID file."

    if not _pid_matches_orchestrator(pid):
        return False, f"PID {pid} does not look like orchestrator.py; refusing to stop."

    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            ORCHESTRATOR_PID_PATH.unlink(missing_ok=True)
            return True, f"Stopped orchestrator PID {pid}."
        time.sleep(0.1)

    os.kill(pid, signal.SIGKILL)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            ORCHESTRATOR_PID_PATH.unlink(missing_ok=True)
            return True, f"Stopped orchestrator PID {pid} (forced)."
        time.sleep(0.1)

    return False, f"Failed to stop orchestrator PID {pid}."


# Static file serving for HTML pages
@app.route('/')
def index():
    """Serve the experiment runs page."""
    return send_from_directory('.', 'index.html')


@app.route('/flags')
def flags():
    """Serve the Flags page (flags.html)."""
    return send_from_directory('.', 'flags.html')


# API Endpoints

@app.route('/api/runs', methods=['GET'])
def get_runs():
    """Get all experiment runs from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Order by iteration descending to show most recent first
        cursor.execute("""
            SELECT 
                id, iteration, commit_hash, status, metric_value, 
                diff, error_log, timestamp
            FROM runs 
            ORDER BY iteration DESC
        """)
        
        rows = cursor.fetchall()
        
        # Convert rows to list of dictionaries
        runs = []
        for row in rows:
            run = {
                'id': row['id'],
                'iteration': row['iteration'],
                'commit_hash': row['commit_hash'],
                'status': row['status'],
                'metric_value': row['metric_value'],
                'diff': row['diff'],
                'error_log': row['error_log'],
                'timestamp': row['timestamp']
            }
            runs.append(run)
        
        conn.close()
        return jsonify(runs)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/flags', methods=['GET'])
def get_flags():
    """Get all flags from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Order by timestamp descending to show most recent first
        cursor.execute("""
            SELECT 
                id, iteration, trigger_reason, context_summary,
                human_instruction, resolved, timestamp
            FROM flags 
            ORDER BY timestamp DESC
        """)
        
        rows = cursor.fetchall()
        
        # Convert rows to list of dictionaries
        flags = []
        for row in rows:
            flag = {
                'id': row['id'],
                'iteration': row['iteration'],
                'trigger_reason': row['trigger_reason'],
                'context_summary': row['context_summary'],
                'human_instruction': row['human_instruction'],
                'resolved': bool(row['resolved']),  # Convert to boolean
                'timestamp': row['timestamp']
            }
            flags.append(flag)
        
        conn.close()
        return jsonify(flags)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/flags/<int:flag_id>/respond', methods=['POST'])
def respond_to_flag(flag_id):
    """Handle human response to a flag."""
    try:
        data = request.get_json()
        human_instruction = data.get('human_instruction', '').strip()
        
        if not human_instruction:
            return jsonify({'error': 'Human instruction cannot be empty'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update the flag with human instruction and auto-resolve it
        cursor.execute("""
            UPDATE flags 
            SET human_instruction = ?, resolved = 1 
            WHERE id = ?
        """, (human_instruction, flag_id))
        
        # Check if the flag was actually updated
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Flag not found'}), 404
        
        conn.commit()
        conn.close()
        
        # Write the human instruction to HUMAN_INSTRUCTION.txt for orchestrator
        HUMAN_INSTRUCTION_PATH.write_text(human_instruction, encoding='utf-8')
        
        return jsonify({
            'success': True,
            'message': 'Flag response recorded and resolved automatically'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/flags/<int:flag_id>/resolve', methods=['POST'])
def resolve_flag(flag_id):
    """Explicitly mark a flag as resolved."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Mark the flag as resolved
        cursor.execute("""
            UPDATE flags 
            SET resolved = 1 
            WHERE id = ?
        """, (flag_id,))
        
        # Check if the flag was actually updated
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Flag not found'}), 404
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Flag marked as resolved'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/flags/<int:flag_id>/summarize', methods=['POST'])
def summarize_flag(flag_id):
    """Generate AI summary for a flag using the Contact Agent."""
    try:
        # Get flag details first
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT iteration, trigger_reason, context_summary
            FROM flags 
            WHERE id = ?
        """, (flag_id,))
        
        flag = cursor.fetchone()
        if not flag:
            conn.close()
            return jsonify({'error': 'Flag not found'}), 404
        
        # Get recent runs context for the AI summary
        cursor.execute("""
            SELECT 
                iteration, status, metric_value, error_log, diff
            FROM runs 
            WHERE iteration <= ?
            ORDER BY iteration DESC
            LIMIT 10
        """, (flag['iteration'],))
        
        recent_runs = cursor.fetchall()
        conn.close()
        
        # Build context for AI summary
        context = {
            'flag_id': flag_id,
            'iteration': flag['iteration'],
            'trigger_reason': flag['trigger_reason'],
            'existing_summary': flag['context_summary'],
            'recent_runs': [dict(run) for run in recent_runs]
        }
        
        # Generate new AI summary using Contact Agent
        new_summary = generate_summary(context)
        
        # Update the flag with the new summary
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE flags 
            SET context_summary = ?
            WHERE id = ?
        """, (new_summary, flag_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'AI summary generated successfully',
            'new_summary': new_summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/exit', methods=['POST'])
def dashboard_exit():
    """Export database payload, clear dashboard tables, and stop orchestrator."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")

        cursor.execute(
            """
            SELECT id, iteration, commit_hash, status, metric_value, diff, error_log, timestamp
            FROM runs
            ORDER BY iteration DESC, id DESC
            """
        )
        runs = _rows_to_dicts(cursor.fetchall())

        cursor.execute(
            """
            SELECT id, iteration, trigger_reason, context_summary, human_instruction, resolved, timestamp
            FROM flags
            ORDER BY timestamp DESC, id DESC
            """
        )
        flags = _rows_to_dicts(cursor.fetchall())

        exported_at = datetime.now(UTC).isoformat()
        payload = {
            "meta": {
                "exported_at": exported_at,
                "source": str(DB_PATH),
                "counts": {"runs": len(runs), "flags": len(flags)},
            },
            "runs": runs,
            "flags": flags,
        }

        cursor.execute("DELETE FROM runs")
        cursor.execute("DELETE FROM flags")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('runs', 'flags')")
        conn.commit()
        conn.close()
        stopped, stop_msg = _stop_orchestrator_process()

        filename = f"autoneuro-db-export-{exported_at.replace(':', '-').replace('+00:00', 'Z')}.json"
        return jsonify(
            {
                "success": True,
                "message": f"Database exported/refreshed. {stop_msg}",
                "filename": filename,
                "export": payload,
                "orchestrator_stopped": stopped,
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    init_db()
    print(f"Database path: {DB_PATH}")
    print(f"Human instruction file: {HUMAN_INSTRUCTION_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=True)