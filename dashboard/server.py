#!/usr/bin/env python3
"""
Flask backend server for AutoNeuro dashboard.
Serves experiment runs and flags management pages with API endpoints.
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add the parent directory to Python path to import contact_agent
sys.path.append(str(Path(__file__).parent.parent))
from agents.contact_agent import escalate, generate_summary

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Database path - one level up from dashboard directory
DB_PATH = Path(__file__).parent.parent / "database.db"
HUMAN_INSTRUCTION_PATH = Path(__file__).parent.parent / "HUMAN_INSTRUCTION.txt"


def get_db_connection():
    """Get SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables dict-like access to rows
    return conn


# Static file serving for HTML pages
@app.route('/')
def index():
    """Serve the experiment runs page."""
    return send_from_directory('.', 'index.html')


@app.route('/flags')
def flags():
    """Serve the flags management page."""
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


if __name__ == '__main__':
    print(f"Database path: {DB_PATH}")
    print(f"Human instruction file: {HUMAN_INSTRUCTION_PATH}")
    
    # Check if database exists
    if not DB_PATH.exists():
        print(f"WARNING: Database file not found at {DB_PATH}")
        print("Make sure you've created the database with the proper schema.")
    
    # Start the Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)