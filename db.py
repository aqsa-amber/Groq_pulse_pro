import sqlite3
import json
import time
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Conversation:
    id: int = None
    name: str = "Conversation"
    messages: List[Dict[str, Any]] = None

class ConversationStore:
    def __init__(self, db_path: str = "groq_pulse_db.sqlite"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """Create database tables if they don't exist and migrate schema if needed"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Conversations table
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                messages TEXT,
                created_at REAL
            )
        """)

        # Interactions table
        c.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                prompt TEXT,
                response TEXT,
                elapsed REAL,
                created_at REAL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
        """)

        # --- Migration: ensure conversation_id column exists ---
        c.execute("PRAGMA table_info(interactions)")
        columns = [row[1] for row in c.fetchall()]
        if "conversation_id" not in columns:
            print("Adding conversation_id column to interactions...")
            c.execute("ALTER TABLE interactions ADD COLUMN conversation_id INTEGER")
        else:
            print("conversation_id column already exists.")

        conn.commit()
        conn.close()

    # -----------------------------
    # Conversation CRUD
    # -----------------------------
    def create_conversation(self, name: str = "New Conversation"):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO conversations (name, messages, created_at) VALUES (?, ?, ?)",
            (name, json.dumps([]), time.time())
        )
        cid = c.lastrowid
        conn.commit()
        conn.close()
        return {"id": cid, "name": name, "messages": []}

    def list_conversations(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, name FROM conversations ORDER BY created_at DESC")
        rows = c.fetchall()
        conn.close()
        return [{"id": r[0], "name": r[1]} for r in rows]

    def get_conversation(self, cid: int):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, name, messages FROM conversations WHERE id=?", (cid,))
        row = c.fetchone()
        conn.close()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "messages": json.loads(row[2] or "[]")}

    def save_conversation(self, conv):
        """Update or create a conversation entry"""
        try:
            cid = getattr(conv, "id", None) or conv.get("id")
            name = getattr(conv, "name", None) or conv.get("name", "Conversation")
            messages = getattr(conv, "messages", None) or conv.get("messages", [])
        except Exception:
            return False

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if cid:
            c.execute(
                "UPDATE conversations SET name=?, messages=? WHERE id=?",
                (name, json.dumps(messages), cid)
            )
        else:
            c.execute(
                "INSERT INTO conversations (name, messages, created_at) VALUES (?, ?, ?)",
                (name, json.dumps(messages), time.time())
            )
        conn.commit()
        conn.close()
        return True

    def rename_conversation(self, cid: int, new_name: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE conversations SET name=? WHERE id=?", (new_name, cid))
        conn.commit()
        conn.close()

    def delete_conversation(self, cid: int):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM conversations WHERE id=?", (cid,))
        c.execute("DELETE FROM interactions WHERE conversation_id=?", (cid,))
        conn.commit()
        conn.close()

    # -----------------------------
    # Interactions
    # -----------------------------
    def log_interaction(self, prompt: str, response: str, elapsed: float = 0.0, conversation_id: Optional[int] = None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO interactions (conversation_id, prompt, response, elapsed, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, prompt, response, elapsed, time.time())
        )
        conn.commit()
        conn.close()

    def get_analytics(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM interactions")
        total = c.fetchone()[0] or 0
        c.execute("SELECT AVG(elapsed) FROM interactions")
        avg = c.fetchone()[0] or 0.0
        conn.close()
        return {"total_prompts": total, "avg_response_time": avg, "total_tokens": 0}

    # -----------------------------
    # Utility
    # -----------------------------
    def clear_all(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM conversations")
        c.execute("DELETE FROM interactions")
        conn.commit()
        conn.close()

    def export_all(self, filepath: str = "backup.json"):
        """Export all conversations + interactions to JSON file"""
        data = {"conversations": [], "interactions": []}
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT * FROM conversations")
        conv_rows = c.fetchall()
        conv_cols = [d[0] for d in c.description]
        data["conversations"] = [dict(zip(conv_cols, row)) for row in conv_rows]

        c.execute("SELECT * FROM interactions")
        int_rows = c.fetchall()
        int_cols = [d[0] for d in c.description]
        data["interactions"] = [dict(zip(int_cols, row)) for row in int_rows]

        conn.close()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath

    def import_all(self, filepath: str = "backup.json"):
        """Import conversations + interactions from JSON file"""
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for conv in data.get("conversations", []):
            c.execute(
                "INSERT OR REPLACE INTO conversations (id, name, messages, created_at) VALUES (?, ?, ?, ?)",
                (conv["id"], conv["name"], conv["messages"], conv["created_at"])
            )

        for inter in data.get("interactions", []):
            c.execute(
                "INSERT OR REPLACE INTO interactions (id, conversation_id, prompt, response, elapsed, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    inter["id"],
                    inter.get("conversation_id"),
                    inter["prompt"],
                    inter["response"],
                    inter["elapsed"],
                    inter["created_at"],
                )
            )

        conn.commit()
        conn.close()
        return True

