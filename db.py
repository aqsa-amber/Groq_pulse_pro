import os
import sqlite3
import json
from typing import List, Dict, Optional

class ConversationStore:
    """
    Manages conversations and interactions in a SQLite database.
    Fully compatible with Streamlit app session state.
    """

    def __init__(self, db_path: str = "groq_pulse_db.sqlite"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        self._migrate_db()

    def _init_db(self):
        """Initialize tables if they do not exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                messages TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                user_input TEXT,
                assistant_output TEXT,
                elapsed REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _migrate_db(self):
        """Add missing columns if upgrading from old DB."""
        cursor = self.conn.cursor()
        # Check if 'user_input' exists
        cursor.execute("PRAGMA table_info(interactions)")
        columns = [col[1] for col in cursor.fetchall()]
        if "user_input" not in columns:
            cursor.execute("ALTER TABLE interactions ADD COLUMN user_input TEXT")
        if "assistant_output" not in columns:
            cursor.execute("ALTER TABLE interactions ADD COLUMN assistant_output TEXT")
        self.conn.commit()

    # -----------------------------
    # Conversation CRUD
    # -----------------------------
    def create_conversation(self, name: str = "New Conversation") -> Dict:
        """Create a new conversation and return its data."""
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO conversations (name, messages) VALUES (?, ?)", (name, json.dumps([])))
        self.conn.commit()
        conv_id = cursor.lastrowid
        return {"id": conv_id, "name": name, "messages": []}

    def get_conversation(self, conv_id: int) -> Optional[Dict]:
        """Retrieve conversation by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, messages FROM conversations WHERE id = ?", (conv_id,))
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "name": row[1], "messages": json.loads(row[2])}
        return None

    def list_conversations(self) -> List[Dict]:
        """List all conversations."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, messages FROM conversations ORDER BY id")
        rows = cursor.fetchall()
        return [{"id": r[0], "name": r[1], "messages": json.loads(r[2])} for r in rows]

    def save_conversation(self, conversation: Dict):
        """Save conversation messages to DB."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE conversations SET messages = ? WHERE id = ?",
            (json.dumps(conversation.get("messages", [])), conversation["id"])
        )
        self.conn.commit()

    def rename_conversation(self, conv_id: int, new_name: str):
        """Rename a conversation."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE conversations SET name = ? WHERE id = ?", (new_name, conv_id))
        self.conn.commit()

    def delete_conversation(self, conv_id: int):
        """Delete a conversation and all its interactions."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        cursor.execute("DELETE FROM interactions WHERE conversation_id = ?", (conv_id,))
        self.conn.commit()

    def clear_all(self):
        """Clear all conversations and interactions."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM conversations")
        cursor.execute("DELETE FROM interactions")
        self.conn.commit()

    # -----------------------------
    # Import / Export
    # -----------------------------
    def export_all(self, path: str = "backup.json") -> str:
        """Export all conversations to JSON file."""
        data = self.list_conversations()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return os.path.abspath(path)

    def import_all(self, path: str) -> bool:
        """Import conversations from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for conv in data:
                self.create_conversation(name=conv.get("name", "Imported Conversation"))
            return True
        except Exception as e:
            print("Import failed:", e)
            return False

    # -----------------------------
    # Interaction logging
    # -----------------------------
    def log_interaction(self, user_input: str, assistant_output: str, elapsed: float, conversation_id: int):
        """Log an interaction in DB."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO interactions (conversation_id, user_input, assistant_output, elapsed) VALUES (?, ?, ?, ?)",
            (conversation_id, user_input, assistant_output, elapsed)
        )
        self.conn.commit()

