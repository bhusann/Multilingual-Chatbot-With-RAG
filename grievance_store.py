"""
grievance_store.py
==================
SQLite ticket store for the grievance_tool.

Table `tickets`:
    ticket_id    TEXT PRIMARY KEY  (auto-generated, e.g. GRV-20260907-A3F9)
    user_name    TEXT NOT NULL
    mobile       TEXT NOT NULL
    state        TEXT NOT NULL
    district     TEXT NOT NULL
    category     TEXT NOT NULL      (PACS / PMFBY / LOAN / SUBSIDY / OTHER)
    description  TEXT NOT NULL
    status       TEXT NOT NULL      (SUBMITTED -> ... -> RESOLVED)
    created_at   TEXT NOT NULL      (UTC ISO timestamp)

Status is written SUBMITTED at creation and changed afterwards only
from the admin panel (or update_status). The DB file lives at
data/grievances.db and survives restarts.
"""

import os
import random
import sqlite3
import string
from datetime import datetime, timezone


# Allowed statuses in forward order
STATUS_FLOW = [
    "SUBMITTED",
    "ACKNOWLEDGED",
    "UNDER_REVIEW",
    "ACTION_REQUIRED",
    "RESOLVED",
]

# Suggested categories (LLM picks from context; stored as text)
CATEGORIES = ["PACS", "PMFBY", "LOAN", "SUBSIDY", "OTHER"]

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "grievances.db",
)


def generate_ticket_id():
    """Generate a unique-ish ticket id: GRV-YYYYMMDD-XXXX."""
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    rand_part = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=4)
    )
    return f"GRV-{date_part}-{rand_part}"


class GrievanceStore:
    """SQLite-backed grievance ticket store."""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        print(f"Grievance store ready at: {self.db_path}")

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id   TEXT PRIMARY KEY,
                    user_name   TEXT NOT NULL,
                    mobile      TEXT NOT NULL,
                    state       TEXT NOT NULL,
                    district    TEXT NOT NULL,
                    category    TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'SUBMITTED',
                    created_at  TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tickets_mobile "
                "ON tickets(mobile)"
            )

    def create_ticket(
        self, user_name, mobile, state,
        district, category, description,
    ):
        """Create a ticket. Returns the ticket dict."""
        category = (category or "OTHER").strip().upper()
        if category not in CATEGORIES:
            category = "OTHER"

        ticket_id = generate_ticket_id()
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            # Retry id on (very unlikely) collision
            for _ in range(3):
                try:
                    conn.execute(
                        "INSERT INTO tickets "
                        "(ticket_id, user_name, mobile, state, "
                        " district, category, description, status,"
                        " created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            ticket_id, user_name.strip(),
                            mobile.strip(), state.strip(),
                            district.strip(), category,
                            description.strip(), "SUBMITTED",
                            created_at,
                        ),
                    )
                    break
                except sqlite3.IntegrityError:
                    ticket_id = generate_ticket_id()
            else:
                raise RuntimeError(
                    "Could not generate unique ticket id"
                )

        print(f"🎫 Ticket created: {ticket_id} ({category})")
        return self.get_ticket(ticket_id)

    def get_ticket(self, ticket_id):
        """Fetch one ticket by id. Returns dict or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?",
                (ticket_id.strip(),),
            ).fetchone()
        return dict(row) if row else None

    def find_by_mobile(self, mobile):
        """All tickets for a mobile number, newest first."""
        digits = "".join(c for c in mobile if c.isdigit())
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE mobile LIKE ? "
                "ORDER BY created_at DESC",
                (f"%{digits}%",),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_status(self, ticket_id, status):
        """Set ticket status. Returns updated dict or None."""
        status = (status or "").strip().upper()
        if status not in STATUS_FLOW:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Allowed: {', '.join(STATUS_FLOW)}"
            )

        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE tickets SET status = ? WHERE ticket_id = ?",
                (status, ticket_id.strip()),
            )
            if cur.rowcount == 0:
                return None

        print(f"🎫 Ticket {ticket_id}: status -> {status}")
        return self.get_ticket(ticket_id)

    def list_tickets(self, limit=100):
        """All tickets, newest first (for admin panel)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self):
        """Total number of tickets."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM tickets"
            ).fetchone()
        return row["n"]


# Shared instance used by the LLM tool + admin endpoints
_grievance_store = None


def get_grievance_store():
    """Return the singleton grievance store."""
    global _grievance_store
    if _grievance_store is None:
        _grievance_store = GrievanceStore()
    return _grievance_store
