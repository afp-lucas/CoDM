"""In-memory session state for the Co-DM prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any

from langchain_core.messages import BaseMessage


@dataclass(slots=True)
class DecisionRecord:
    """A collaborative decision registered for a table session."""

    decisao: str
    timestamp: str
    contexto: str | None = None
    participantes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiceRollRecord:
    """A dice roll registered for a table session."""

    expressao: str
    quantidade: int
    lados: int
    rolagens: list[int]
    modificador: int
    total: int
    timestamp: str
    motivo: str | None = None


@dataclass(slots=True)
class SessionData:
    """Mutable state owned by one RPG table session."""

    session_id: str
    messages: list[BaseMessage] = field(default_factory=list)
    inventory: dict[str, int] = field(default_factory=dict)
    decisions: list[DecisionRecord] = field(default_factory=list)
    dice_rolls: list[DiceRollRecord] = field(default_factory=list)


class SessionStore:
    """Small in-memory repository for session data.

    The class intentionally isolates persistence details so the prototype can
    later swap this dictionary-backed implementation for SQLite or another
    store without spreading global state through the application.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> SessionData:
        """Return an existing session or create an empty one."""

        normalized_id = session_id.strip()
        if not normalized_id:
            raise ValueError("session_id não pode ser vazio.")

        with self._lock:
            if normalized_id not in self._sessions:
                self._sessions[normalized_id] = SessionData(session_id=normalized_id)
            return self._sessions[normalized_id]

    def reset(self, session_id: str) -> None:
        """Clear all state for a session."""

        normalized_id = session_id.strip()
        if not normalized_id:
            raise ValueError("session_id não pode ser vazio.")

        with self._lock:
            self._sessions[normalized_id] = SessionData(session_id=normalized_id)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """Return API-safe state for a session."""

        session = self.get_or_create(session_id)
        with self._lock:
            return {
                "inventory": dict(session.inventory),
                "decisions": [asdict(decision) for decision in session.decisions],
                "dice_rolls": [asdict(roll) for roll in session.dice_rolls],
            }


SESSION_STORE = SessionStore()

