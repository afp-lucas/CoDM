"""FastAPI entrypoint for the Co-DM prototype."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException

from app.agent import run_agent_turn
from app.schemas import ChatRequest, ChatResponse, ErrorResponse, StateResponse
from app.state import SESSION_STORE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Co-DM",
    description="Assistente de coordenação para mesas de RPG de mesa.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "Co-DM"}


@app.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        return run_agent_turn(
            session_id=request.session_id,
            user_message=request.message,
            player_name=request.player_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Erro inesperado ao executar o agente.")
        raise HTTPException(status_code=500, detail="Erro ao processar a mensagem do agente.") from exc


@app.get("/sessions/{session_id}/state", response_model=StateResponse)
def get_session_state(session_id: str) -> dict[str, Any]:
    snapshot = SESSION_STORE.snapshot(session_id)
    return {"session_id": session_id, **snapshot}


@app.post("/sessions/{session_id}/reset", response_model=StateResponse)
def reset_session(session_id: str) -> dict[str, Any]:
    SESSION_STORE.reset(session_id)
    snapshot = SESSION_STORE.snapshot(session_id)
    return {"session_id": session_id, **snapshot}

