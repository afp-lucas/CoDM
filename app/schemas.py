"""Pydantic schemas for the Co-DM API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=4000)
    player_name: str | None = Field(default=None, max_length=100)

    @field_validator("session_id", "message")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("não pode ser vazio")
        return value

    @field_validator("player_name")
    @classmethod
    def player_name_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value


class ToolCallInfo(BaseModel):
    name: str
    args: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    tool_calls: list[ToolCallInfo]
    state: dict[str, Any]


class StateResponse(BaseModel):
    session_id: str
    inventory: dict[str, int]
    decisions: list[dict[str, Any]]
    dice_rolls: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str

