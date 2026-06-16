import pytest
from langchain_core.messages import HumanMessage

from app.state import DecisionRecord, DiceRollRecord, SessionStore


def test_session_store_normalizes_ids_and_reuses_session():
    store = SessionStore()

    session = store.get_or_create(" mesa-1 ")
    session.messages.append(HumanMessage(content="olá"))

    same_session = store.get_or_create("mesa-1")

    assert same_session is session
    assert same_session.session_id == "mesa-1"
    assert len(same_session.messages) == 1


def test_session_store_rejects_blank_session_ids():
    store = SessionStore()

    with pytest.raises(ValueError, match="session_id"):
        store.get_or_create("   ")

    with pytest.raises(ValueError, match="session_id"):
        store.reset("")


def test_session_store_snapshot_serializes_mutable_state():
    store = SessionStore()
    session = store.get_or_create("mesa-snapshot")
    session.inventory["poção"] = 2
    session.decisions.append(
        DecisionRecord(
            decisao="Seguir pela floresta",
            contexto="Ao amanhecer",
            participantes=["Ana", "Bran"],
            timestamp="2026-06-16T00:00:00+00:00",
        )
    )
    session.dice_rolls.append(
        DiceRollRecord(
            expressao="1d20+3",
            quantidade=1,
            lados=20,
            rolagens=[17],
            modificador=3,
            total=20,
            motivo="barganha",
            timestamp="2026-06-16T00:01:00+00:00",
        )
    )

    snapshot = store.snapshot("mesa-snapshot")

    assert snapshot["inventory"] == {"poção": 2}
    assert snapshot["decisions"][0]["decisao"] == "Seguir pela floresta"
    assert snapshot["decisions"][0]["participantes"] == ["Ana", "Bran"]
    assert snapshot["dice_rolls"][0]["total"] == 20
