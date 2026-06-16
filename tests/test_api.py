from app.main import get_session_state, health, reset_session


def test_health_endpoint_returns_status():
    assert health() == {"status": "ok", "app": "Co-DM"}


def test_session_state_and_reset_endpoints_return_empty_state():
    session_id = "mesa-api-state"

    state_response = get_session_state(session_id)
    reset_response = reset_session(session_id)

    assert state_response == {
        "session_id": session_id,
        "inventory": {},
        "decisions": [],
        "dice_rolls": [],
    }
    assert reset_response == {
        "session_id": session_id,
        "inventory": {},
        "decisions": [],
        "dice_rolls": [],
    }
