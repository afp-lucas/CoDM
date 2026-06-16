from langchain_core.messages import AIMessage, HumanMessage

from app import agent as agent_module
from app.state import SESSION_STORE


class FakeReactAgent:
    def stream(self, state, stream_mode):
        messages = list(state["messages"])
        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "rolar_dados", "args": {"expressao": "1d20"}, "id": "call-1"}],
        )
        final = AIMessage(content=[{"text": "Resultado final da rodada."}])

        yield {"messages": [*messages, tool_call]}
        yield {"messages": [*messages, tool_call, final]}


def test_run_agent_turn_records_player_message_and_tool_calls(monkeypatch):
    session_id = "mesa-agent-turn"
    SESSION_STORE.reset(session_id)
    monkeypatch.setattr(agent_module, "build_react_agent", lambda _session_id: FakeReactAgent())

    result = agent_module.run_agent_turn(session_id, "Rola 1d20", " Ana ")
    session = SESSION_STORE.get_or_create(session_id)

    assert result["session_id"] == session_id
    assert result["response"] == "Resultado final da rodada."
    assert result["tool_calls"] == [{"name": "rolar_dados", "args": {"expressao": "1d20"}}]
    assert isinstance(session.messages[0], HumanMessage)
    assert session.messages[0].content == "[Jogador: Ana] Rola 1d20"


def test_run_agent_turn_uses_fallback_when_final_response_is_empty(monkeypatch):
    class EmptyFinalAgent:
        def stream(self, state, stream_mode):
            yield {"messages": [*state["messages"], AIMessage(content="")]}

    session_id = "mesa-agent-empty"
    SESSION_STORE.reset(session_id)
    monkeypatch.setattr(agent_module, "build_react_agent", lambda _session_id: EmptyFinalAgent())

    result = agent_module.run_agent_turn(session_id, "Oi")

    assert result["response"] == "Não consegui gerar uma resposta final para esta interação."
    assert result["tool_calls"] == []


def test_message_text_joins_structured_content():
    message = AIMessage(content=[{"text": "linha 1"}, {"text": "linha 2"}])

    assert agent_module._message_text(message) == "linha 1\nlinha 2"
