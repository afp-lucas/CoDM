"""LangGraph ReAct agent for the Co-DM assistant."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr, ValidationError

from app.config import get_settings
from app.state import SESSION_STORE
from app.tools import build_tools_for_session


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Você é o Co-DM, um assistente de coordenação para uma mesa de RPG de mesa em português brasileiro.

Seu papel é apoiar jogadores e Mestre em tarefas mecânicas e de organização, sem substituir a criatividade humana.

Regras de comportamento:
1. Responda sempre em português brasileiro.
2. Seja claro, objetivo e útil.
3. Não decida pelo grupo. As decisões pertencem aos jogadores.
4. Não compre itens, remova itens ou registre decisões se a fala do usuário for ambígua. Nesses casos, peça confirmação.
5. Use ferramentas quando a mensagem pedir uma ação objetiva:
   - rolar dados;
   - adicionar, remover ou consultar itens do inventário do grupo;
   - registrar ou consultar decisões do grupo;
   - buscar regras, monstros, orientações ou conteúdo dos manuais em PDF.
6. Quando a pergunta depender dos manuais, use buscar_manuais_rpg antes de responder.
7. Ao responder com base nos manuais, cite o nome do manual e a página retornada pela ferramenta.
8. Se a mensagem for apenas roleplay, opinião, conversa ou pedido narrativo simples, responda diretamente sem usar ferramenta.
9. Após usar uma ferramenta, explique o resultado de forma natural e curta.
10. Não invente estado do inventário. Consulte a ferramenta se precisar saber o estado atual.
11. Não invente decisões passadas. Consulte a ferramenta se precisar saber o histórico.
12. Seja transparente quando algo não puder ser feito.
13. O Co-DM não é o Mestre principal. Ele é um assistente de apoio mecânico e colaborativo.

Exemplos de uso de ferramentas:
- "rola 1d20+3 para barganhar" -> use rolar_dados.
- "compramos 3 poções de cura" -> use atualizar_inventario_grupo com acao adicionar.
- "gastamos uma poção de cura" -> use atualizar_inventario_grupo com acao remover.
- "registrar que vamos pela estrada da floresta" -> use registrar_decisao_grupo.
- "o que temos no inventário?" -> use consultar_inventario_grupo.
- "o que decidimos até agora?" -> use consultar_decisoes_grupo.
- "como funciona vantagem?", "qual a CA do goblin?", "como criar personagem?" -> use buscar_manuais_rpg.

Exemplos em que NÃO deve usar ferramenta:
- "Meu guerreiro encara o taverneiro."
- "Acho melhor irmos pela floresta, o que vocês acham?"
- "Descreva a taverna de forma breve."
- "Qual estratégia parece mais segura?"

Em caso de dúvida, pergunte ao grupo antes de alterar qualquer estado compartilhado."""


def build_react_agent(session_id: str):
    """Build the LangGraph ReAct loop for one session."""

    settings = get_settings()
    if not settings.openrouter_api_key.strip():
        raise RuntimeError("OPENROUTER_API_KEY precisa estar configurada para usar o agente.")

    tools = build_tools_for_session(session_id)
    llm = ChatOpenRouter(
        model=settings.openrouter_model,
        api_key=SecretStr(settings.openrouter_api_key),
        temperature=0.2,
        max_tokens=1000,
    ).bind_tools(tools)

    def agent(state: MessagesState) -> MessagesState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None) or []
    extracted: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            extracted.append({"name": call.get("name", ""), "args": call.get("args") or {}})
        else:
            extracted.append(
                {
                    "name": getattr(call, "name", ""),
                    "args": getattr(call, "args", None) or {},
                }
            )
    return [call for call in extracted if call["name"]]


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def run_agent_turn(session_id: str, user_message: str, player_name: str | None = None) -> dict[str, Any]:
    """Run one user turn, preserving session message history."""

    try:
        session = SESSION_STORE.get_or_create(session_id)
        react_agent = build_react_agent(session_id)
    except ValidationError as exc:
        logger.warning("Configuração inválida do OpenRouter: %s", exc)
        raise RuntimeError("OPENROUTER_API_KEY precisa estar configurada para usar o /chat.") from exc

    content = f"[Jogador: {player_name.strip()}] {user_message}" if player_name and player_name.strip() else user_message
    session.messages.append(HumanMessage(content=content))

    seen_tool_call_keys: set[tuple[str, str]] = set()
    tool_calls: list[dict[str, Any]] = []
    final_messages = session.messages

    for step in react_agent.stream({"messages": session.messages}, stream_mode="values"):
        messages = step.get("messages", [])
        if messages:
            final_messages = messages
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                for call in _extract_tool_calls(last_message):
                    key = (call["name"], repr(call.get("args") or {}))
                    if key not in seen_tool_call_keys:
                        seen_tool_call_keys.add(key)
                        tool_calls.append(call)

    session.messages = list(final_messages)

    final_ai_messages = [message for message in reversed(session.messages) if isinstance(message, AIMessage)]
    response = _message_text(final_ai_messages[0]).strip() if final_ai_messages else ""
    if not response:
        response = "Não consegui gerar uma resposta final para esta interação."

    return {
        "session_id": session.session_id,
        "response": response,
        "tool_calls": tool_calls,
        "state": SESSION_STORE.snapshot(session.session_id),
    }
