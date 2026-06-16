"""Streamlit interface for the Co-DM collaborative assistant."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import run_agent_turn
from app.config import get_settings
from app.rag import discover_pdf_paths
from app.state import SESSION_STORE


st.set_page_config(page_title="Co-DM", page_icon="CD", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --codm-right-sidebar-width: 21.5rem;
        --codm-right-sidebar-gap: 1.5rem;
        --codm-right-sidebar-total: calc(var(--codm-right-sidebar-width) + var(--codm-right-sidebar-gap));
        --codm-left-sidebar-width: min(24rem, 28vw);
        --codm-app-bg: var(--background-color, rgb(14, 17, 23));
    }

    section.main > div,
    section[data-testid="stMain"] > div,
    div[data-testid="stMainBlockContainer"],
    .main .block-container,
    .stMain .block-container {
        box-sizing: border-box;
        max-width: none;
        padding-right: var(--codm-right-sidebar-total) !important;
    }

    div[data-testid="stChatMessage"] {
        box-sizing: border-box;
        max-width: calc(100vw - var(--codm-left-sidebar-width) - var(--codm-right-sidebar-total) - 4rem);
    }

    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: var(--codm-left-sidebar-width);
        right: var(--codm-right-sidebar-total);
        z-index: 200;
        padding: 0.75rem 1rem 1rem;
        background: var(--codm-app-bg);
    }

    .st-key-codm_right_sidebar {
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        width: var(--codm-right-sidebar-width);
        min-width: var(--codm-right-sidebar-width);
        max-width: var(--codm-right-sidebar-width);
        z-index: 300;
        padding: 2.75rem 1rem 1rem;
        overflow-y: auto;
        background: var(--codm-app-bg) !important;
        border-left: 1px solid rgba(128, 128, 128, 0.22);
        box-shadow: -1rem 0 1.5rem rgba(14, 17, 23, 0.88);
        isolation: isolate;
    }

    .st-key-codm_right_sidebar::before {
        content: "";
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        width: var(--codm-right-sidebar-width);
        z-index: -1;
        background: var(--codm-app-bg);
    }

    .st-key-codm_right_sidebar > div {
        width: 100%;
        position: relative;
        z-index: 1;
    }

    div[data-testid="stDialog"] div[role="dialog"] {
        max-height: 85vh;
    }

    @media (max-width: 900px) {
        section.main > div,
        section[data-testid="stMain"] > div,
        div[data-testid="stMainBlockContainer"],
        .main .block-container,
        .stMain .block-container {
            padding-right: 1rem;
        }

        div[data-testid="stChatMessage"] {
            max-width: none;
        }

        div[data-testid="stChatInput"] {
            left: 1rem;
            right: 1rem;
        }

        .st-key-codm_right_sidebar {
            position: static;
            width: 100% !important;
            min-width: 0 !important;
            max-width: none !important;
            padding: 1rem 0;
            border-left: 0;
            box-shadow: none;
            background: transparent !important;
        }

        .st-key-codm_right_sidebar::before {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _human_label_and_text(content: str) -> tuple[str, str]:
    if content.startswith("[Jogador: ") and "] " in content:
        label, text = content.split("] ", 1)
        return label.replace("[Jogador: ", "").strip(), text
    return "Usuário", content


def _render_history(session_id: str) -> None:
    session = SESSION_STORE.get_or_create(session_id)
    for message in session.messages:
        if isinstance(message, HumanMessage):
            label, text = _human_label_and_text(str(message.content))
            with st.chat_message("user"):
                st.caption(label)
                st.markdown(text)
        elif isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None) and not str(message.content or "").strip():
                continue
            with st.chat_message("assistant"):
                st.markdown(str(message.content))
        elif isinstance(message, ToolMessage):
            continue


def _render_inventory_popup(session_id: str) -> None:
    state = SESSION_STORE.snapshot(session_id)

    with st.container(height=420, border=False):
        if state["inventory"]:
            st.dataframe(
                [{"Item": item, "Quantidade": quantity} for item, quantity in state["inventory"].items()],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("Nenhum item registrado.")


def _render_decisions_popup(session_id: str) -> None:
    state = SESSION_STORE.snapshot(session_id)

    with st.container(height=420, border=False):
        if state["decisions"]:
            for decision in reversed(state["decisions"]):
                st.markdown(f"**{decision['decisao']}**")
                if decision.get("contexto"):
                    st.caption(decision["contexto"])
                if decision.get("participantes"):
                    st.caption(f"Participantes: {', '.join(decision['participantes'])}")
                st.caption(decision["timestamp"])
                st.divider()
        else:
            st.caption("Nenhuma decisão registrada.")


def _render_rolls_popup(session_id: str) -> None:
    state = SESSION_STORE.snapshot(session_id)

    with st.container(height=420, border=False):
        if state["dice_rolls"]:
            for roll in reversed(state["dice_rolls"]):
                st.markdown(f"`{roll['expressao']}` -> **{roll['total']}**")
                st.caption(f"Resultados: {roll['rolagens']}")
                if roll.get("motivo"):
                    st.caption(roll["motivo"])
                st.caption(roll["timestamp"])
                st.divider()
        else:
            st.caption("Nenhuma rolagem registrada.")


@st.dialog("Inventário da equipe")
def _inventory_dialog(session_id: str) -> None:
    _render_inventory_popup(session_id)


@st.dialog("Decisões")
def _decisions_dialog(session_id: str) -> None:
    _render_decisions_popup(session_id)


@st.dialog("Rolagens")
def _rolls_dialog(session_id: str) -> None:
    _render_rolls_popup(session_id)


def _init_players() -> None:
    if "codm_players" not in st.session_state:
        st.session_state.codm_players = []
    if "codm_active_player" not in st.session_state:
        st.session_state.codm_active_player = "Mestre"


def _add_player_from_sidebar() -> None:
    new_player = st.session_state.get("codm_new_player", "").strip()
    if not new_player:
        return
    if new_player != "Mestre" and new_player not in st.session_state.codm_players:
        st.session_state.codm_players.append(new_player)
    st.session_state.codm_active_player = new_player
    st.session_state.codm_new_player = ""


settings = get_settings()
_init_players()

st.title("Co-DM")
st.caption("Assistente colaborativo para Mestre e jogadores consultarem manuais, coordenarem decisões e registrarem estado da mesa.")

with st.sidebar:
    st.header("Sessão")
    session_id = st.text_input("ID da sessão", value="mesa-1", max_chars=200)

    st.header("Usuários")
    user_options = ["Mestre", *st.session_state.codm_players]
    player_name = st.segmented_control(
        "Usuário ativo",
        options=user_options,
        default=st.session_state.codm_active_player
        if st.session_state.codm_active_player in user_options
        else "Mestre",
        key="codm_active_player",
    )
    st.text_input("Nome do jogador", key="codm_new_player", max_chars=100)
    st.button("Adicionar jogador", on_click=_add_player_from_sidebar, use_container_width=True)

    if st.button("Resetar sessão", use_container_width=True):
        SESSION_STORE.reset(session_id)
        st.rerun()

    st.header("Manuais")
    pdfs = discover_pdf_paths(settings.manuals_dir)
    if pdfs:
        for pdf in pdfs:
            st.caption(pdf.name)
    else:
        st.warning(f"Nenhum PDF encontrado em `{settings.manuals_dir}`.")

session_id = session_id.strip() or "mesa-1"
player_name = player_name.strip() or None

with st.container(key="codm_right_sidebar"):
    st.subheader("Mesa")
    st.caption(f"Sessão: `{session_id}`")
    st.caption(f"Usuário: `{player_name or 'Usuário'}`")

    if st.button("Inventário da equipe", use_container_width=True):
        _inventory_dialog(session_id)
    if st.button("Decisões", use_container_width=True):
        _decisions_dialog(session_id)
    if st.button("Rolagens", use_container_width=True):
        _rolls_dialog(session_id)

_render_history(session_id)
st.markdown('<div style="height: 5.5rem;"></div>', unsafe_allow_html=True)

prompt = st.chat_input("Envie uma mensagem para o grupo e para o Co-DM")
if prompt:
    with st.chat_message("user"):
        st.caption(player_name or "Usuário")
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Co-DM pensando..."):
            try:
                result = run_agent_turn(session_id, prompt, player_name)
                st.markdown(result["response"])
                if result["tool_calls"]:
                    with st.expander("Ferramentas usadas nesta rodada"):
                        for call in result["tool_calls"]:
                            st.code(f"{call['name']}({call.get('args') or {}})", language="python")
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Erro ao processar a mensagem: {exc}")
