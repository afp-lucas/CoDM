from app import tools as tools_module
from app.state import SESSION_STORE
from app.tools import MAX_DICE_QUANTITY, build_tools_for_session


def _tools_for(session_id: str):
    SESSION_STORE.reset(session_id)
    return {tool.name: tool for tool in build_tools_for_session(session_id)}


def test_rolar_dados_normalizes_records_and_returns_total(monkeypatch):
    session_id = "mesa-tools-dice"
    available_tools = _tools_for(session_id)
    rolls = iter([4, 6])
    monkeypatch.setattr(tools_module.random, "randint", lambda _start, _end: next(rolls))

    result = available_tools["rolar_dados"].invoke({"expressao": " 2D6+3 ", "motivo": " ataque "})
    snapshot = SESSION_STORE.snapshot(session_id)

    assert result["ok"] is True
    assert result["expressao"] == "2d6+3"
    assert result["rolagens"] == [4, 6]
    assert result["total"] == 13
    assert result["motivo"] == "ataque"
    assert snapshot["dice_rolls"][0]["expressao"] == "2d6+3"
    assert snapshot["dice_rolls"][0]["total"] == 13


def test_rolar_dados_rejects_invalid_expression_without_recording():
    session_id = "mesa-tools-dice-invalid"
    available_tools = _tools_for(session_id)

    result = available_tools["rolar_dados"].invoke({"expressao": "d20"})

    assert result["ok"] is False
    assert "Expressão inválida" in result["erro"]
    assert SESSION_STORE.snapshot(session_id)["dice_rolls"] == []


def test_rolar_dados_rejects_too_many_dice():
    available_tools = _tools_for("mesa-tools-dice-limit")

    result = available_tools["rolar_dados"].invoke({"expressao": f"{MAX_DICE_QUANTITY + 1}d6"})

    assert result["ok"] is False
    assert str(MAX_DICE_QUANTITY) in result["erro"]


def test_atualizar_inventario_adds_removes_and_normalizes_items():
    session_id = "mesa-tools-inventory"
    available_tools = _tools_for(session_id)
    inventory_tool = available_tools["atualizar_inventario_grupo"]

    added = inventory_tool.invoke({"acao": " adicionar ", "item": " Poção de Cura ", "quantidade": 3})
    removed = inventory_tool.invoke({"acao": "remover", "item": "poção de cura", "quantidade": 1})

    assert added["ok"] is True
    assert added["inventario"] == {"poção de cura": 3}
    assert removed["ok"] is True
    assert removed["inventario"] == {"poção de cura": 2}
    assert SESSION_STORE.snapshot(session_id)["inventory"] == {"poção de cura": 2}


def test_atualizar_inventario_rejects_removing_more_than_available():
    session_id = "mesa-tools-inventory-invalid"
    available_tools = _tools_for(session_id)
    inventory_tool = available_tools["atualizar_inventario_grupo"]
    inventory_tool.invoke({"acao": "adicionar", "item": "flecha", "quantidade": 2})

    result = inventory_tool.invoke({"acao": "remover", "item": "flecha", "quantidade": 3})

    assert result["ok"] is False
    assert result["disponivel"] == 2
    assert SESSION_STORE.snapshot(session_id)["inventory"] == {"flecha": 2}


def test_registrar_e_consultar_decisoes_normalizes_fields():
    session_id = "mesa-tools-decisions"
    available_tools = _tools_for(session_id)

    registered = available_tools["registrar_decisao_grupo"].invoke(
        {
            "decisao": "  Seguir pela estrada  ",
            "contexto": "  depois do descanso  ",
            "participantes": [" Ana ", "", "Bran"],
        }
    )
    consulted = available_tools["consultar_decisoes_grupo"].invoke({"limite": 5})

    assert registered["ok"] is True
    assert registered["decisao"] == "Seguir pela estrada"
    assert registered["contexto"] == "depois do descanso"
    assert registered["participantes"] == ["Ana", "Bran"]
    assert consulted["decisoes"][0]["participantes"] == ["Ana", "Bran"]
    assert consulted["total_decisoes"] == 1


def test_consultar_inventario_returns_current_session_inventory():
    session_id = "mesa-tools-consult-inventory"
    available_tools = _tools_for(session_id)
    available_tools["atualizar_inventario_grupo"].invoke(
        {"acao": "adicionar", "item": "corda", "quantidade": 1}
    )

    result = available_tools["consultar_inventario_grupo"].invoke({})

    assert result == {"ok": True, "inventario": {"corda": 1}}
