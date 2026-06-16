"""LangChain tools used by the Co-DM ReAct agent."""

from __future__ import annotations

import random
import re
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

from app.state import DecisionRecord, DiceRollRecord, SESSION_STORE


DICE_EXPRESSION_RE = re.compile(r"^\s*(?P<quantity>\d+)d(?P<sides>\d+)(?P<modifier>[+-]\d+)?\s*$", re.IGNORECASE)
MAX_DICE_QUANTITY = 20
MAX_DICE_SIDES = 1000


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "erro": message, **extra}


def build_tools_for_session(session_id: str) -> list[Any]:
    """Build tool closures bound to a single collaborative session."""

    session = SESSION_STORE.get_or_create(session_id)

    @tool
    def rolar_dados(expressao: str, motivo: str | None = None) -> dict[str, Any]:
        """Rola dados para uma ação mecânica solicitada pelo grupo.

        Use esta ferramenta quando os jogadores pedirem uma rolagem objetiva,
        como testes de atributo, ataque, dano, barganha, percepção ou qualquer
        outra expressão de dados no formato NdM, opcionalmente com modificador
        +K ou -K. Exemplos válidos: "1d20", "2d6", "1d20+3",
        "2d8-1", "4d6+2". Não use para narrar cenas ou decidir ações pelo
        grupo. A ferramenta valida limites de abuso e registra a rolagem na
        sessão compartilhada.
        """

        match = DICE_EXPRESSION_RE.match(expressao or "")
        if not match:
            return _error(
                "Expressão inválida. Use o formato NdM, opcionalmente com +K ou -K.",
                expressao=expressao,
            )

        quantidade = int(match.group("quantity"))
        lados = int(match.group("sides"))
        modificador = int(match.group("modifier") or 0)
        expressao_normalizada = f"{quantidade}d{lados}{modificador:+d}" if modificador else f"{quantidade}d{lados}"

        if quantidade < 1:
            return _error("A quantidade de dados deve ser pelo menos 1.", expressao=expressao)
        if quantidade > MAX_DICE_QUANTITY:
            return _error(
                f"A quantidade máxima permitida é {MAX_DICE_QUANTITY} dados.",
                expressao=expressao,
                quantidade=quantidade,
            )
        if lados < 1:
            return _error("O dado deve ter pelo menos 1 lado.", expressao=expressao)
        if lados > MAX_DICE_SIDES:
            return _error(
                f"O número máximo de lados permitido é {MAX_DICE_SIDES}.",
                expressao=expressao,
                lados=lados,
            )

        rolagens = [random.randint(1, lados) for _ in range(quantidade)]
        total = sum(rolagens) + modificador
        motivo_normalizado = motivo.strip() if isinstance(motivo, str) and motivo.strip() else None

        record = DiceRollRecord(
            expressao=expressao_normalizada,
            quantidade=quantidade,
            lados=lados,
            rolagens=rolagens,
            modificador=modificador,
            total=total,
            motivo=motivo_normalizado,
            timestamp=_utc_now_iso(),
        )
        session.dice_rolls.append(record)

        return {
            "ok": True,
            "expressao": expressao_normalizada,
            "quantidade": quantidade,
            "lados": lados,
            "rolagens": rolagens,
            "modificador": modificador,
            "total": total,
            "motivo": motivo_normalizado,
        }

    @tool
    def atualizar_inventario_grupo(acao: str, item: str, quantidade: int) -> dict[str, Any]:
        """Atualiza o inventário compartilhado da party.

        Use esta ferramenta quando o grupo afirmar claramente que adicionou,
        removeu/gastou ou definiu uma quantidade de um item compartilhado. A
        ação deve ser "adicionar", "remover" ou "definir". O item pertence ao
        grupo, não a um jogador isolado. Não use quando a fala for ambígua,
        quando os jogadores estiverem apenas discutindo uma compra possível, ou
        quando ainda houver dúvida sobre a decisão coletiva; nesses casos peça
        confirmação antes de alterar o estado.
        """

        acao_normalizada = (acao or "").strip().lower()
        item_normalizado = (item or "").strip().lower()

        if acao_normalizada not in {"adicionar", "remover", "definir"}:
            return _error("Ação inválida. Use adicionar, remover ou definir.", acao=acao)
        if not item_normalizado:
            return _error("Item não pode ser vazio.", item=item)
        if not isinstance(quantidade, int) or quantidade <= 0:
            return _error("Quantidade deve ser um inteiro positivo.", quantidade=quantidade)

        inventario_atual = dict(session.inventory)
        quantidade_atual = inventario_atual.get(item_normalizado, 0)

        if acao_normalizada == "adicionar":
            quantidade_final = quantidade_atual + quantidade
        elif acao_normalizada == "remover":
            if item_normalizado not in inventario_atual:
                return _error("Não é possível remover um item inexistente.", item=item_normalizado)
            if quantidade > quantidade_atual:
                return _error(
                    "Não é possível remover mais itens do que o grupo possui.",
                    item=item_normalizado,
                    disponivel=quantidade_atual,
                    solicitado=quantidade,
                )
            quantidade_final = quantidade_atual - quantidade
        else:
            quantidade_final = quantidade

        if quantidade_final == 0:
            inventario_atual.pop(item_normalizado, None)
        else:
            inventario_atual[item_normalizado] = quantidade_final

        session.inventory.clear()
        session.inventory.update(inventario_atual)

        return {
            "ok": True,
            "acao": acao_normalizada,
            "item": item_normalizado,
            "quantidade": quantidade,
            "inventario": dict(session.inventory),
        }

    @tool
    def registrar_decisao_grupo(
        decisao: str,
        contexto: str | None = None,
        participantes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Registra uma decisão coletiva já confirmada pelos jogadores.

        Use esta ferramenta quando o usuário pedir explicitamente para registrar
        uma decisão do grupo ou declarar de forma clara que a party decidiu um
        plano, divisão de recursos, rota, estratégia ou combinado. Não use para
        sugestões, opiniões individuais ou frases ambíguas; nesses casos, peça
        confirmação ao grupo antes de registrar.
        """

        decisao_normalizada = (decisao or "").strip()
        if not decisao_normalizada:
            return _error("Decisão não pode ser vazia.")

        contexto_normalizado = contexto.strip() if isinstance(contexto, str) and contexto.strip() else None
        participantes_normalizados = [
            participante.strip()
            for participante in (participantes or [])
            if isinstance(participante, str) and participante.strip()
        ]

        record = DecisionRecord(
            decisao=decisao_normalizada,
            contexto=contexto_normalizado,
            participantes=participantes_normalizados,
            timestamp=_utc_now_iso(),
        )
        session.decisions.append(record)

        return {
            "ok": True,
            "decisao": record.decisao,
            "contexto": record.contexto,
            "participantes": list(record.participantes),
            "timestamp": record.timestamp,
            "total_decisoes": len(session.decisions),
        }

    @tool
    def consultar_inventario_grupo() -> dict[str, Any]:
        """Consulta o inventário compartilhado atual da party.

        Use esta ferramenta quando alguém perguntar o que o grupo possui,
        quantas unidades restam de um item, se há determinado recurso no
        inventário ou quando você precisar do estado real antes de responder.
        Nunca invente itens ou quantidades.
        """

        return {"ok": True, "inventario": dict(session.inventory)}

    @tool
    def consultar_decisoes_grupo(limite: int = 10) -> dict[str, Any]:
        """Consulta as decisões coletivas registradas na sessão.

        Use esta ferramenta quando alguém perguntar o que foi decidido, qual era
        o plano combinado, quais decisões recentes existem ou quando você
        precisar do histórico real antes de responder. Não invente decisões.
        """

        limite_seguro = max(1, min(int(limite or 10), 50))
        decisoes = session.decisions[-limite_seguro:]
        return {
            "ok": True,
            "limite": limite_seguro,
            "decisoes": [
                {
                    "decisao": decision.decisao,
                    "contexto": decision.contexto,
                    "participantes": list(decision.participantes),
                    "timestamp": decision.timestamp,
                }
                for decision in decisoes
            ],
            "total_decisoes": len(session.decisions),
        }

    @tool
    def buscar_manuais_rpg(query: str) -> str:
        """Busca informações nos manuais de RPG carregados em PDF.

        Use esta ferramenta quando o grupo fizer perguntas sobre regras,
        criação de personagem, monstros, condução de sessão, itens, magias,
        encontros, testes ou qualquer conteúdo que dependa dos manuais. A
        resposta da ferramenta traz trechos relevantes com nome do manual e
        página, que devem ser citados na resposta final. Se os manuais não
        contiverem a resposta, diga isso claramente ao grupo.
        """

        from app.rag import search_manuals

        return search_manuals(query)

    return [
        rolar_dados,
        atualizar_inventario_grupo,
        registrar_decisao_grupo,
        consultar_inventario_grupo,
        consultar_decisoes_grupo,
        buscar_manuais_rpg,
    ]
