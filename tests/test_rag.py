from pathlib import Path

from app import rag
from app.tools import build_tools_for_session


class DummyDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class EmptyRetriever:
    def invoke(self, query):
        return []


def test_discover_pdf_paths_returns_sorted_limited_pdfs(tmp_path):
    for name in ["zeta.pdf", "alfa.pdf", "beta.txt", "delta.pdf", "gama.pdf", "epsilon.pdf", "omega.pdf"]:
        (tmp_path / name).write_text("conteudo", encoding="utf-8")

    result = rag.discover_pdf_paths(tmp_path, limit=3)

    assert [path.name for path in result] == ["alfa.pdf", "delta.pdf", "epsilon.pdf"]


def test_discover_pdf_paths_missing_directory_returns_empty(tmp_path):
    assert rag.discover_pdf_paths(tmp_path / "nao-existe") == []


def test_format_rag_documents_includes_source_and_one_based_page():
    docs = [
        DummyDoc(
            "Texto relevante do manual.",
            {"manual_name": "manual-jogador.pdf", "page": 4},
        )
    ]

    result = rag.format_rag_documents(docs)

    assert "manual-jogador.pdf" in result
    assert "página 5" in result
    assert "Texto relevante do manual." in result


def test_search_manuals_empty_result_has_clear_message(monkeypatch):
    monkeypatch.setattr(rag, "get_retriever", lambda: EmptyRetriever())

    result = rag.search_manuals("pergunta sem resposta")

    assert "Nenhum trecho relevante" in result


def test_tools_include_rag_tool():
    tools = build_tools_for_session("mesa-teste-rag")

    assert "buscar_manuais_rpg" in {tool.name for tool in tools}
