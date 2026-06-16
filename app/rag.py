"""RAG utilities for searching the Co-DM RPG manuals."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings


COLLECTION_PREFIX = "codm_manuals"
MAX_PDFS = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def discover_pdf_paths(manuals_dir: str | Path, limit: int = MAX_PDFS) -> list[Path]:
    """Return up to ``limit`` PDFs from the manuals directory in stable order."""

    directory = Path(manuals_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(directory.glob("*.pdf"), key=lambda path: path.name.lower())[:limit]


def _pdf_signature(pdf_paths: list[Path], embedding_model: str) -> str:
    payload = {
        "embedding_model": embedding_model,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "pdfs": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in pdf_paths
        ],
    }
    return json.dumps(payload, sort_keys=True)


def _collection_name(pdf_paths: list[Path], embedding_model: str) -> str:
    digest = hashlib.sha256(_pdf_signature(pdf_paths, embedding_model).encode("utf-8")).hexdigest()[:12]
    return f"{COLLECTION_PREFIX}_{digest}"


@lru_cache(maxsize=4)
def _get_embeddings(embedding_model: str) -> Any:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=embedding_model)


def _load_pdf_chunks(pdf_paths: list[Path]) -> list[Any]:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages: list[Any] = []
    for pdf_path in pdf_paths:
        for page in PyPDFLoader(str(pdf_path)).load():
            page.metadata["manual_name"] = pdf_path.name
            pages.append(page)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def _collection_count(vectorstore: Any) -> int:
    try:
        return int(vectorstore._collection.count())
    except Exception:
        return 0


@lru_cache(maxsize=4)
def get_retriever(
    manuals_dir: str | None = None,
    chroma_dir: str | None = None,
    embedding_model: str | None = None,
    retriever_k: int | None = None,
) -> Any:
    """Build or load a Chroma retriever for the configured manuals."""

    settings = get_settings()
    resolved_manuals_dir = manuals_dir or settings.manuals_dir
    resolved_chroma_dir = chroma_dir or settings.chroma_dir
    resolved_embedding_model = embedding_model or settings.embedding_model
    resolved_retriever_k = retriever_k or settings.retriever_k

    pdf_paths = discover_pdf_paths(resolved_manuals_dir)
    if not pdf_paths:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {resolved_manuals_dir}.")

    from langchain_chroma import Chroma

    Path(resolved_chroma_dir).mkdir(parents=True, exist_ok=True)
    embeddings = _get_embeddings(resolved_embedding_model)
    collection_name = _collection_name(pdf_paths, resolved_embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=resolved_chroma_dir,
    )

    if _collection_count(vectorstore) == 0:
        chunks = _load_pdf_chunks(pdf_paths)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=resolved_chroma_dir,
            collection_name=collection_name,
        )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": max(1, int(resolved_retriever_k))},
    )


def format_rag_documents(docs: list[Any], max_chars_per_doc: int = 1400) -> str:
    """Format retrieved documents with source and page metadata for the LLM."""

    if not docs:
        return "Nenhum trecho relevante foi encontrado nos manuais."

    formatted: list[str] = []
    for index, doc in enumerate(docs, 1):
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("manual_name") or Path(str(metadata.get("source", "manual desconhecido"))).name
        page = metadata.get("page", "?")
        if isinstance(page, int):
            page = page + 1
        content = " ".join(str(getattr(doc, "page_content", "")).split())
        if len(content) > max_chars_per_doc:
            content = f"{content[:max_chars_per_doc].rstrip()}..."
        formatted.append(f"[Trecho {index} | {source} | página {page}]\n{content}")
    return "\n\n".join(formatted)


def search_manuals(query: str) -> str:
    """Search the configured RPG manuals and return formatted excerpts."""

    normalized_query = (query or "").strip()
    if not normalized_query:
        return "Informe uma pergunta ou termo de busca para consultar os manuais."

    try:
        retriever = get_retriever()
        docs = retriever.invoke(normalized_query)
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return (
            "Não consegui consultar os manuais agora. "
            f"Verifique as dependências de RAG e os PDFs configurados. Detalhe: {exc}"
        )

    return format_rag_documents(list(docs))
