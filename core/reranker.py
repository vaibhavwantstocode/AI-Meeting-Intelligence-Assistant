import os

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None
_model_name = None


def _get_model() -> CrossEncoder:
    global _model, _model_name
    requested = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    if _model is None or _model_name != requested:
        print(f"Loading cross-encoder reranker: {requested} ...")
        _model = CrossEncoder(requested)
        _model_name = requested
        print("Cross-encoder reranker loaded.")
    return _model


def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    if not docs:
        return []
    if len(docs) <= 1:
        return docs[:top_k]

    model = _get_model()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    scored = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]
