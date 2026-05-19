import shutil
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.retrying import retry_external_call

CHROMA_ROOT = Path("vector_db")
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def session_persist_dir(session_id: str) -> str:
    return str(CHROMA_ROOT / session_id)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


@retry_external_call
def _build_chroma_from_documents(docs: list[Document], embeddings, persist_dir: str) -> Chroma:
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )


def build_vector_store(transcript: str, session_id: str, source: str = "") -> Chroma:
    print(f"Building vector store for session {session_id}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "session_id": session_id,
                "chunk_index": i,
                "source": source,
                "start_time": None,
                "end_time": None,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    persist_dir = session_persist_dir(session_id)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    return _build_chroma_from_documents(docs, embeddings, persist_dir)


def load_vector_store(session_id: str) -> Chroma:
    persist_dir = session_persist_dir(session_id)
    if not Path(persist_dir).exists():
        raise FileNotFoundError(f"No vector store found for session '{session_id}' at {persist_dir}")

    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def delete_session_store(session_id: str) -> None:
    persist_dir = Path(session_persist_dir(session_id))
    if persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
