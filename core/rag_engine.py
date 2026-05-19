import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI

from core.vector_store import build_vector_store, get_retriever, load_vector_store
from utils.retrying import invoke_chain


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def _format_docs_with_refs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        chunk_index = doc.metadata.get("chunk_index", "?")
        parts.append(f"[chunk {chunk_index}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_chain(vector_store):
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

Each transcript excerpt is labelled with its chunk number, like "[chunk 3]".
When you use information from an excerpt, cite it inline as (chunk 3).

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    answer_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs_with_refs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"answer": answer_chain, "retriever": retriever}


def build_rag_chain(transcript: str, session_id: str, source: str = ""):
    vector_store = build_vector_store(transcript, session_id=session_id, source=source)
    return _build_chain(vector_store)


def load_rag_chain(session_id: str):
    vector_store = load_vector_store(session_id)
    return _build_chain(vector_store)


def ask_question(rag_chain, question: str) -> dict:
    if rag_chain is None:
        return {
            "answer": "RAG chat is unavailable because the retrieval engine could not be built.",
            "sources": [],
        }

    print(f"Question: {question}")
    sources = rag_chain["retriever"].invoke(question)
    answer = invoke_chain(rag_chain["answer"], question)
    print(f"Answer: {answer}")

    return {
        "answer": answer,
        "sources": [
            {
                "chunk_index": doc.metadata.get("chunk_index"),
                "session_id": doc.metadata.get("session_id"),
                "source": doc.metadata.get("source"),
                "preview": doc.page_content,
            }
            for doc in sources
        ],
    }
