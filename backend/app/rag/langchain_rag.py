"""
backend/app/rag/langchain_rag.py
----------------------------------------------------
LangChain Retriever Adapter for Genkit AI Hybrid RAG Engine.
Wraps BM25, TF-IDF, and Qdrant Dense Vector retrieval into a standard
LangChain BaseRetriever compatible with LCEL pipelines.
"""

from typing import List, Optional
from pydantic import Field

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.rag.pipeline import get_rag_pipeline, HybridRAGPipeline
from app.llm.langchain_wrapper import GenkitLLM


class GenkitRAGRetriever(BaseRetriever):
    """
    LangChain BaseRetriever wrapper over Genkit AI HybridRAGPipeline.
    Returns standard LangChain Document objects.
    """

    top_k: int = Field(default=3, description="Number of top chunks to retrieve")

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Synchronously retrieves relevant documents via Hybrid RAG."""
        pipeline = get_rag_pipeline()
        chunks, confidence, is_grounded = pipeline.retrieve(query, top_k=self.top_k)

        documents: List[Document] = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "id": chunk.id,
                    "title": chunk.title,
                    "category": chunk.category,
                    "source": chunk.source,
                    "score": getattr(chunk, "score", 0.0),
                    "confidence": confidence,
                    "is_grounded": is_grounded,
                },
            )
            documents.append(doc)

        return documents


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved LangChain documents into clean context string."""
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", f"Document {i}")
        formatted_chunks.append(f"[{i}] {title}\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


def create_genkit_rag_chain(
    llm: Optional[GenkitLLM] = None,
    retriever: Optional[GenkitRAGRetriever] = None,
):
    """
    Creates an end-to-end LangChain LCEL RAG chain:
    question -> retriever -> prompt template -> custom LLM -> string output.
    """
    llm = llm or GenkitLLM()
    retriever = retriever or GenkitRAGRetriever(top_k=3)

    template = (
        "SYSTEM: You are Genkit AI, an enterprise assistant. Answer the user question accurately using ONLY the provided context.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER:"
    )
    prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

