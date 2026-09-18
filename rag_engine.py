"""
ParamedicAI RAG Engine
Core class that handles retrieval and generation for medical emergency queries.

Usage:
    from rag_engine import ParamedicAI

    bot = ParamedicAI()
    result = bot.query("My friend fell off a roof and his neck hurts...")
    print(result["response"])
    print(result["sources"])
"""

import os
import sys

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    CHROMA_DB_DIR,
    OLLAMA_BASE_URL,
    MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
    TOP_K,
    SYSTEM_PROMPT,
)


class ParamedicAI:
    """RAG-powered medical emergency assistant."""

    def __init__(self):
        """Initialize the RAG engine: load vector store and connect to Ollama."""
        if not os.path.exists(CHROMA_DB_DIR):
            print("Vector store not found. Running document ingestion first...")
            from ingest import ingest
            ingest()

        # Load embedding model
        print("Loading embedding model...")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load vector store
        print("Loading vector store...")
        self._vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self._embeddings,
        )
        self._retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )

        # Connect to Ollama
        print(f"Connecting to Ollama ({MODEL_NAME})...")
        self._llm = OllamaLLM(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,  # Low temperature for factual medical responses
            num_predict=900,
        )

        # Build the RAG chain
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        self._chain = (
            {
                "context": self._retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self._prompt
            | self._llm
            | StrOutputParser()
        )

        print("ParamedicAI ready.\n")

    @staticmethod
    def _format_docs(docs):
        """Format retrieved documents into a context string."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(
                f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def query(self, user_input: str, source_document: str | None = None) -> dict:
        """
        Process a medical emergency query.

        Args:
            user_input: The user's emergency scenario description.
            source_document: Optional source filename to restrict retrieval to
                the guideline used for an evaluation case.

        Returns:
            dict with:
                - "response": The AI's emergency guidance (str)
                - "sources": List of source document filenames consulted (list[str])
        """
        # Evaluation cases can pin retrieval to their declared source. Normal
        # interactive queries continue to use similarity retrieval across the store.
        if source_document:
            retrieved_docs = self._vector_store.similarity_search(
                user_input,
                k=TOP_K,
                filter={"source_file": source_document},
            )
        else:
            retrieved_docs = self._retriever.invoke(user_input)

        # Extract unique source filenames
        sources = list({
            doc.metadata.get("source_file", "Unknown")
            for doc in retrieved_docs
        })
        sources.sort()

        # Generate response using the same retrieved documents used for sources.
        if source_document:
            response = (
                self._prompt
                | self._llm
                | StrOutputParser()
            ).invoke({
                "context": self._format_docs(retrieved_docs),
                "question": user_input,
            })
        else:
            response = self._chain.invoke(user_input)

        return {
            "response": response,
            "sources": sources,
        }

    def query_stream(self, user_input: str):
        """
        Stream a medical emergency query response token by token.

        Args:
            user_input: The user's emergency scenario description.

        Yields:
            str: Response tokens as they are generated.
        """
        for chunk in self._chain.stream(user_input):
            yield chunk
