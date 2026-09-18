"""
ParamedicAI Document Ingestion
Loads WMS PDF guides, chunks them, embeds them, and stores in ChromaDB.

Usage:
    python ingest.py            # Ingest (skips if already done)
    python ingest.py --force    # Force re-ingestion
"""

import os
import sys
import argparse
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    DOCUMENTS_DIR,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)


def get_pdf_files():
    """Get all PDF files from the Documents directory."""
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"Error: Documents directory not found at {DOCUMENTS_DIR}")
        sys.exit(1)

    pdf_files = [
        os.path.join(DOCUMENTS_DIR, f)
        for f in sorted(os.listdir(DOCUMENTS_DIR))
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"Error: No PDF files found in {DOCUMENTS_DIR}")
        sys.exit(1)

    return pdf_files


def load_and_chunk_pdfs(pdf_files):
    """Load PDFs and split into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"  Loading: {filename}...", end=" ")

        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            # Add source filename to metadata
            for page in pages:
                page.metadata["source_file"] = filename

            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"{len(chunks)} chunks")
        except Exception as e:
            print(f"FAILED ({e})")

    return all_chunks


def create_vector_store(chunks):
    """Embed chunks and store in ChromaDB."""
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Embedding {len(chunks)} chunks and storing in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )

    return vector_store


def ingest(force=False):
    """Main ingestion pipeline."""
    # Check if already ingested
    if os.path.exists(CHROMA_DB_DIR) and not force:
        print(f"Vector store already exists at {CHROMA_DB_DIR}")
        print("Use --force to re-ingest.")
        return

    # Clear existing store if forcing
    if force and os.path.exists(CHROMA_DB_DIR):
        print("Removing existing vector store...")
        shutil.rmtree(CHROMA_DB_DIR)

    print("=" * 60)
    print("ParamedicAI Document Ingestion")
    print("=" * 60)

    # Step 1: Find PDFs
    pdf_files = get_pdf_files()
    print(f"\nFound {len(pdf_files)} PDF files in {DOCUMENTS_DIR}\n")

    # Step 2: Load and chunk
    print("Step 1/2: Loading and chunking PDFs...")
    chunks = load_and_chunk_pdfs(pdf_files)
    print(f"\nTotal chunks: {len(chunks)}")

    # Step 3: Embed and store
    print("\nStep 2/2: Embedding and storing...")
    vector_store = create_vector_store(chunks)

    print("\n" + "=" * 60)
    print(f"Ingestion complete!")
    print(f"  Chunks stored: {len(chunks)}")
    print(f"  Vector store:  {CHROMA_DB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest WMS PDF documents into ChromaDB")
    parser.add_argument(
        "--force", action="store_true", help="Force re-ingestion even if vector store exists"
    )
    args = parser.parse_args()
    ingest(force=args.force)
