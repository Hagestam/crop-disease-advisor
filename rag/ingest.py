"""
Ingests agricultural extension PDFs into ChromaDB vector store.
Run this once, or re-run when you add new documents.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCS_DIR        = Path("rag/documents")
VECTORSTORE_DIR = Path("rag/vectorstore")

def main():
    print(f"Loading documents from {DOCS_DIR}...")

    # Load all PDFs
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDFs")

    if not docs:
        print("No documents found! Add PDF files to rag/documents/")
        return

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # characters per chunk
        chunk_overlap=200,      # overlap to preserve context across chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings (local model, no API key needed)
    print("Loading embedding model (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Build and persist ChromaDB vector store
    print("Building vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="crop_advisory",
    )
    vectorstore.persist()
    print(f"✓ Vector store saved to {VECTORSTORE_DIR}")
    print(f"  Total vectors: {vectorstore._collection.count()}")

if __name__ == "__main__":
    main()