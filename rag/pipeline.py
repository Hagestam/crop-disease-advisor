"""
RAG pipeline for generating crop disease treatment advisories.
Uses ChromaDB + LangChain + Claude API.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

VECTORSTORE_DIR = Path("rag/vectorstore")

ADVISORY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert agronomist and crop disease specialist.
Using only the context below from agricultural extension documents, answer the question.
If the context does not contain enough information, say so clearly and give general best practices.

Context:
{context}

Question: {question}

Answer in a clear, farmer-friendly format with these sections:
1. **Disease Overview** – Brief explanation of the disease
2. **Symptoms** – What to look for on the plant
3. **Treatment Options** – Chemical and organic treatments with product examples
4. **Preventive Measures** – How to avoid future outbreaks
5. **Urgency** – How quickly should the farmer act?

Keep it practical and actionable."""
)


class CropAdvisoryRAG:
    """
    RAG pipeline for crop disease advisory.
    Retrieves relevant extension document chunks and generates
    structured treatment recommendations via Claude.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain = None
        self._load()

    def _load(self):
        if not VECTORSTORE_DIR.exists():
            print("Warning: Vector store not found. Run rag/ingest.py first.")
            return

        self.vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=self.embeddings,
            collection_name="crop_advisory",
        )
        print(f"Loaded vector store with {self.vectorstore._collection.count()} vectors")
        self._build_chain()

    def _build_chain(self):
        llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0.3,
            max_tokens=1500,
        )
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",          # Maximum Marginal Relevance — more diverse results
            search_kwargs={"k": 6, "fetch_k": 20},
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": ADVISORY_PROMPT},
            return_source_documents=True,
        )

    def get_advisory(self, crop: str, disease: str) -> dict:
        """
        Returns treatment advisory for a detected disease.
        
        Args:
            crop:    e.g. "Tomato"
            disease: e.g. "Late blight"
        
        Returns:
            {"advisory": str, "sources": list[str]}
        """
        if self.qa_chain is None:
            return {
                "advisory": "Advisory system not available. Please run rag/ingest.py first.",
                "sources": [],
            }

        query = (
            f"What are the treatment recommendations and management strategies for "
            f"{disease} disease in {crop} crops? "
            f"Include symptoms, chemical and organic treatment options, and prevention."
        )

        result = self.qa_chain({"query": query})
        sources = [
            doc.metadata.get("source", "Unknown")
            for doc in result.get("source_documents", [])
        ]
        # Deduplicate sources
        sources = list(dict.fromkeys(sources))

        return {
            "advisory": result["result"],
            "sources": sources,
        }

    def query(self, question: str) -> dict:
        """Free-form query against the document store."""
        if self.qa_chain is None:
            return {"advisory": "System not ready.", "sources": []}
        result = self.qa_chain({"query": question})
        return {
            "advisory": result["result"],
            "sources": [d.metadata.get("source", "") for d in result.get("source_documents", [])],
        }


# Singleton — loaded once when the module is imported
_rag_instance: Optional[CropAdvisoryRAG] = None

def get_rag() -> CropAdvisoryRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = CropAdvisoryRAG()
    return _rag_instance