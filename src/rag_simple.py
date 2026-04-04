"""
ÉTAPE 3 — RAG Simple : Meilleur LLM + Base de connaissance vectorielle.
Architecture : Query → Retriever → LLM → Réponse.
"""
import logging
from typing import List, Dict, Optional

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import src.config as cfg
from src.config import RAG_SCORE_THRESHOLD
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Prompt RAG ──────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """Tu es un assistant expert. Utilise UNIQUEMENT le contexte ci-dessous pour répondre.
Si l'information n'est pas dans le contexte, dis-le clairement.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE (précise et structurée) :"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


class SimpleRAG:
    """
    RAG Simple — Retrieval-Augmented Generation basique.
    Pipeline : Question → Recherche vectorielle → Génération LLM.
    """

    def __init__(self, llm, vector_store: VectorStore, top_k: Optional[int] = None):
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k if top_k is not None else cfg.RAG_TOP_K
        self.chain = self._build_chain()

    def _build_chain(self):
        """Construit la chaîne RAG avec LangChain."""
        retriever = self.vector_store.get_retriever(search_kwargs={"k": self.top_k})

        def format_docs(docs: List[Document]) -> str:
            return "\n\n---\n\n".join(
                f"[Source: {d.metadata.get('source_file', 'N/A')}]\n{d.page_content}"
                for d in docs
            )

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, question: str) -> Dict:
        """Pose une question au système RAG."""
        logger.info(f"RAG Simple — Question : {question[:80]}...")

        # 1. Récupérer les documents
        docs = self.vector_store.similarity_search(question, k=self.top_k)
        logger.info(f"  → {len(docs)} documents récupérés")

        # 2. Générer la réponse
        answer = self.chain.invoke(question)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": d.page_content[:200],
                    "source": d.metadata.get("source_file", "N/A"),
                    "chunk_id": d.metadata.get("chunk_id", -1),
                }
                for d in docs
            ],
            "num_sources": len(docs),
            "pipeline": "rag_simple",
        }

    def query_with_scores(self, question: str) -> Dict:
        """Pose une question et retourne aussi les scores de similarité."""
        results_with_scores = self.vector_store.similarity_search_with_scores(
            question, k=self.top_k
        )

        docs = [doc for doc, _ in results_with_scores]
        scores = [score for _, score in results_with_scores]

        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source_file', 'N/A')} | Score: {s:.3f}]\n{d.page_content}"
            for d, s in zip(docs, scores)
        )

        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": answer.content if hasattr(answer, "content") else str(answer),
            "sources": [
                {
                    "content": d.page_content[:200],
                    "source": d.metadata.get("source_file", "N/A"),
                    "score": float(s),
                }
                for d, s in zip(docs, scores)
            ],
            "pipeline": "rag_simple_scored",
        }

    def batch_query(self, questions: List[str]) -> List[Dict]:
        """Traite un lot de questions."""
        return [self.query(q) for q in questions]


def create_simple_rag(llm, chunks: List[Document]) -> SimpleRAG:
    """Factory pour créer un RAG simple à partir de chunks."""
    vs = VectorStore()
    vs.create_from_documents(chunks)
    return SimpleRAG(llm=llm, vector_store=vs)
