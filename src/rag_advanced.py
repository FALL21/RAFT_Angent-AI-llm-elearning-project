"""
ÉTAPE 4 — RAG Avancé : RAG + couches d'optimisation.
Optimisations :
  • Reranking avec Cross-Encoder
  • Expansion de requête (HyDE — Hypothetical Document Embeddings)
  • Recherche hybride (dense + sparse / BM25)
  • Contextual Compression
  • Multi-Query Retriever
"""
import logging
from typing import List, Dict, Optional

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter

import src.config as cfg
from src.config import RERANKER_MODEL
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Prompts ─────────────────────────────────────────────────
ADVANCED_RAG_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un assistant expert avec accès à une base de connaissances vérifiée.
Analyse le contexte fourni avec rigueur et réponds de manière détaillée.
Cite les sources pertinentes dans ta réponse.

CONTEXTE (classé par pertinence) :
{context}

QUESTION : {question}

INSTRUCTIONS :
1. Synthétise les informations des sources les plus pertinentes.
2. Si les sources se contredisent, mentionne-le.
3. Structure ta réponse clairement.

RÉPONSE :"""
)

HYDE_PROMPT = ChatPromptTemplate.from_template(
    """Génère un court paragraphe hypothétique qui répondrait parfaitement à cette question.
Ce paragraphe sera utilisé pour la recherche sémantique.

Question : {question}

Paragraphe hypothétique :"""
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """Génère 3 reformulations différentes de la question suivante pour améliorer la recherche.
Retourne chaque reformulation sur une ligne séparée, sans numérotation.

Question originale : {question}

Reformulations :"""
)


class CrossEncoderReranker:
    """Reranker basé sur un Cross-Encoder pour réordonner les documents."""

    def __init__(self, model_name: str = RERANKER_MODEL, top_k: int = 3):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Réordonne les documents par pertinence."""
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )

        reranked = []
        for doc, score in scored_docs[: self.top_k]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)

        logger.info(f"Reranking : {len(documents)} → {len(reranked)} documents (top scores: "
                     f"{[f'{s:.3f}' for _, s in scored_docs[:3]]})")
        return reranked


class BM25Retriever:
    """Retriever BM25 (sparse) pour la recherche hybride."""

    def __init__(self, documents: List[Document]):
        from rank_bm25 import BM25Okapi
        self.documents = documents
        corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> List[Document]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_indices]


class AdvancedRAG:
    """
    RAG Avancé avec couches d'optimisation multiples.
    """

    def __init__(
        self,
        llm,
        vector_store: VectorStore,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
        use_hyde: bool = True,
        use_hybrid: bool = True,
        use_multi_query: bool = True,
        all_documents: Optional[List[Document]] = None,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k if top_k is not None else cfg.RAG_TOP_K
        self.use_reranker = use_reranker
        self.use_hyde = use_hyde
        self.use_hybrid = use_hybrid
        self.use_multi_query = use_multi_query

        # Initialiser les composants optionnels
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.bm25 = BM25Retriever(all_documents) if use_hybrid and all_documents else None

    # ── HyDE : Hypothetical Document Embeddings ─────────────
    def _hyde_expand(self, question: str) -> str:
        """Génère un document hypothétique pour améliorer la recherche."""
        chain = HYDE_PROMPT | self.llm | StrOutputParser()
        hypothetical = chain.invoke({"question": question})
        logger.info(f"HyDE : document hypothétique généré ({len(hypothetical)} chars)")
        return hypothetical

    # ── Multi-Query ─────────────────────────────────────────
    def _multi_query_expand(self, question: str) -> List[str]:
        """Génère plusieurs reformulations de la question."""
        chain = MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"question": question})
        queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
        queries.insert(0, question)  # Garder la question originale
        logger.info(f"Multi-Query : {len(queries)} requêtes générées")
        return queries[:4]  # Max 4 requêtes

    # ── Recherche hybride ───────────────────────────────────
    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """Combine recherche dense (vectorielle) et sparse (BM25)."""
        dense_results = self.vector_store.similarity_search(query, k=k)

        if self.bm25:
            sparse_results = self.bm25.search(query, k=k)
            # Fusion RRF (Reciprocal Rank Fusion)
            return self._reciprocal_rank_fusion(dense_results, sparse_results, k=k)

        return dense_results

    def _reciprocal_rank_fusion(
        self, list1: List[Document], list2: List[Document], k: int = 5, rrf_k: int = 60
    ) -> List[Document]:
        """Fusionne deux listes de résultats avec Reciprocal Rank Fusion."""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(list1):
            key = doc.page_content[:100]
            doc_scores[key] = doc_scores.get(key, 0) + 1 / (rank + rrf_k)
            doc_map[key] = doc

        for rank, doc in enumerate(list2):
            key = doc.page_content[:100]
            doc_scores[key] = doc_scores.get(key, 0) + 1 / (rank + rrf_k)
            doc_map[key] = doc

        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)[:k]
        return [doc_map[key] for key in sorted_keys]

    # ── Pipeline principal ──────────────────────────────────
    def query(self, question: str) -> Dict:
        """Pipeline RAG avancé complet."""
        logger.info(f"\nRAG Avancé — Question : {question[:80]}...")
        all_docs = []

        # 1. Expansion HyDE
        search_queries = [question]
        if self.use_hyde:
            hyde_doc = self._hyde_expand(question)
            search_queries.append(hyde_doc)

        # 2. Multi-Query
        if self.use_multi_query:
            search_queries = self._multi_query_expand(question)

        # 3. Recherche (hybride ou dense)
        for sq in search_queries:
            if self.use_hybrid:
                docs = self._hybrid_search(sq, k=self.top_k)
            else:
                docs = self.vector_store.similarity_search(sq, k=self.top_k)
            all_docs.extend(docs)

        # Dédoublonner
        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        # 4. Reranking
        if self.reranker and unique_docs:
            unique_docs = self.reranker.rerank(question, unique_docs)

        # 5. Génération
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source_file', 'N/A')}]\n{d.page_content}"
            for d in unique_docs[:self.top_k]
        )

        chain = ADVANCED_RAG_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": d.page_content[:200],
                    "source": d.metadata.get("source_file", "N/A"),
                    "rerank_score": d.metadata.get("rerank_score"),
                }
                for d in unique_docs[:self.top_k]
            ],
            "num_sources": len(unique_docs),
            "optimizations": {
                "hyde": self.use_hyde,
                "multi_query": self.use_multi_query,
                "hybrid_search": self.use_hybrid,
                "reranker": self.use_reranker,
            },
            "pipeline": "rag_advanced",
        }
