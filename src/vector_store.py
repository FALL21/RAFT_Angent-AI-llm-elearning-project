"""
Module de base de données vectorielle.
Supporte ChromaDB et FAISS pour l'indexation et la recherche sémantique.
"""
import logging
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from overrides import override

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent

import src.config as cfg
from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR, VECTOR_DB_TYPE

logger = logging.getLogger(__name__)


class ChromaNoOpTelemetry(ProductTelemetryClient):
    """Ne appelle pas PostHog (évite capture() incompatible avec certaines versions)."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None


def _chroma_client_settings(persist_directory: str):
    """Settings Chroma : télémétrie désactivée + client produit no-op."""
    from chromadb.config import Settings

    noop = "src.vector_store.ChromaNoOpTelemetry"
    s = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        chroma_product_telemetry_impl=noop,
        chroma_telemetry_impl=noop,
    )
    s.persist_directory = persist_directory
    return s


# Chroma n’accepte que str / int / float / bool ; PyPDFLoader peut ajouter des types exotiques.
_CHROMA_META_STR_MAX = 4096


def _chroma_safe_key(key: Any) -> str:
    s = str(key).strip()
    if not s:
        return "_"
    s = re.sub(r"[^\w\-.]", "_", s, flags=re.ASCII)[:128]
    return s or "_"


def chroma_safe_metadata(meta: Optional[Dict]) -> Dict[str, Any]:
    """Métadonnées compatibles Chroma (évite 500 à l’indexation)."""
    if not meta:
        return {}
    out: Dict[str, Any] = {}
    for raw_k, v in meta.items():
        if v is None:
            continue
        k = _chroma_safe_key(raw_k)
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, int):
            out[k] = int(v)
        elif isinstance(v, float):
            out[k] = float(v)
        elif isinstance(v, str):
            out[k] = v[:_CHROMA_META_STR_MAX]
        else:
            out[k] = str(v)[:_CHROMA_META_STR_MAX]
    return out


def documents_for_chroma(documents: List[Document]) -> List[Document]:
    """Copie des documents avec métadonnées assainies pour Chroma."""
    return [
        Document(
            page_content=d.page_content,
            metadata=chroma_safe_metadata(d.metadata),
        )
        for d in documents
    ]


def clear_persisted_vectorstore(
    db_type: str = VECTOR_DB_TYPE,
    persist_dir: Optional[Path] = None,
) -> None:
    """Supprime les fichiers d’index sur disque (réindexation complète)."""
    base = persist_dir or VECTORSTORE_DIR
    if db_type == "chroma":
        path = base / "chroma"
        if path.is_dir():
            shutil.rmtree(path)
            logger.info("Index Chroma supprimé : %s", path)
            # Laisse le système de fichiers / SQLite relâcher le dossier (reloader Flask, index rapide).
            time.sleep(0.15)
    elif db_type == "faiss":
        path = base / "faiss"
        if path.is_dir():
            shutil.rmtree(path)
            logger.info("Index FAISS supprimé : %s", path)


class VectorStore:
    """Gestion de la base de données vectorielle."""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        db_type: str = VECTOR_DB_TYPE,
        persist_dir: Optional[Path] = None,
    ):
        self.db_type = db_type
        self.persist_dir = persist_dir or VECTORSTORE_DIR
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._db = None

    def create_from_documents(self, documents: List[Document], collection_name: str = "main"):
        """Crée la base vectorielle à partir des documents."""
        logger.info(f"Création de la base vectorielle ({self.db_type}) : {len(documents)} documents")

        if self.db_type == "chroma":
            from langchain_chroma import Chroma

            persist = str(self.persist_dir / "chroma")
            safe_docs = documents_for_chroma(documents)
            self._db = Chroma.from_documents(
                documents=safe_docs,
                embedding=self.embeddings,
                persist_directory=persist,
                client_settings=_chroma_client_settings(persist),
                collection_name=collection_name,
            )
        elif self.db_type == "faiss":
            from langchain_community.vectorstores import FAISS
            self._db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
            )
            self._db.save_local(str(self.persist_dir / "faiss"))

        logger.info("Base vectorielle créée avec succès")

    def load(self, collection_name: str = "main"):
        """Charge une base vectorielle existante."""
        logger.info(f"Chargement de la base vectorielle ({self.db_type})")

        if self.db_type == "chroma":
            from langchain_chroma import Chroma

            persist = str(self.persist_dir / "chroma")
            self._db = Chroma(
                persist_directory=persist,
                client_settings=_chroma_client_settings(persist),
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
        elif self.db_type == "faiss":
            from langchain_community.vectorstores import FAISS
            self._db = FAISS.load_local(
                str(self.persist_dir / "faiss"),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def similarity_search(
        self, query: str, k: Optional[int] = None, score_threshold: float = 0.0
    ) -> List[Document]:
        """Recherche les k documents les plus similaires."""
        if self._db is None:
            raise RuntimeError("Base vectorielle non initialisée. Appeler create_from_documents() ou load().")

        if k is None:
            k = cfg.RAG_TOP_K

        if score_threshold > 0 and hasattr(self._db, "similarity_search_with_relevance_scores"):
            results = self._db.similarity_search_with_relevance_scores(query, k=k)
            return [doc for doc, score in results if score >= score_threshold]

        return self._db.similarity_search(query, k=k)

    def similarity_search_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """Recherche avec scores de similarité."""
        if self._db is None:
            raise RuntimeError("Base vectorielle non initialisée.")
        if k is None:
            k = cfg.RAG_TOP_K
        return self._db.similarity_search_with_relevance_scores(query, k=k)

    def add_documents(self, documents: List[Document]):
        """Ajoute des documents à la base existante."""
        if self._db is None:
            raise RuntimeError("Base vectorielle non initialisée.")
        self._db.add_documents(documents)
        logger.info(f"{len(documents)} documents ajoutés à la base")

    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """Retourne un retriever LangChain."""
        if self._db is None:
            raise RuntimeError("Base vectorielle non initialisée.")
        kwargs = search_kwargs or {"k": cfg.RAG_TOP_K}
        return self._db.as_retriever(search_kwargs=kwargs)

    @property
    def count(self) -> int:
        """Nombre de documents dans la base."""
        if self._db is None:
            return 0
        if hasattr(self._db, "_collection"):
            return self._db._collection.count()
        return -1  # Inconnu pour FAISS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vs = VectorStore()
    print(f"VectorStore prêt (type={vs.db_type})")
