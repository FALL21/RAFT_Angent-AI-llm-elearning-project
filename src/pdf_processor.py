"""
Module de traitement des documents PDF.
Extraction de texte, découpage en chunks, et prétraitement.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import src.config as cfg
from src.config import RAW_PDF_DIR, PROCESSED_DIR, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Pipeline complète de traitement des PDFs."""

    def __init__(
        self,
        pdf_dir: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.pdf_dir = pdf_dir or RAW_PDF_DIR
        cs = chunk_size if chunk_size is not None else cfg.CHUNK_SIZE
        co = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs,
            chunk_overlap=co,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_single_pdf(self, pdf_path: str) -> List[Document]:
        """Charge un seul fichier PDF."""
        logger.info(f"Chargement du PDF : {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logger.info(f"  → {len(pages)} pages extraites")
        return pages

    def load_all_pdfs(self) -> List[Document]:
        """Charge tous les PDFs du répertoire configuré."""
        logger.info(f"Chargement de tous les PDFs depuis : {self.pdf_dir}")
        loader = DirectoryLoader(
            str(self.pdf_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents = loader.load()
        logger.info(f"  → {len(documents)} pages chargées au total")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Découpe les documents en chunks."""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(
            f"  → {len(chunks)} chunks créés (taille={cfg.CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
        )
        return chunks

    def clean_text(self, text: str) -> str:
        """Nettoie le texte extrait des PDFs."""
        import re
        # Supprimer les sauts de ligne multiples
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Supprimer les espaces multiples
        text = re.sub(r" {2,}", " ", text)
        # Supprimer les en-têtes/pieds de page courants
        text = re.sub(r"Page \d+ of \d+", "", text)
        return text.strip()

    def process_pipeline(self) -> List[Document]:
        """
        Pipeline complète : chargement → nettoyage → découpage.
        Retourne les chunks prêts pour l'indexation.
        """
        logger.info("=" * 60)
        logger.info("PIPELINE DE TRAITEMENT DES PDFs")
        logger.info("=" * 60)

        # 1. Chargement
        documents = self.load_all_pdfs()
        if not documents:
            logger.warning("Aucun PDF trouvé ! Placez vos PDFs dans : %s", self.pdf_dir)
            return []

        # 2. Nettoyage
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)

        # 3. Découpage
        chunks = self.split_documents(documents)

        # 4. Enrichissement des métadonnées
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source_file"] = Path(chunk.metadata.get("source", "")).name

        logger.info(f"Pipeline terminée : {len(chunks)} chunks prêts")
        return chunks

    def get_stats(self, chunks: List[Document]) -> Dict:
        """Statistiques sur les chunks créés."""
        if not chunks:
            return {"total_chunks": 0}

        lengths = [len(c.page_content) for c in chunks]
        sources = set(c.metadata.get("source_file", "") for c in chunks)

        return {
            "total_chunks": len(chunks),
            "sources": list(sources),
            "avg_chunk_length": sum(lengths) / len(lengths),
            "min_chunk_length": min(lengths),
            "max_chunk_length": max(lengths),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = PDFProcessor()
    chunks = processor.process_pipeline()
    stats = processor.get_stats(chunks)
    print("\n📊 Statistiques :")
    for k, v in stats.items():
        print(f"  {k}: {v}")
