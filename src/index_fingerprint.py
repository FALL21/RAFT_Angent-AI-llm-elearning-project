"""
Empreinte des PDF dans data/raw_pdfs/ pour savoir si l’index vectoriel est à jour.
Sans cela, charger Chroma depuis le disque garde d’anciens vecteurs alors que de nouveaux fichiers ont été ajoutés.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from src.config import RAW_PDF_DIR, VECTORSTORE_DIR

FINGERPRINT_FILE = VECTORSTORE_DIR / ".pdf_sources_fingerprint"


def compute_raw_pdfs_fingerprint() -> str:
    """Hash stable : noms + taille + date de modification de chaque PDF."""
    paths = sorted(RAW_PDF_DIR.glob("**/*.pdf"), key=lambda p: p.as_posix().lower())
    lines: list[str] = []
    for p in paths:
        st = p.stat()
        rel = p.relative_to(RAW_PDF_DIR).as_posix()
        lines.append(f"{rel}\x00{st.st_size}\x00{st.st_mtime_ns}")
    blob = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_stored_fingerprint() -> Optional[str]:
    if not FINGERPRINT_FILE.is_file():
        return None
    text = FINGERPRINT_FILE.read_text(encoding="utf-8").strip()
    return text or None


def write_stored_fingerprint(digest: str) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_FILE.write_text(digest, encoding="utf-8")
