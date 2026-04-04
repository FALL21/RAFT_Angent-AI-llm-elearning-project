#!/usr/bin/env python3
"""
Téléchargement de PDFs pour le domaine E-LEARNING.
Sources : articles en libre accès (arXiv, ERIC, WJARR, MDPI, etc.).

Usage :
    python scripts/download_pdfs.py
    python scripts/download_pdfs.py --list     # Afficher la liste sans télécharger
"""
import os
import sys
import argparse
import requests
from pathlib import Path

# Répertoire de destination
BASE_DIR = Path(__file__).parent.parent
PDF_DIR = BASE_DIR / "data" / "raw_pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════
# PDFs E-LEARNING en libre accès (Open Access / Creative Commons)
# ════════════════════════════════════════════════════════════

PDF_SOURCES = [
    # ── IA & Education ──────────────────────────────────────
    {
        "name": "01_educators_academic_insights_AI.pdf",
        "url": "https://files.eric.ed.gov/fulltext/EJ1425485.pdf",
        "title": "Educators' Academic Insights on Artificial Intelligence in Education",
        "source": "ERIC / Electronic Journal of e-Learning (2024)",
        "topics": ["IA en éducation", "perception des enseignants", "ChatGPT"],
    },
    {
        "name": "02_role_AI_education_conceptual_review.pdf",
        "url": "https://wjarr.com/sites/default/files/WJARR-2024-1217.pdf",
        "title": "Role of Artificial Intelligence in Education: A Conceptual Review",
        "source": "World Journal of Advanced Research and Reviews (2024)",
        "topics": ["IA en éducation", "apprentissage adaptatif", "gamification"],
    },
    {
        "name": "03_AI_literacy_education.pdf",
        "url": "http://elearning.tame.org.tw/filecenter/A/8DCC11DEB331CAD071/8DCC11DEB331CAD0712.pdf",
        "title": "AI Literacy in Education: Competence Areas and Frameworks",
        "source": "Computers and Education Open (2024)",
        "topics": ["littératie IA", "compétences numériques", "K-12"],
    },
    {
        "name": "04_LLM_good_tutors_english_education.pdf",
        "url": "https://arxiv.org/pdf/2502.05467",
        "title": "Position: LLMs Can be Good Tutors in English Education",
        "source": "arXiv (2025)",
        "topics": ["LLM tuteur", "enseignement des langues", "feedback automatique"],
    },
    {
        "name": "05_intelligent_tutoring_systems_review.pdf",
        "url": "https://arxiv.org/pdf/2507.18882",
        "title": "A Comprehensive Review of AI-based Intelligent Tutoring Systems",
        "source": "arXiv (2025)",
        "topics": ["tuteur intelligent", "apprentissage adaptatif", "NLP éducatif"],
    },
    # ── E-Learning & Technologies ───────────────────────────
    {
        "name": "06_AI_ML_higher_education.pdf",
        "url": "https://www.mdpi.com/2071-1050/13/18/10424/pdf",
        "title": "Opportunities and Challenges of AI and ML in Higher Education",
        "source": "MDPI Sustainability (2021) — Open Access",
        "topics": ["ML en éducation supérieure", "e-learning", "défis IA"],
    },
    {
        "name": "07_LLMs_education_NLP_perspective.pdf",
        "url": "https://arxiv.org/pdf/2412.00477",
        "title": "Opportunities and Challenges of LLMs in Education: An NLP Perspective",
        "source": "arXiv (2024)",
        "topics": ["NLP éducatif", "évaluation automatique", "tuteur IA"],
    },
    {
        "name": "08_generative_AI_education_review.pdf",
        "url": "https://arxiv.org/pdf/2305.18734",
        "title": "A Survey on Generative AI and LLMs for Education",
        "source": "arXiv (2023)",
        "topics": ["IA générative", "MOOC", "contenu automatique"],
    },
    {
        "name": "09_adaptive_learning_deep_learning.pdf",
        "url": "https://arxiv.org/pdf/2402.14601",
        "title": "Adaptive Learning Systems with Deep Learning",
        "source": "arXiv (2024)",
        "topics": ["apprentissage adaptatif", "deep learning", "personnalisation"],
    },
    {
        "name": "10_RAG_survey_comprehensive.pdf",
        "url": "https://arxiv.org/pdf/2312.10997",
        "title": "Retrieval-Augmented Generation for LLMs: A Survey",
        "source": "arXiv (2024)",
        "topics": ["RAG", "retrieval augmented generation", "survey"],
    },
]

# ════════════════════════════════════════════════════════════
# SOURCES ALTERNATIVES (si téléchargement impossible)
# ════════════════════════════════════════════════════════════
ALTERNATIVE_SOURCES = """
======================================================================
  SOURCES ALTERNATIVES POUR OBTENIR DES PDFs E-LEARNING MANUELLEMENT
======================================================================

  1. ERIC (Education Resources) — https://eric.ed.gov
     -> Chercher : "e-learning" "artificial intelligence"
     -> Filtrer : Full Text Available
     -> PDFs gratuits et académiques

  2. arXiv — https://arxiv.org
     -> Chercher : cs.CY + "education" ou cs.AI + "tutoring"
     -> Tous les PDFs sont en libre accès

  3. Google Scholar — https://scholar.google.com
     -> Chercher : "e-learning" "LLM" "NLP"
     -> Cliquer sur [PDF] a droite des résultats

  4. MDPI (Open Access) — https://www.mdpi.com
     -> Journaux : Education Sciences, Sustainability
     -> Tous les articles sont en Open Access

  5. UNESCO IITE — https://iite.unesco.org
     -> Publications sur les technologies éducatives

  6. HAL Archives Ouvertes — https://hal.science
     -> Chercher : "e-learning" "intelligence artificielle"
     -> Articles en français et anglais

  7. PubMed Central — https://www.ncbi.nlm.nih.gov/pmc
     -> Chercher : "e-learning" "medical education AI"
     -> Articles biomédicaux en libre accès
======================================================================
"""


def download_pdf(url: str, filepath: Path, timeout: int = 60) -> bool:
    """Télécharge un PDF depuis une URL."""
    try:
        print(f"  Telechargement...")
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Academic Research Bot)",
        }
        response = requests.get(url, timeout=timeout, stream=True, headers=headers)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                total += len(chunk)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb < 0.01:
            filepath.unlink()
            print(f"  ERREUR: Fichier trop petit ({size_mb:.3f} MB)")
            return False

        print(f"  OK - {size_mb:.1f} MB")
        return True

    except requests.exceptions.Timeout:
        print(f"  ERREUR: Timeout apres {timeout}s")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"  ERREUR HTTP : {e.response.status_code}")
        return False
    except Exception as e:
        print(f"  ERREUR : {e}")
        return False


def list_sources():
    """Affiche la liste des PDFs disponibles."""
    print("\n" + "=" * 70)
    print("  PDFs E-LEARNING DISPONIBLES AU TELECHARGEMENT")
    print("=" * 70)
    for i, pdf in enumerate(PDF_SOURCES, 1):
        print(f"\n  [{i:02d}] {pdf['title']}")
        print(f"       Source  : {pdf['source']}")
        print(f"       Topics  : {', '.join(pdf['topics'])}")
        print(f"       Fichier : {pdf['name']}")
        print(f"       URL     : {pdf['url']}")
    print(ALTERNATIVE_SOURCES)


def main():
    parser = argparse.ArgumentParser(description="Telechargement de PDFs e-learning")
    parser.add_argument("--list", action="store_true", help="Afficher la liste sans telecharger")
    parser.add_argument("--index", type=int, nargs="+",
                        help="Telecharger uniquement les PDFs aux indices donnes (1-10)")
    args = parser.parse_args()

    if args.list:
        list_sources()
        return

    sources = PDF_SOURCES
    if args.index:
        sources = [PDF_SOURCES[i - 1] for i in args.index if 1 <= i <= len(PDF_SOURCES)]

    print("\n" + "=" * 70)
    print("  TELECHARGEMENT DES PDFs — DOMAINE E-LEARNING")
    print(f"  Destination : {PDF_DIR}")
    print("=" * 70)

    total = len(sources)
    success = 0
    skipped = 0

    for i, pdf in enumerate(sources, 1):
        filepath = PDF_DIR / pdf["name"]
        print(f"\n[{i}/{total}] {pdf['title']}")
        print(f"  Source : {pdf['source']}")

        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  Existe deja ({size_mb:.1f} MB)")
            success += 1
            skipped += 1
            continue

        if download_pdf(pdf["url"], filepath):
            success += 1

    print(f"\n{'=' * 70}")
    print(f"  RESULTAT : {success}/{total} PDFs disponibles ({skipped} deja existants)")
    print(f"  Dossier  : {PDF_DIR}")
    print(f"{'=' * 70}")

    if success < total:
        print("\n  Certains PDFs n'ont pas pu etre telecharges.")
        print("  Consultez les sources alternatives :")
        print(ALTERNATIVE_SOURCES)

    print("\n  Prochaine etape :")
    print("   python run.py --step serve")
    print("   -> Onglet Documents -> Indexer les documents")


if __name__ == "__main__":
    main()
