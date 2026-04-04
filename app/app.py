"""
Application Flask — Interface web interactive pour le système RAG/LLM Multi-Agent.
"""
import os
import sys
import json
import logging
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FLASK_CONFIG, RAW_PDF_DIR, EVALUATION_DIR, VECTORSTORE_DIR, VECTOR_DB_TYPE
from src.index_fingerprint import (
    compute_raw_pdfs_fingerprint,
    read_stored_fingerprint,
    write_stored_fingerprint,
)
from src.vector_store import clear_persisted_vectorstore

logger = logging.getLogger(__name__)

# ── Variables globales des pipelines ────────────────────────
pipelines = {}


def _chroma_persist_exists() -> bool:
    """True si un index Chroma a déjà été écrit sur le disque."""
    root = VECTORSTORE_DIR / "chroma"
    if not root.is_dir():
        return False
    sqlite = root / "chroma.sqlite3"
    if sqlite.is_file() and sqlite.stat().st_size > 0:
        return True
    return any(root.iterdir())


def _faiss_persist_exists() -> bool:
    d = VECTORSTORE_DIR / "faiss"
    return d.is_dir() and any(d.iterdir())


def _vector_index_persist_exists() -> bool:
    if VECTOR_DB_TYPE == "chroma":
        return _chroma_persist_exists()
    if VECTOR_DB_TYPE == "faiss":
        return _faiss_persist_exists()
    return False


def _ensure_vector_store_for_query() -> bool:
    """
    Prépare la base vectorielle pour /api/query :
    - réutilise l'index Chroma persisté seulement si les PDFs n'ont pas changé depuis ;
    - sinon réindexe (sinon les nouveaux fichiers ne sont jamais dans les vecteurs).
    """
    current_fp = compute_raw_pdfs_fingerprint()
    stored_fp = read_stored_fingerprint()

    if pipelines.get("vector_store") is not None:
        if stored_fp == current_fp:
            return True
        logger.info("PDFs modifiés — réinitialisation de l’index en mémoire.")
        pipelines.pop("vector_store", None)
        pipelines.pop("chunks", None)

    from src.pdf_processor import PDFProcessor
    from src.vector_store import VectorStore

    index_matches_sources = stored_fp is not None and stored_fp == current_fp

    if _vector_index_persist_exists() and index_matches_sources:
        try:
            vs = VectorStore()
            vs.load()
            pipelines["vector_store"] = vs
            proc = PDFProcessor()
            pipelines["chunks"] = proc.process_pipeline()
            logger.info("Base vectorielle rechargée depuis le disque (%s).", VECTORSTORE_DIR)
            return True
        except Exception as e:
            logger.warning("Chargement de l'index persisté impossible : %s", e)

    if _vector_index_persist_exists() and not index_matches_sources:
        logger.info(
            "PDFs modifiés ou nouveaux par rapport à l'index — réindexation (empreinte différente)."
        )
        clear_persisted_vectorstore()

    processor = PDFProcessor()
    chunks = processor.process_pipeline()
    if not chunks:
        return False

    try:
        vs = VectorStore()
        vs.create_from_documents(chunks)
        pipelines["vector_store"] = vs
        pipelines["chunks"] = chunks
        write_stored_fingerprint(current_fp)
        logger.info("Indexation automatique : %s chunks.", len(chunks))
        return True
    except Exception as e:
        logger.error("Indexation automatique échouée : %s", e)
        return False


def create_app() -> Flask:
    """Factory Flask."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.update(FLASK_CONFIG)

    # Dossier uploads
    UPLOAD_FOLDER = str(RAW_PDF_DIR)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max

    # ── Routes ──────────────────────────────────────────────
    @app.route("/")
    def index():
        """Page principale."""
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        """Santé de l'API."""
        pdfs_n = len(list(RAW_PDF_DIR.glob("*.pdf")))
        return jsonify({
            "status": "ok",
            "pipelines_loaded": list(pipelines.keys()),
            "pdfs_available": pdfs_n,
            "index_persisted": _vector_index_persist_exists(),
            "vector_store_in_memory": pipelines.get("vector_store") is not None,
        })

    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        """Paramètres effectifs (secrets masqués)."""
        from src.env_persist import get_public_settings

        return jsonify(get_public_settings())

    @app.route("/api/settings", methods=["POST"])
    def post_settings():
        """Enregistre dans .env et applique à chaud (sauf secrets vides = inchangés)."""
        from src.env_persist import apply_settings

        body = request.get_json(silent=True) or {}
        payload, err = apply_settings(body)
        if err:
            return jsonify({"error": err}), 400
        if payload.get("invalidate_vector_cache"):
            pipelines.pop("vector_store", None)
            pipelines.pop("chunks", None)
            logger.info("Cache vectoriel vidé après changement de chunk size.")
        return jsonify(payload)

    @app.route("/api/upload", methods=["POST"])
    def upload_pdf():
        """Upload de fichiers PDF."""
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier fourni"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Nom de fichier vide"}), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Seuls les PDFs sont acceptés"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        return jsonify({
            "message": f"Fichier '{filename}' uploadé avec succès",
            "filename": filename,
            "path": filepath,
        })

    @app.route("/api/pdfs")
    def list_pdfs():
        """Liste les PDFs disponibles."""
        pdfs = [f.name for f in RAW_PDF_DIR.glob("*.pdf")]
        return jsonify({"pdfs": pdfs, "count": len(pdfs)})

    @app.route("/api/index", methods=["POST"])
    def index_pdfs():
        """Indexe les PDFs dans la base vectorielle."""
        try:
            from src.pdf_processor import PDFProcessor
            from src.vector_store import VectorStore

            processor = PDFProcessor()
            chunks = processor.process_pipeline()

            if not chunks:
                return jsonify({"error": "Aucun PDF à indexer"}), 400

            clear_persisted_vectorstore()
            vs = VectorStore()
            vs.create_from_documents(chunks)

            stats = processor.get_stats(chunks)
            pipelines["vector_store"] = vs
            pipelines["chunks"] = chunks
            write_stored_fingerprint(compute_raw_pdfs_fingerprint())

            return jsonify({
                "message": "Indexation terminée",
                "stats": stats,
            })
        except Exception as e:
            logger.exception("POST /api/index")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/query", methods=["POST"])
    def query():
        """Interroge le système RAG/Agent."""
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Champ 'question' requis"}), 400

        question = data["question"]
        pipeline_type = data.get("pipeline", "rag_simple")

        try:
            result = _run_pipeline(question, pipeline_type)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Erreur query : {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/dataset")
    def get_dataset():
        """Retourne le dataset d'évaluation (fichier JSON s'il existe)."""
        try:
            path = EVALUATION_DIR / "dataset_evaluation.json"
            if not path.exists():
                return jsonify({
                    "metadata": {
                        "version": "1.1",
                        "total_questions": 0,
                        "hint": (
                            "Aucun dataset enregistré. Indexez les PDF puis utilisez "
                            "« Générer depuis les PDFs » (POST /api/dataset/generate)."
                        ),
                    },
                    "data": [],
                })

            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/dataset/generate", methods=["POST"])
    def generate_dataset_from_pdfs():
        """Génère le dataset à partir des chunks des PDFs (LLM), puis sauvegarde."""
        from src.config import EVALUATION_CONFIG
        from src.dataset_generator import DatasetGenerator
        from src.llm_factory import get_default_chat_llm
        from src.pdf_processor import PDFProcessor

        payload = request.get_json(silent=True) or {}
        try:
            max_pairs = payload.get("max_pairs")
            if max_pairs is not None:
                max_pairs = int(max_pairs)
        except (TypeError, ValueError):
            return jsonify({"error": "max_pairs doit être un entier"}), 400

        if max_pairs is None:
            max_pairs = EVALUATION_CONFIG.get("dataset_pdf_max_pairs", 60)
        max_pairs = max(1, min(max_pairs, EVALUATION_CONFIG["num_questions"]))

        if not _ensure_vector_store_for_query():
            return jsonify({
                "error": (
                    "Aucun PDF indexable. Ajoutez des PDF dans data/raw_pdfs/, "
                    "indexez (bouton Indexer), puis réessayez."
                ),
            }), 400

        chunks = pipelines.get("chunks")
        if not chunks:
            processor = PDFProcessor()
            chunks = processor.process_pipeline()
        if not chunks:
            return jsonify({"error": "Aucun chunk PDF disponible."}), 400

        try:
            llm = get_default_chat_llm()
            gen = DatasetGenerator()
            dataset = gen.generate_from_pdf_chunks(chunks, llm, max_pairs=max_pairs)
            if not dataset:
                return jsonify({
                    "error": "Aucune paire Q/R générée (vérifiez le LLM et le contenu des PDFs).",
                }), 502
            gen.save_dataset(
                dataset,
                metadata_extra={"generation": "pdf_chunks", "max_pairs_requested": max_pairs},
            )
            return jsonify({
                "message": "Dataset généré depuis les PDFs",
                "count": len(dataset),
                "metadata": {
                    "generation": "pdf_chunks",
                    "pdf_sources": sorted({d.get("source_file") for d in dataset if d.get("source_file")}),
                },
            })
        except Exception as e:
            logger.exception("generate_dataset_from_pdfs")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/benchmark")
    def get_benchmark():
        """Retourne les résultats du benchmark."""
        path = EVALUATION_DIR / "benchmark_results.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({"message": "Pas encore de résultats. Lancez le benchmark depuis l’interface ou : python run.py --step benchmark"})

    @app.route("/api/benchmark/run", methods=["POST"])
    def run_benchmark_api():
        """
        Exécute le benchmark sur un échantillon du dataset (évite les timeouts).
        Sauvegarde dans data/evaluation/benchmark_results.json.
        """
        payload = request.get_json(silent=True) or {}
        try:
            max_questions = int(payload.get("max_questions", 15))
        except (TypeError, ValueError):
            return jsonify({"error": "max_questions doit être un entier"}), 400
        max_questions = max(1, min(max_questions, 100))

        ds_path = EVALUATION_DIR / "dataset_evaluation.json"
        if not ds_path.is_file():
            return jsonify({
                "error": "Aucun dataset. Générez-le dans l’onglet Dataset (depuis les PDFs) ou : python run.py --step dataset",
            }), 400

        with open(ds_path, encoding="utf-8") as f:
            bundle = json.load(f)
        dataset = bundle.get("data", [])
        if not dataset:
            return jsonify({"error": "Le fichier dataset est vide."}), 400

        sample = dataset[:max_questions]

        try:
            from src.env_persist import apply_benchmark_winner_to_chat
            from src.llm_benchmark import LLMBenchmark

            benchmark = LLMBenchmark()
            benchmark.load_models()
            output = benchmark.run_benchmark(sample)
            output["chat_alignment"] = apply_benchmark_winner_to_chat(
                output.get("best_model") or {}
            )
            benchmark.save_results(output)
            return jsonify(output)
        except Exception as e:
            logger.exception("run_benchmark_api")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/pipelines")
    def list_pipelines():
        """Liste les pipelines disponibles."""
        return jsonify({
            "pipelines": [
                {"id": "rag_simple", "name": "RAG Simple", "description": "LLM + Base vectorielle"},
                {"id": "rag_advanced", "name": "RAG Avancé", "description": "RAG + Reranking + HyDE + Hybride"},
                {"id": "rag_agent", "name": "RAG + Agent", "description": "Agent ReAct avec outils"},
                {"id": "multi_agent", "name": "Multi-Agent", "description": "RAFT + Système multi-agents"},
            ]
        })

    return app


def _run_pipeline(question: str, pipeline_type: str) -> dict:
    """Exécute la pipeline demandée."""
    if not _ensure_vector_store_for_query():
        return {
            "error": (
                "Aucun document indexable : ajoutez des fichiers PDF (dossier data/raw_pdfs/ "
                "ou onglet Documents), puis lancez l’indexation (bouton « Indexer » ou POST /api/index)."
            ),
        }

    vs = pipelines.get("vector_store")

    llm = _get_llm()

    if pipeline_type == "rag_simple":
        from src.rag_simple import SimpleRAG
        rag = SimpleRAG(llm=llm, vector_store=vs)
        return rag.query(question)

    elif pipeline_type == "rag_advanced":
        from src.rag_advanced import AdvancedRAG
        chunks = pipelines.get("chunks", [])
        rag = AdvancedRAG(llm=llm, vector_store=vs, all_documents=chunks)
        return rag.query(question)

    elif pipeline_type == "rag_agent":
        from src.rag_agent import RAGAgent
        agent = RAGAgent(llm=llm, vector_store=vs)
        return agent.query(question)

    elif pipeline_type == "multi_agent":
        from src.raft_multi_agent import MultiAgentSupervisor
        supervisor = MultiAgentSupervisor(llm=llm, vector_store=vs)
        return supervisor.process(question)

    else:
        return {"error": f"Pipeline inconnue : {pipeline_type}"}


def _get_llm():
    """Retourne le LLM configuré (voir LLM_PROVIDER dans .env)."""
    from src.llm_factory import get_default_chat_llm

    return get_default_chat_llm()


# ── Point d'entrée ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(
        host=FLASK_CONFIG["HOST"],
        port=FLASK_CONFIG["PORT"],
        debug=FLASK_CONFIG["DEBUG"],
    )
