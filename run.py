"""
Pipeline principal — Exécute toutes les étapes du projet séquentiellement.

Usage :
    python run.py --step all          # Toutes les étapes
    python run.py --step dataset      # Étape 1 uniquement
    python run.py --step benchmark    # Étape 2 uniquement
    python run.py --step rag          # Étape 3 : RAG Simple
    python run.py --step rag-adv      # Étape 4 : RAG Avancé
    python run.py --step finetune     # Étape 5 : Fine-Tuning
    python run.py --step raft         # Étape 6 : RAFT
    python run.py --step agent        # Étape 7 : RAG + Agent
    python run.py --step multi-agent  # Étape 8 : RAFT + Multi-Agent
    python run.py --step evaluate     # Évaluation comparative
    python run.py --step serve        # Lancer l'interface Flask
"""
import argparse
import logging
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import LOG_LEVEL, LOG_FORMAT

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("pipeline")


def step_1_dataset():
    """Étape 1 : Générer le dataset d'évaluation (PDFs + LLM si possible, sinon templates)."""
    logger.info("━" * 60)
    logger.info("ÉTAPE 1 — Génération du dataset d'évaluation")
    logger.info("━" * 60)

    from src.config import EVALUATION_CONFIG
    from src.dataset_generator import DatasetGenerator
    from src.pdf_processor import PDFProcessor
    from src.llm_factory import get_default_chat_llm

    gen = DatasetGenerator()
    processor = PDFProcessor()
    chunks = processor.process_pipeline()
    path = None

    if chunks:
        logger.info("Génération alignée sur les PDFs (%s chunks)...", len(chunks))
        llm = get_default_chat_llm()
        n = EVALUATION_CONFIG["num_questions"]
        dataset = gen.generate_from_pdf_chunks(chunks, llm, max_pairs=n)
        if not dataset:
            logger.warning("Échec génération depuis PDFs — repli sur les templates.")
            dataset = gen.generate_local_dataset()
            path = gen.save_dataset(dataset, metadata_extra={"generation": "template"})
        else:
            path = gen.save_dataset(dataset, metadata_extra={"generation": "pdf_chunks"})
    else:
        logger.warning("Aucun PDF — dataset par templates (e-learning).")
        dataset = gen.generate_local_dataset()
        path = gen.save_dataset(dataset, metadata_extra={"generation": "template"})

    logger.info("✅ Dataset : %s questions → %s", len(dataset), path)
    return dataset


def step_2_benchmark():
    """Étape 2 : Benchmark des 3 LLMs."""
    logger.info("━" * 60)
    logger.info("ÉTAPE 2 — Benchmark LLM")
    logger.info("━" * 60)

    from src.llm_benchmark import LLMBenchmark
    from src.dataset_generator import DatasetGenerator

    gen = DatasetGenerator()
    try:
        dataset = gen.load_dataset()
    except FileNotFoundError:
        dataset = gen.generate_local_dataset()
        gen.save_dataset(dataset)

    from src.env_persist import apply_benchmark_winner_to_chat

    benchmark = LLMBenchmark()
    benchmark.load_models()
    results = benchmark.run_benchmark(dataset[:30])  # Échantillon pour test rapide
    results["chat_alignment"] = apply_benchmark_winner_to_chat(
        results.get("best_model") or {}
    )
    benchmark.save_results(results)
    logger.info(f"✅ Meilleur modèle : {results['best_model']['name']}")
    return results


def step_3_rag_simple(llm=None):
    """Étape 3 : RAG Simple."""
    logger.info("━" * 60)
    logger.info("ÉTAPE 3 — RAG Simple")
    logger.info("━" * 60)

    from src.pdf_processor import PDFProcessor
    from src.rag_simple import create_simple_rag

    processor = PDFProcessor()
    chunks = processor.process_pipeline()

    if not chunks:
        logger.warning("⚠ Aucun PDF trouvé. Placez vos PDFs dans data/raw_pdfs/")
        return None

    if llm is None:
        llm = _get_default_llm()

    rag = create_simple_rag(llm, chunks)
    logger.info("✅ RAG Simple prêt")
    return rag


def step_4_rag_advanced(llm=None):
    """Étape 4 : RAG Avancé."""
    logger.info("━" * 60)
    logger.info("ÉTAPE 4 — RAG Avancé")
    logger.info("━" * 60)

    from src.pdf_processor import PDFProcessor
    from src.vector_store import VectorStore
    from src.rag_advanced import AdvancedRAG

    processor = PDFProcessor()
    chunks = processor.process_pipeline()

    if not chunks:
        logger.warning("⚠ Aucun PDF trouvé.")
        return None

    if llm is None:
        llm = _get_default_llm()

    vs = VectorStore()
    vs.create_from_documents(chunks)

    rag_adv = AdvancedRAG(
        llm=llm,
        vector_store=vs,
        all_documents=chunks,
        use_reranker=True,
        use_hyde=True,
        use_hybrid=True,
        use_multi_query=True,
    )
    logger.info("✅ RAG Avancé prêt")
    return rag_adv


def step_5_finetune():
    """Étape 5 : Fine-Tuning."""
    logger.info("━" * 60)
    logger.info("ÉTAPE 5 — Fine-Tuning (LoRA / QLoRA)")
    logger.info("━" * 60)

    from src.fine_tuning import FineTuner, DatasetFormatter
    from src.dataset_generator import DatasetGenerator
    from src.config import LLM_MODELS

    gen = DatasetGenerator()
    try:
        dataset = gen.load_dataset()
    except FileNotFoundError:
        dataset = gen.generate_local_dataset()
        gen.save_dataset(dataset)

    # Formater le dataset
    formatter = DatasetFormatter()
    formatted = formatter.to_chat_format(dataset)

    # Sélectionner le modèle open-source
    base_model = LLM_MODELS["qwen-7b"]["model_id"]
    tuner = FineTuner(base_model)

    logger.info("Fine-tuning configuré. Lancez avec GPU pour exécuter.")
    logger.info(f"  Modèle : {base_model}")
    logger.info(f"  Dataset : {len(formatted)} exemples")
    return tuner


def step_6_raft():
    """Étape 6 : RAFT."""
    logger.info("━" * 60)
    logger.info("ÉTAPE 6 — RAFT (Retrieval Augmented Fine-Tuning)")
    logger.info("━" * 60)

    from src.raft import RAFTDatasetBuilder
    from src.pdf_processor import PDFProcessor
    from src.dataset_generator import DatasetGenerator

    gen = DatasetGenerator()
    try:
        dataset = gen.load_dataset()
    except FileNotFoundError:
        dataset = gen.generate_local_dataset()
        gen.save_dataset(dataset)

    processor = PDFProcessor()
    chunks = processor.process_pipeline()

    if chunks:
        builder = RAFTDatasetBuilder(chunks)
        raft_examples = builder.build_raft_examples(dataset)
        path = builder.save_dataset(raft_examples)
        logger.info(f"✅ Dataset RAFT créé : {len(raft_examples)} exemples → {path}")
    else:
        logger.warning("⚠ Pas de chunks. Le dataset RAFT sera basé sur les templates.")

    return None


def step_7_agent(llm=None):
    """Étape 7 : RAG + Agent IA."""
    logger.info("━" * 60)
    logger.info("ÉTAPE 7 — RAG + Agent IA")
    logger.info("━" * 60)

    from src.pdf_processor import PDFProcessor
    from src.vector_store import VectorStore
    from src.rag_agent import RAGAgent

    processor = PDFProcessor()
    chunks = processor.process_pipeline()

    if not chunks:
        logger.warning("⚠ Aucun PDF trouvé.")
        return None

    if llm is None:
        llm = _get_default_llm()

    vs = VectorStore()
    vs.create_from_documents(chunks)

    agent = RAGAgent(llm=llm, vector_store=vs)
    logger.info("✅ Agent RAG prêt")
    return agent


def step_8_multi_agent(llm=None):
    """Étape 8 : RAFT + Multi-Agent IA."""
    logger.info("━" * 60)
    logger.info("ÉTAPE 8 — RAFT + Multi-Agent IA")
    logger.info("━" * 60)

    from src.pdf_processor import PDFProcessor
    from src.vector_store import VectorStore
    from src.raft_multi_agent import create_multi_agent_system

    processor = PDFProcessor()
    chunks = processor.process_pipeline()

    if not chunks:
        logger.warning("⚠ Aucun PDF trouvé.")
        return None

    if llm is None:
        llm = _get_default_llm()

    vs = VectorStore()
    vs.create_from_documents(chunks)

    system = create_multi_agent_system(llm, vs)
    logger.info("✅ Système Multi-Agent prêt")
    return system


def step_serve():
    """Lance l'interface Flask."""
    logger.info("━" * 60)
    logger.info("LANCEMENT DE L'INTERFACE WEB")
    logger.info("━" * 60)

    from app.app import create_app
    from src.config import FLASK_CONFIG

    app = create_app()
    app.run(
        host=FLASK_CONFIG["HOST"],
        port=FLASK_CONFIG["PORT"],
        debug=FLASK_CONFIG["DEBUG"],
    )


def _get_default_llm():
    """Retourne un LLM par défaut (OpenAI ou HuggingFace)."""
    from src.llm_factory import get_default_chat_llm

    return get_default_chat_llm()


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
STEPS = {
    "dataset": step_1_dataset,
    "benchmark": step_2_benchmark,
    "rag": step_3_rag_simple,
    "rag-adv": step_4_rag_advanced,
    "finetune": step_5_finetune,
    "raft": step_6_raft,
    "agent": step_7_agent,
    "multi-agent": step_8_multi_agent,
    "serve": step_serve,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG-LLM Multi-Agent")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()) + ["all"],
        default="dataset",
        help="Étape à exécuter",
    )
    args = parser.parse_args()

    if args.step == "all":
        for name, func in STEPS.items():
            if name != "serve":
                func()
    else:
        STEPS[args.step]()
