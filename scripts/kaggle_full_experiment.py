#!/usr/bin/env python3
"""
Orchestration expériences Kaggle (étapes 0–8) avec sauvegarde systématique dans data/evaluation/.

Usage (depuis la racine du projet) :
    export HUGGINGFACE_TOKEN=...
    export KAGGLE_MAX_QUESTIONS_PER_EVAL=15   # limite par évaluation RAG / agent
    export FINETUNE_RUN=1                     # optionnel : étape 5 entraînement QLoRA
    export KAGGLE_RUN_RAFT_TRAIN=0            # 1 pour lancer aussi l’entraînement RAFT (long)

    python scripts/kaggle_full_experiment.py

Variables utiles :
    KAGGLE_SKIP_STEPS=0,1,...   # indices à sauter (voir _STEP_NAMES)
    KAGGLE_DATASET_REGEN=0      # 1 pour régénérer dataset_evaluation.json (LLM)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Racine projet
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("kaggle_experiment")

EVAL_DIR = ROOT / "data" / "evaluation"


def _inject_kaggle_secrets() -> None:
    """
    Sur Kaggle, `!python script.py` ne voit pas les variables posées dans une autre cellule Python.
    On lit alors les Add-ons → Secrets (même noms que dans l’UI Kaggle).
    """
    has_hf = bool((os.getenv("HUGGINGFACE_TOKEN") or "").strip())
    has_oai = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    if has_hf or has_oai:
        return
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for secret_name, env_name in (
            ("HUGGINGFACE_TOKEN", "HUGGINGFACE_TOKEN"),
            ("HF_TOKEN", "HUGGINGFACE_TOKEN"),
            ("OPENAI_API_KEY", "OPENAI_API_KEY"),
        ):
            try:
                val = (client.get_secret(secret_name) or "").strip()
            except Exception:
                val = ""
            if val and not (os.getenv(env_name) or "").strip():
                os.environ[env_name] = val
                logger.info("Secret Kaggle injecté : %s → %s", secret_name, env_name)
    except Exception as exc:
        logger.debug("kaggle_secrets indisponible (%s)", exc)

    if (os.getenv("HUGGINGFACE_TOKEN") or "").strip() and not (os.getenv("OPENAI_API_KEY") or "").strip():
        os.environ.setdefault("LLM_PROVIDER", "huggingface")

    if not (os.getenv("HUGGINGFACE_TOKEN") or "").strip() and not (os.getenv("OPENAI_API_KEY") or "").strip():
        logger.warning(
            "Aucune clé LLM dans l’environnement. Ajoutez le secret HUGGINGFACE_TOKEN dans Kaggle "
            "(Add-ons → Secrets) ou exportez-la avant : os.environ['HUGGINGFACE_TOKEN']=..."
        )


_STEP_NAMES = [
    "00_knowledge_base",
    "01_qa_dataset",
    "02_llm_only",
    "03_rag_simple",
    "04_rag_advanced",
    "05_finetune",
    "06_raft",
    "07_rag_agent",
    "08_rag_multi_agent",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_json(name: str, payload: dict) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / name
    payload = {"generated_at": _utc_now(), **payload}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Écrit %s", path)
    return path


def _parse_skip() -> set[int]:
    raw = os.getenv("KAGGLE_SKIP_STEPS", "").strip()
    if not raw:
        return set()
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            logger.warning("KAGGLE_SKIP_STEPS : ignoré '%s'", part)
    return out


def _max_questions() -> int:
    v = os.getenv("KAGGLE_MAX_QUESTIONS_PER_EVAL", "20").strip()
    try:
        n = max(1, int(v))
    except ValueError:
        n = 20
    return n


class _MultiAgentAsPipeline:
    """Adapte MultiAgentSupervisor.process → .query() pour PipelineEvaluator."""

    def __init__(self, supervisor):
        self._sup = supervisor

    def query(self, question: str):
        return self._sup.process(question)


def step_00_knowledge_base() -> dict:
    from src.config import RAW_PDF_DIR
    from src.pdf_processor import PDFProcessor

    pdfs = sorted(RAW_PDF_DIR.glob("**/*.pdf"))
    processor = PDFProcessor()
    chunks = processor.process_pipeline()
    stats = processor.get_stats(chunks) if chunks else {"total_chunks": 0}

    return {
        "step": "0",
        "title": "Base de connaissances (corpus RAG)",
        "pdf_dir": str(RAW_PDF_DIR),
        "pdf_count": len(pdfs),
        "pdf_files": [p.name for p in pdfs[:200]],
        "chunk_stats": stats,
    }


def step_01_qa_dataset(regenerate: bool) -> dict:
    from src.dataset_generator import DatasetGenerator

    gen = DatasetGenerator()
    path = EVAL_DIR / "dataset_evaluation.json"

    if regenerate or not path.is_file():
        logger.info("Génération dataset Q/R (LLM + PDFs si chunks disponibles)…")
        from run import step_1_dataset

        step_1_dataset()
    else:
        logger.info("Réutilisation de dataset_evaluation.json existant.")

    dataset = gen.load_dataset()
    return {
        "step": "1",
        "title": "Dataset questions-réponses",
        "dataset_path": str(path),
        "num_pairs": len(dataset),
    }


def step_02_llm_only(dataset_slice: list) -> dict:
    from src.llm_benchmark import LLMBenchmark
    from src.env_persist import apply_benchmark_winner_to_chat

    bench = LLMBenchmark()
    bench.load_models()
    out = bench.run_benchmark(dataset_slice)
    out["chat_alignment"] = apply_benchmark_winner_to_chat(out.get("best_model") or {})
    bench.save_results(out)
    _save_json("kaggle_step02_llm_only.json", {"step": "2", "title": "LLM seuls (benchmark)", "benchmark": out})
    return {"benchmark_path": str(EVAL_DIR / "benchmark_results.json"), "best_model": out.get("best_model")}


def _build_vector_and_llm():
    from src.llm_factory import get_default_chat_llm
    from src.pdf_processor import PDFProcessor
    from src.vector_store import VectorStore

    processor = PDFProcessor()
    chunks = processor.process_pipeline()
    if not chunks:
        raise RuntimeError("Aucun chunk : placez des PDF dans data/raw_pdfs/")
    vs = VectorStore()
    vs.create_from_documents(chunks)
    llm = get_default_chat_llm()
    return llm, vs, chunks


def steps_03_04_07_08_rag_family(max_q: int, skip: set[int]) -> dict:
    from src.dataset_generator import DatasetGenerator
    from src.evaluator import PipelineEvaluator
    from src.rag_simple import SimpleRAG
    from src.rag_advanced import AdvancedRAG
    from src.rag_agent import RAGAgent
    from src.raft_multi_agent import MultiAgentSupervisor

    gen = DatasetGenerator()
    dataset = gen.load_dataset()
    ds = dataset[:max_q]

    llm, vs, chunks = _build_vector_and_llm()
    ev = PipelineEvaluator()

    files = {}

    if 3 not in skip:
        logger.info("Évaluation RAG simple…")
        ev.evaluate_pipeline(SimpleRAG(llm=llm, vector_store=vs), "rag_simple", ds)
        files["03"] = _save_json(
            "kaggle_step03_rag_simple.json",
            {"step": "3", "title": "RAG simple", "summary": ev.results["rag_simple"]["summary"]},
        )

    if 4 not in skip:
        logger.info("Évaluation RAG avancé…")
        ev.evaluate_pipeline(
            AdvancedRAG(llm=llm, vector_store=vs, all_documents=chunks),
            "rag_advanced",
            ds,
        )
        files["04"] = _save_json(
            "kaggle_step04_rag_advanced.json",
            {"step": "4", "title": "RAG optimisé", "summary": ev.results["rag_advanced"]["summary"]},
        )

    if 7 not in skip:
        logger.info("Évaluation RAG + agent…")
        ev.evaluate_pipeline(RAGAgent(llm=llm, vector_store=vs), "rag_agent", ds)
        files["07"] = _save_json(
            "kaggle_step07_rag_agent.json",
            {"step": "7", "title": "RAG + agent IA", "summary": ev.results["rag_agent"]["summary"]},
        )

    if 8 not in skip:
        logger.info("Évaluation multi-agent…")
        ev.evaluate_pipeline(
            _MultiAgentAsPipeline(MultiAgentSupervisor(llm=llm, vector_store=vs)),
            "rag_multi_agent",
            ds,
        )
        files["08"] = _save_json(
            "kaggle_step08_rag_multi_agent.json",
            {"step": "8", "title": "RAG + multi-agent", "summary": ev.results["rag_multi_agent"]["summary"]},
        )

    report_path = EVAL_DIR / "kaggle_rag_family_report.json"
    report = {
        "generated_at": _utc_now(),
        "max_questions": max_q,
        "comparison": ev.compare_all(),
        "summaries": {k: v["summary"] for k, v in ev.results.items()},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Rapport comparatif RAG → %s", report_path)
    files["compare"] = str(report_path)
    return files


def step_05_finetune(dataset: list) -> dict:
    from src.fine_tuning import DatasetFormatter, run_finetune_from_evaluation_dataset

    fmt_path = EVAL_DIR / "kaggle_step05a_finetune_chat_format.jsonl"
    DatasetFormatter.save_formatted(DatasetFormatter.to_chat_format(dataset), fmt_path)

    out = {
        "step": "5",
        "title": "Fine-tuning",
        "formatted_dataset_jsonl": str(fmt_path),
        "num_examples": len(dataset),
    }

    run_train = os.getenv("FINETUNE_RUN", "").strip().lower() in ("1", "true", "yes")
    if run_train:
        method = os.getenv("FINETUNE_METHOD", "qlora").strip().lower()
        max_s = os.getenv("FINETUNE_MAX_SAMPLES", "").strip()
        max_samples = int(max_s) if max_s.isdigit() else None
        logger.info("Fine-tuning (%s)…", method)
        result = run_finetune_from_evaluation_dataset(
            dataset_rows=dataset,
            base_model_id=os.getenv("FINETUNE_BASE_MODEL", "").strip()
            or __import__("src.config", fromlist=["LLM_MODELS"]).LLM_MODELS["qwen-7b"]["model_id"],
            method=method,
            max_samples=max_samples,
        )
        out["training"] = {
            "method": result.method,
            "base_model": result.base_model,
            "final_loss": result.final_loss,
            "eval_loss": result.eval_loss,
            "training_time_min": result.training_time_min,
            "model_path": result.model_path,
        }
        out["finetune_manifest"] = str(EVAL_DIR / "finetune_manifest.json")
    else:
        out["note"] = (
            "Entraînement non lancé (FINETUNE_RUN absent). "
            "Sur Kaggle : FINETUNE_RUN=1 python scripts/kaggle_full_experiment.py (ou run.py --step finetune)."
        )

    _save_json("kaggle_step05_finetune.json", out)
    return out


def step_06_raft(dataset: list) -> dict:
    from src.raft import RAFTDatasetBuilder
    from src.pdf_processor import PDFProcessor
    from src.config import LLM_MODELS

    processor = PDFProcessor()
    chunks = processor.process_pipeline()
    out = {
        "step": "6",
        "title": "RAFT (dataset + option entraînement)",
        "num_chunks": len(chunks),
        "num_qa_pairs": len(dataset),
    }

    if not chunks:
        out["error"] = "Pas de chunks — impossible de construire le dataset RAFT."
        _save_json("kaggle_step06_raft.json", out)
        return out

    builder = RAFTDatasetBuilder(chunks)
    raft_examples = builder.build_raft_examples(dataset)
    raft_path = builder.save_dataset(raft_examples, filename="raft_dataset.jsonl")
    out["raft_dataset_path"] = str(raft_path)
    out["raft_num_examples"] = len(raft_examples)

    if os.getenv("KAGGLE_RUN_RAFT_TRAIN", "").strip() in ("1", "true", "yes"):
        from src.raft import RAFTTrainer

        base = os.getenv("FINETUNE_BASE_MODEL", "").strip() or LLM_MODELS["qwen-7b"]["model_id"]
        trainer = RAFTTrainer(base)
        r = trainer.train(raft_path, method=os.getenv("RAFT_TRAIN_METHOD", "qlora"))
        out["raft_training"] = {
            "final_loss": r.final_loss,
            "model_path": r.model_path,
        }
    else:
        out["note"] = (
            "Dataset RAFT écrit. Évaluation « RAG + modèle RAFT fine-tuné » : charger l’adaptateur + base "
            "et inférer avec contexte RAG (hors routeur API). KAGGLE_RUN_RAFT_TRAIN=1 pour entraîner ici."
        )

    _save_json("kaggle_step06_raft.json", out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Pipeline expérimental Kaggle (0–8)")
    parser.add_argument(
        "--regen-dataset",
        action="store_true",
        help="Forcer la régénération du dataset Q/R (étape 1)",
    )
    args = parser.parse_args()

    _inject_kaggle_secrets()

    skip = _parse_skip()
    max_q = _max_questions()
    index = {
        "started_at": _utc_now(),
        "max_questions_per_rag_eval": max_q,
        "skip_step_indices": sorted(skip),
        "output_files": [],
    }

    try:
        if 0 not in skip:
            p = _save_json("kaggle_step00_knowledge_base.json", step_00_knowledge_base())
            index["output_files"].append(str(p))

        regen = args.regen_dataset or os.getenv("KAGGLE_DATASET_REGEN", "").strip() in ("1", "true", "yes")
        if 1 not in skip:
            p = _save_json("kaggle_step01_qa_dataset.json", step_01_qa_dataset(regen))
            index["output_files"].append(str(p))

        from src.dataset_generator import DatasetGenerator

        gen = DatasetGenerator()
        try:
            full_dataset = gen.load_dataset()
        except FileNotFoundError as e:
            raise RuntimeError(
                "dataset_evaluation.json introuvable. Lancez l’étape 1 (ne pas mettre 1 dans KAGGLE_SKIP_STEPS) "
                "ou créez data/evaluation/dataset_evaluation.json."
            ) from e
        dataset_eval_slice = full_dataset[:max_q]

        if 2 not in skip:
            step_02_llm_only(dataset_eval_slice)
            index["output_files"].append(str(EVAL_DIR / "kaggle_step02_llm_only.json"))
            index["output_files"].append(str(EVAL_DIR / "benchmark_results.json"))

        if any(s not in skip for s in (3, 4, 7, 8)):
            rag_files = steps_03_04_07_08_rag_family(max_q, skip)
            index["rag_outputs"] = rag_files
            for v in rag_files.values():
                if v not in index["output_files"]:
                    index["output_files"].append(v)

        if 5 not in skip:
            p = EVAL_DIR / "kaggle_step05_finetune.json"
            step_05_finetune(full_dataset)
            index["output_files"].append(str(p))
            if (EVAL_DIR / "finetune_manifest.json").is_file():
                index["output_files"].append(str(EVAL_DIR / "finetune_manifest.json"))

        if 6 not in skip:
            p = _save_json("kaggle_step06_raft.json", step_06_raft(full_dataset))
            index["output_files"].append(str(p))
            rp = EVAL_DIR / "raft_dataset.jsonl"
            if rp.is_file():
                index["output_files"].append(str(rp))

    finally:
        index["finished_at"] = _utc_now()
        manifest_path = EVAL_DIR / "kaggle_experiment_index.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        logger.info("Index global → %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
