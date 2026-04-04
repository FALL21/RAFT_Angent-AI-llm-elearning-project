"""
ÉTAPE 2 — Benchmark de 3 LLMs pour sélectionner le meilleur.
Trois modèles : Qwen 2.5 7B, Mistral 7B, Llama 3.1 8B (IDs dans `LLM_MODELS`).

Métriques : accuracy, F1, BLEU, ROUGE-L, latence.
"""
import json
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

import src.config as project_cfg
from src.config import LLM_MODELS, EVALUATION_DIR

logger = logging.getLogger(__name__)


def _is_quota_exhausted(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "402" in msg
        or "payment required" in msg
        or "depleted your monthly" in msg
        or "pre-paid credits" in msg
    )


@dataclass
class BenchmarkResult:
    model_name: str
    domain: str
    accuracy: float
    f1_score: float
    bleu_score: float
    rouge_l: float
    avg_latency_ms: float
    total_questions: int
    correct_answers: int


class LLMProvider:
    """Interface unifiée pour les différents fournisseurs de LLM."""

    def __init__(self, model_config: dict):
        self.config = model_config
        self.provider = model_config["provider"]
        self.model_id = model_config["model_id"]
        self._client = None

    def initialize(self):
        """Initialise le client du fournisseur."""
        if self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=project_cfg.OPENAI_API_KEY)

        elif self.provider == "huggingface":
            from openai import OpenAI

            token = (project_cfg.HUGGINGFACE_TOKEN or "").strip()
            if not token:
                raise ValueError("HUGGINGFACE_TOKEN requis pour le benchmark des modèles HF.")
            # Même routeur que llm_factory (plus d’appel /models/… sur api-inference)
            self._client = OpenAI(
                base_url=project_cfg.HF_ROUTER_BASE_URL,
                api_key=token,
            )

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Génère une réponse à partir du prompt."""
        if self._client is None:
            self.initialize()

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "huggingface":
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""

        return ""


class LLMBenchmark:
    """Benchmark comparatif de 3 LLMs sur le dataset d'évaluation."""

    def __init__(self):
        self.models: Dict[str, LLMProvider] = {}
        self.results: List[BenchmarkResult] = []

    def load_models(self, model_names: Optional[List[str]] = None):
        """Charge les modèles à évaluer (ignore ceux sans clé API)."""
        names = model_names or list(LLM_MODELS.keys())
        for name in names:
            if name not in LLM_MODELS:
                continue
            mconf = LLM_MODELS[name]
            if mconf["provider"] == "openai" and not (project_cfg.OPENAI_API_KEY or "").strip():
                logger.warning("Benchmark : ignore %s (OPENAI_API_KEY absente).", name)
                continue
            if mconf["provider"] == "huggingface" and not (project_cfg.HUGGINGFACE_TOKEN or "").strip():
                logger.warning("Benchmark : ignore %s (HUGGINGFACE_TOKEN absente).", name)
                continue
            self.models[name] = LLMProvider(mconf)
            logger.info("Modèle chargé : %s (%s)", name, mconf["description"])

    def compute_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calcule les métriques de qualité entre la prédiction et la référence."""
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(reference, prediction)
        rouge_l = rouge_scores["rougeL"].fmeasure

        # BLEU
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)

        # F1 basé sur les tokens
        ref_set = set(ref_tokens)
        pred_set = set(pred_tokens)
        if len(pred_set) == 0 or len(ref_set) == 0:
            f1 = 0.0
        else:
            precision = len(ref_set & pred_set) / len(pred_set)
            recall = len(ref_set & pred_set) / len(ref_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy simplifiée (chevauchement sémantique > seuil)
        accuracy = 1.0 if f1 > 0.3 else 0.0

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "bleu_score": bleu,
            "rouge_l": rouge_l,
        }

    def evaluate_model(self, model_name: str, dataset: List[Dict]) -> List[BenchmarkResult]:
        """Évalue un modèle sur le dataset complet."""
        logger.info(f"\n{'='*60}")
        logger.info(f"ÉVALUATION : {model_name}")
        logger.info(f"{'='*60}")

        provider = self.models[model_name]
        results_by_domain: Dict[str, Dict] = {}

        for item in dataset:
            domain = item["domain"]
            if domain not in results_by_domain:
                results_by_domain[domain] = {
                    "scores": [], "latencies": [], "correct": 0, "total": 0
                }

            prompt = f"""Réponds précisément à la question suivante du domaine "{domain}".
Question : {item['question']}
Réponse :"""

            try:
                start = time.time()
                prediction = provider.generate(prompt)
                latency = (time.time() - start) * 1000  # ms

                metrics = self.compute_metrics(prediction, item["answer"])
                results_by_domain[domain]["scores"].append(metrics)
                results_by_domain[domain]["latencies"].append(latency)
                results_by_domain[domain]["total"] += 1
                if metrics["accuracy"] > 0.5:
                    results_by_domain[domain]["correct"] += 1

            except Exception as e:
                results_by_domain[domain]["total"] += 1
                if _is_quota_exhausted(e):
                    logger.error(
                        "Crédits API épuisés (402) — fin de l’évaluation pour %s "
                        "(questions comptées pour ce modèle : %s).",
                        model_name,
                        sum(d["total"] for d in results_by_domain.values()),
                    )
                    break
                logger.error(f"Erreur avec {model_name} : {e}")

        # Agrégation par domaine
        domain_results = []
        for domain, data in results_by_domain.items():
            if not data["scores"]:
                continue
            n = len(data["scores"])
            result = BenchmarkResult(
                model_name=model_name,
                domain=domain,
                accuracy=sum(s["accuracy"] for s in data["scores"]) / n,
                f1_score=sum(s["f1_score"] for s in data["scores"]) / n,
                bleu_score=sum(s["bleu_score"] for s in data["scores"]) / n,
                rouge_l=sum(s["rouge_l"] for s in data["scores"]) / n,
                avg_latency_ms=sum(data["latencies"]) / n if data["latencies"] else 0,
                total_questions=data["total"],
                correct_answers=data["correct"],
            )
            domain_results.append(result)
            logger.info(f"  [{domain}] Acc={result.accuracy:.2%} F1={result.f1_score:.3f} "
                        f"BLEU={result.bleu_score:.3f} ROUGE-L={result.rouge_l:.3f} "
                        f"Latence={result.avg_latency_ms:.0f}ms")

        total_q = sum(d["total"] for d in results_by_domain.values())
        successful = sum(len(d["scores"]) for d in results_by_domain.values())
        failures = total_q - successful

        if not domain_results and total_q > 0:
            logger.warning(
                "%s : aucune réponse utile (%s/%s appels API en échec — quota, clé, modèle indisponible, etc.).",
                model_name,
                failures,
                total_q,
            )
            domain_results.append(
                BenchmarkResult(
                    model_name=model_name,
                    domain=f"échecs API ({failures}/{total_q})",
                    accuracy=0.0,
                    f1_score=0.0,
                    bleu_score=0.0,
                    rouge_l=0.0,
                    avg_latency_ms=0.0,
                    total_questions=total_q,
                    correct_answers=0,
                )
            )

        return domain_results

    def run_benchmark(self, dataset: List[Dict]) -> Dict:
        """Lance le benchmark complet sur tous les modèles."""
        logger.info("\n" + "🏆" * 30)
        logger.info("DÉMARRAGE DU BENCHMARK LLM")
        logger.info("🏆" * 30)

        self.results = []

        if not self.models:
            return {
                "results": {},
                "best_model": {
                    "name": None,
                    "score": None,
                    "all_scores": {},
                    "note": "Aucun modèle chargé : renseignez OPENAI_API_KEY et/ou HUGGINGFACE_TOKEN.",
                },
            }

        all_results = {}
        for model_name in self.models:
            results = self.evaluate_model(model_name, dataset)
            all_results[model_name] = results
            self.results.extend(results)

        best = self.select_best_model()
        return {
            "results": {k: [asdict(r) for r in v] for k, v in all_results.items()},
            "best_model": best,
        }

    def select_best_model(self) -> Dict:
        """Sélectionne le meilleur modèle basé sur un score composite."""
        model_scores = {}
        for result in self.results:
            name = result.model_name
            if name not in model_scores:
                model_scores[name] = []
            # Score composite pondéré
            composite = (
                0.3 * result.accuracy
                + 0.25 * result.f1_score
                + 0.2 * result.rouge_l
                + 0.15 * result.bleu_score
                + 0.1 * (1 - min(result.avg_latency_ms / 5000, 1))  # Bonus rapidité
            )
            model_scores[name].append(composite)

        avg_scores = {k: sum(v) / len(v) for k, v in model_scores.items()}
        if not avg_scores:
            logger.warning("Aucun score agrégé : tous les appels ont peut-être échoué.")
            return {
                "name": None,
                "score": None,
                "all_scores": {},
                "note": "Aucune métrique (erreurs API sur tous les modèles ou scores vides).",
            }

        best_name = max(avg_scores, key=avg_scores.get)

        logger.info(f"\n🏆 MEILLEUR MODÈLE : {best_name} (score={avg_scores[best_name]:.4f})")
        return {"name": best_name, "score": avg_scores[best_name], "all_scores": avg_scores}

    def save_results(self, output: Dict, filename: str = "benchmark_results.json") -> Path:
        """Sauvegarde les résultats du benchmark."""
        path = EVALUATION_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Résultats sauvegardés → {path}")
        return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    benchmark = LLMBenchmark()
    benchmark.load_models()

    # Charger le dataset
    from src.dataset_generator import DatasetGenerator
    gen = DatasetGenerator()
    try:
        dataset = gen.load_dataset()
    except FileNotFoundError:
        dataset = gen.generate_local_dataset()
        gen.save_dataset(dataset)

    from src.env_persist import apply_benchmark_winner_to_chat

    results = benchmark.run_benchmark(dataset)
    results["chat_alignment"] = apply_benchmark_winner_to_chat(results.get("best_model") or {})
    benchmark.save_results(results)
