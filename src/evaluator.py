"""
Module d'évaluation comparative de toutes les pipelines.
Compare : RAG Simple, RAG Avancé, RAFT, Agent RAG, Multi-Agent.
"""
import json
import time
import logging
from typing import List, Dict
from pathlib import Path
from datetime import datetime

from src.config import EVALUATION_DIR, EVALUATION_CONFIG

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Évalue et compare les performances de chaque pipeline."""

    def __init__(self):
        self.results: Dict[str, List[Dict]] = {}
        self.metrics_names = EVALUATION_CONFIG["metrics"]

    def evaluate_pipeline(self, pipeline, pipeline_name: str, dataset: List[Dict]) -> Dict:
        """Évalue une pipeline sur le dataset complet."""
        logger.info(f"\n{'─'*50}")
        logger.info(f"Évaluation : {pipeline_name}")
        logger.info(f"{'─'*50}")

        scores = []
        latencies = []

        for i, item in enumerate(dataset):
            try:
                start = time.time()
                result = pipeline.query(item["question"])
                latency = (time.time() - start) * 1000

                prediction = result.get("answer", "")
                reference = item["answer"]

                metrics = self._compute_metrics(prediction, reference)
                metrics["latency_ms"] = latency
                metrics["question_id"] = item.get("id", str(i))
                metrics["domain"] = item.get("domain", "")

                scores.append(metrics)
                latencies.append(latency)

                if (i + 1) % 50 == 0:
                    logger.info(f"  Progression : {i+1}/{len(dataset)}")

            except Exception as e:
                logger.error(f"  Erreur question {i} : {e}")
                scores.append({"error": str(e), "question_id": item.get("id", str(i))})

        # Agrégation
        valid_scores = [s for s in scores if "error" not in s]
        summary = {
            "pipeline": pipeline_name,
            "total_questions": len(dataset),
            "successful": len(valid_scores),
            "errors": len(dataset) - len(valid_scores),
            "avg_accuracy": self._mean([s.get("accuracy", 0) for s in valid_scores]),
            "avg_f1": self._mean([s.get("f1_score", 0) for s in valid_scores]),
            "avg_bleu": self._mean([s.get("bleu_score", 0) for s in valid_scores]),
            "avg_rouge_l": self._mean([s.get("rouge_l", 0) for s in valid_scores]),
            "avg_latency_ms": self._mean(latencies),
            "p95_latency_ms": self._percentile(latencies, 95),
        }

        # Par domaine
        domains = set(s.get("domain", "") for s in valid_scores)
        summary["by_domain"] = {}
        for domain in domains:
            domain_scores = [s for s in valid_scores if s.get("domain") == domain]
            summary["by_domain"][domain] = {
                "accuracy": self._mean([s.get("accuracy", 0) for s in domain_scores]),
                "f1": self._mean([s.get("f1_score", 0) for s in domain_scores]),
                "count": len(domain_scores),
            }

        self.results[pipeline_name] = {"summary": summary, "details": scores}

        logger.info(f"  Résultats {pipeline_name}:")
        logger.info(f"    Accuracy = {summary['avg_accuracy']:.2%}")
        logger.info(f"    F1       = {summary['avg_f1']:.3f}")
        logger.info(f"    ROUGE-L  = {summary['avg_rouge_l']:.3f}")
        logger.info(f"    Latence  = {summary['avg_latency_ms']:.0f} ms")

        return summary

    def _compute_metrics(self, prediction: str, reference: str) -> Dict:
        """Calcule les métriques pour une paire prédiction/référence."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        # F1 basé sur les tokens
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        if not pred_set or not ref_set:
            f1 = 0.0
        else:
            p = len(pred_set & ref_set) / len(pred_set)
            r = len(pred_set & ref_set) / len(ref_set)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # ROUGE-L (approximation simplifiée via LCS)
        rouge_l = self._lcs_ratio(pred_tokens, ref_tokens)

        # BLEU simplifié (unigram)
        if pred_tokens:
            bleu = sum(1 for t in pred_tokens if t in ref_set) / len(pred_tokens)
        else:
            bleu = 0.0

        return {
            "accuracy": 1.0 if f1 > 0.3 else 0.0,
            "f1_score": f1,
            "bleu_score": bleu,
            "rouge_l": rouge_l,
        }

    def _lcs_ratio(self, seq1, seq2) -> float:
        """Calcule le ratio LCS (Longest Common Subsequence)."""
        if not seq1 or not seq2:
            return 0.0
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]
        precision = lcs_len / m if m else 0
        recall = lcs_len / n if n else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def _mean(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _percentile(self, values: List[float], p: int) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def compare_all(self) -> Dict:
        """Compare toutes les pipelines évaluées."""
        if not self.results:
            return {}

        comparison = {}
        for name, data in self.results.items():
            comparison[name] = data["summary"]

        # Classement
        ranking = sorted(
            comparison.items(),
            key=lambda x: x[1]["avg_f1"],
            reverse=True,
        )

        return {
            "comparison": comparison,
            "ranking": [{"rank": i + 1, "pipeline": name, "f1": data["avg_f1"]}
                        for i, (name, data) in enumerate(ranking)],
            "best_pipeline": ranking[0][0] if ranking else None,
        }

    def save_report(self, filename: str = "evaluation_report.json") -> Path:
        """Sauvegarde le rapport complet."""
        path = EVALUATION_DIR / filename
        report = {
            "generated_at": datetime.now().isoformat(),
            "results": {k: v["summary"] for k, v in self.results.items()},
            "comparison": self.compare_all(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Rapport sauvegardé → {path}")
        return path
