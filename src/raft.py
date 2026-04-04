"""
ÉTAPE 6 — RAFT : Retrieval Augmented Fine-Tuning.
Combine RAG et Fine-Tuning en entraînant le modèle à :
  1. Identifier les documents pertinents parmi des distracteurs.
  2. Extraire l'information correcte pour répondre.
  3. Générer un raisonnement Chain-of-Thought avant la réponse.

Référence : "RAFT: Adapting Language Model to Domain Specific RAG" (2024).
"""
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain.schema import Document

from src.config import RAFT_CONFIG, FINE_TUNING_CONFIG, EVALUATION_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class RAFTDatasetBuilder:
    """
    Construit le dataset RAFT avec :
    - Documents oracles (contenant la réponse)
    - Documents distracteurs (ne contenant pas la réponse)
    - Raisonnement Chain-of-Thought
    """

    def __init__(
        self,
        chunks: List[Document],
        num_distractors: int = RAFT_CONFIG["num_distractors"],
        oracle_prob: float = RAFT_CONFIG["oracle_probability"],
        use_cot: bool = RAFT_CONFIG["chain_of_thought"],
    ):
        self.chunks = chunks
        self.num_distractors = num_distractors
        self.oracle_prob = oracle_prob
        self.use_cot = use_cot

    def build_raft_examples(self, qa_dataset: List[Dict]) -> List[Dict]:
        """
        Pour chaque Q/R du dataset, construit un exemple RAFT :
        - Avec probabilité P : inclut le document oracle + distracteurs
        - Avec probabilité 1-P : n'inclut que des distracteurs
        """
        raft_examples = []

        for item in qa_dataset:
            question = item["question"]
            answer = item["answer"]

            # Sélectionner un document oracle (le plus pertinent)
            oracle_doc = self._find_oracle_document(question, answer)

            # Sélectionner des distracteurs
            distractors = self._select_distractors(oracle_doc, self.num_distractors)

            # Décider si on inclut l'oracle
            include_oracle = random.random() < self.oracle_prob

            if include_oracle and oracle_doc:
                context_docs = [oracle_doc] + distractors
                random.shuffle(context_docs)
            else:
                context_docs = distractors

            # Formater le contexte
            context = self._format_context(context_docs)

            # Construire la réponse avec CoT
            if self.use_cot and oracle_doc:
                cot_answer = self._build_cot_response(question, answer, oracle_doc)
            else:
                cot_answer = answer

            raft_examples.append({
                "instruction": f"En te basant sur les documents fournis, réponds à la question suivante.\n\n"
                               f"DOCUMENTS :\n{context}\n\nQUESTION : {question}",
                "input": "",
                "output": cot_answer,
                "metadata": {
                    "domain": item.get("domain", ""),
                    "has_oracle": include_oracle and oracle_doc is not None,
                    "num_distractors": len(distractors),
                },
            })

        logger.info(f"Dataset RAFT construit : {len(raft_examples)} exemples")
        return raft_examples

    def _find_oracle_document(self, question: str, answer: str) -> Optional[Document]:
        """Trouve le chunk le plus pertinent pour la Q/R."""
        if not self.chunks:
            return None

        best_doc = None
        best_score = 0

        answer_words = set(answer.lower().split())
        for chunk in self.chunks:
            chunk_words = set(chunk.page_content.lower().split())
            overlap = len(answer_words & chunk_words) / max(len(answer_words), 1)
            if overlap > best_score:
                best_score = overlap
                best_doc = chunk

        return best_doc

    def _select_distractors(self, oracle: Optional[Document], k: int) -> List[Document]:
        """Sélectionne k documents distracteurs (différents de l'oracle)."""
        candidates = [c for c in self.chunks if c != oracle]
        return random.sample(candidates, min(k, len(candidates)))

    def _format_context(self, docs: List[Document]) -> str:
        """Formate les documents en contexte textuel."""
        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source_file", f"doc_{i}")
            parts.append(f"[Document {i+1} — {source}]\n{doc.page_content[:500]}")
        return "\n\n".join(parts)

    def _build_cot_response(self, question: str, answer: str, oracle: Document) -> str:
        """Construit une réponse avec Chain-of-Thought."""
        source = oracle.metadata.get("source_file", "le document fourni")
        return (
            f"<raisonnement>\n"
            f"La question demande : {question}\n"
            f"En analysant les documents fournis, je trouve l'information pertinente dans {source}.\n"
            f"Le passage clé mentionne les éléments nécessaires pour répondre.\n"
            f"</raisonnement>\n\n"
            f"<réponse>\n{answer}\n</réponse>"
        )

    def save_dataset(self, examples: List[Dict], filename: str = "raft_dataset.jsonl") -> Path:
        """Sauvegarde le dataset RAFT au format JSONL."""
        path = EVALUATION_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"Dataset RAFT sauvegardé → {path}")
        return path


class RAFTTrainer:
    """Entraîne un modèle avec la méthode RAFT."""

    def __init__(self, base_model_id: str):
        self.base_model_id = base_model_id
        self.output_dir = MODELS_DIR / "raft"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, raft_dataset_path: Path, method: str = "qlora"):
        """
        Lance l'entraînement RAFT en utilisant LoRA ou QLoRA.
        Le dataset RAFT est un fine-tuning spécialisé.
        """
        from src.fine_tuning import FineTuner
        from datasets import load_dataset

        logger.info(f"Entraînement RAFT avec méthode : {method}")

        # Charger le dataset RAFT
        dataset = load_dataset("json", data_files=str(raft_dataset_path), split="train")
        split = dataset.train_test_split(test_size=0.1)

        # Utiliser le FineTuner existant
        tuner = FineTuner(self.base_model_id, output_dir=self.output_dir)

        if method == "lora":
            result = tuner.train_lora(split["train"], split["test"])
        elif method == "qlora":
            result = tuner.train_qlora(split["train"], split["test"])
        else:
            result = tuner.train_full(split["train"], split["test"])

        logger.info(f"RAFT terminé — Loss finale : {result.final_loss:.4f}")
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print("Module RAFT prêt. Utiliser RAFTDatasetBuilder pour construire le dataset.")
