"""
ÉTAPE 5 — Fine-Tuning : LoRA, QLoRA et Full Fine-Tuning.
Adapte un LLM pré-entraîné au domaine spécifique du dataset.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.config import FINE_TUNING_CONFIG, MODELS_DIR, EVALUATION_DIR

logger = logging.getLogger(__name__)


@dataclass
class FineTuningResult:
    method: str
    base_model: str
    epochs: int
    final_loss: float
    eval_loss: float
    training_time_min: float
    model_path: str


class DatasetFormatter:
    """Formate le dataset d'évaluation pour le fine-tuning."""

    @staticmethod
    def to_instruction_format(dataset: List[Dict]) -> List[Dict]:
        """Convertit en format instruction (Alpaca-style)."""
        formatted = []
        for item in dataset:
            formatted.append({
                "instruction": item["question"],
                "input": f"Domaine: {item.get('domain', 'général')}",
                "output": item["answer"],
            })
        return formatted

    @staticmethod
    def to_chat_format(dataset: List[Dict]) -> List[Dict]:
        """Convertit en format conversationnel (ChatML)."""
        formatted = []
        for item in dataset:
            formatted.append({
                "messages": [
                    {"role": "system", "content": f"Tu es un expert en {item.get('domain', 'général')}. Réponds avec précision."},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ]
            })
        return formatted

    @staticmethod
    def save_formatted(data: List[Dict], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Dataset formaté sauvegardé → {path} ({len(data)} exemples)")


class FineTuner:
    """
    Fine-Tuning avec 3 méthodes :
    1. LoRA (Low-Rank Adaptation)
    2. QLoRA (Quantized LoRA)
    3. Full Fine-Tuning
    """

    def __init__(self, base_model_id: str, output_dir: Optional[Path] = None):
        self.base_model_id = base_model_id
        self.output_dir = output_dir or MODELS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = FINE_TUNING_CONFIG

    # ── LoRA ────────────────────────────────────────────────
    def train_lora(self, train_dataset, eval_dataset=None) -> FineTuningResult:
        """Fine-tuning avec LoRA (Low-Rank Adaptation)."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer

        logger.info("=" * 60)
        logger.info("FINE-TUNING LoRA")
        logger.info("=" * 60)

        # Charger le modèle et tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Configuration LoRA
        lora_cfg = self.config["lora"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Arguments d'entraînement
        train_cfg = self.config["training"]
        output_path = self.output_dir / "lora"
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=train_cfg["num_epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            warmup_steps=train_cfg["warmup_steps"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            report_to="none",
        )

        # Entraînement
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=train_cfg["max_seq_length"],
        )

        import time
        start = time.time()
        result = trainer.train()
        duration = (time.time() - start) / 60

        # Sauvegarder
        model.save_pretrained(str(output_path / "final"))
        tokenizer.save_pretrained(str(output_path / "final"))

        return FineTuningResult(
            method="lora",
            base_model=self.base_model_id,
            epochs=train_cfg["num_epochs"],
            final_loss=result.training_loss,
            eval_loss=result.metrics.get("eval_loss", -1),
            training_time_min=duration,
            model_path=str(output_path / "final"),
        )

    # ── QLoRA ───────────────────────────────────────────────
    def train_qlora(self, train_dataset, eval_dataset=None) -> FineTuningResult:
        """Fine-tuning avec QLoRA (4-bit quantization + LoRA)."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from trl import SFTTrainer

        logger.info("=" * 60)
        logger.info("FINE-TUNING QLoRA (4-bit)")
        logger.info("=" * 60)

        qlora_cfg = self.config["qlora"]

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=qlora_cfg["r"],
            lora_alpha=qlora_cfg["lora_alpha"],
            lora_dropout=qlora_cfg["lora_dropout"],
            target_modules=self.config["lora"]["target_modules"],
            bias="none",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        train_cfg = self.config["training"]
        output_path = self.output_dir / "qlora"
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=train_cfg["num_epochs"],
            per_device_train_batch_size=train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            warmup_steps=train_cfg["warmup_steps"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=train_cfg["max_seq_length"],
        )

        import time
        start = time.time()
        result = trainer.train()
        duration = (time.time() - start) / 60

        model.save_pretrained(str(output_path / "final"))
        tokenizer.save_pretrained(str(output_path / "final"))

        return FineTuningResult(
            method="qlora",
            base_model=self.base_model_id,
            epochs=train_cfg["num_epochs"],
            final_loss=result.training_loss,
            eval_loss=result.metrics.get("eval_loss", -1),
            training_time_min=duration,
            model_path=str(output_path / "final"),
        )

    # ── Full Fine-Tuning ────────────────────────────────────
    def train_full(self, train_dataset, eval_dataset=None) -> FineTuningResult:
        """Full fine-tuning (tous les paramètres)."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
        )
        from trl import SFTTrainer

        logger.info("=" * 60)
        logger.info("FULL FINE-TUNING")
        logger.info("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        train_cfg = self.config["training"]
        output_path = self.output_dir / "full_ft"
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=train_cfg["num_epochs"],
            per_device_train_batch_size=max(1, train_cfg["batch_size"] // 2),
            learning_rate=train_cfg["learning_rate"] / 10,  # LR plus faible
            warmup_steps=train_cfg["warmup_steps"] * 2,
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"] * 2,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=train_cfg["max_seq_length"],
        )

        import time
        start = time.time()
        result = trainer.train()
        duration = (time.time() - start) / 60

        model.save_pretrained(str(output_path / "final"))
        tokenizer.save_pretrained(str(output_path / "final"))

        return FineTuningResult(
            method="full",
            base_model=self.base_model_id,
            epochs=train_cfg["num_epochs"],
            final_loss=result.training_loss,
            eval_loss=result.metrics.get("eval_loss", -1),
            training_time_min=duration,
            model_path=str(output_path / "final"),
        )
