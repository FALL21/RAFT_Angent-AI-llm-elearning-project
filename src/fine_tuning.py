"""
ÉTAPE 5 — Fine-Tuning : LoRA, QLoRA et Full Fine-Tuning.
Adapte un LLM pré-entraîné au domaine spécifique du dataset.

Exécution réelle : définir FINETUNE_RUN=1 (ex. sur Kaggle avec GPU).
Sinon `run.py --step finetune` ne fait qu’afficher la config (mode sec).

Multi-GPU : par défaut le modèle est chargé sur cuda:0 seul pour éviter les erreurs de device avec QLoRA ;
FINETUNE_DEVICE_MAP=auto pour forcer le sharding.
"""
import dataclasses
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import BASE_DIR, EVALUATION_DIR, FINE_TUNING_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)

_BNB_QLORA_FAIL_HINT = (
    "QLoRA (bitsandbytes 4-bit) a échoué sur ce GPU/CUDA. "
    "Sur Kaggle, exécute avant le script : !pip install -U 'bitsandbytes>=0.45.0' "
    "(ou une version alignée sur le CUDA du notebook). "
    "Sinon : os.environ['FINETUNE_METHOD']='lora' (FP16, besoin de plus de VRAM) "
    "ou FINETUNE_QLORA_FALLBACK_LORA=1 pour basculer automatiquement en LoRA."
)


def _is_bnb_cuda_kernel_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return (
        "no kernel image" in text
        or "cudaerror" in text
        or "kernel image" in text
        or "acceleratorerror" in text
    )


def _finetune_device_map():
    """
    Carte des devices pour `from_pretrained` pendant l'entraînement.

    Avec `device_map="auto"` et 2+ GPU (ex. Kaggle T4×2), le modèle est étalé sur plusieurs cartes ;
    la loss causal LM + PEFT peut alors mélanger cuda:0 et cuda:1 → RuntimeError.
    Par défaut : tout sur `cuda:0` dès qu'il y a plusieurs GPU.
    """
    import torch

    raw = (os.getenv("FINETUNE_DEVICE_MAP") or "").strip().lower()
    if raw == "auto":
        return "auto"
    if raw in ("single", "0", "cuda0"):
        return {"": 0}
    if raw.isdigit():
        return {"": int(raw)}
    if os.getenv("FINETUNE_MULTI_GPU", "").strip().lower() in ("1", "true", "yes"):
        return "auto"
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Fine-tuning : %d GPU(s) détecté(s) — chargement du modèle sur cuda:0 uniquement "
            "(QLoRA/PEFT). Pour forcer le sharding multi-GPU : FINETUNE_DEVICE_MAP=auto.",
            torch.cuda.device_count(),
        )
        return {"": 0}
    return "auto"


def _training_args_eval_kw(eval_dataset) -> Dict[str, str]:
    """`eval_strategy` (transformers ≥ 4.46) ou `evaluation_strategy` (anciennes versions)."""
    import inspect

    from transformers import TrainingArguments

    mode = "epoch" if eval_dataset else "no"
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"eval_strategy": mode}
    return {"evaluation_strategy": mode}


def _sft_trainer_accepts_legacy_dataset_kwargs() -> bool:
    """TRL ≤ 0.13 : `dataset_text_field` / `max_seq_length` sur SFTTrainer ; TRL récent : SFTConfig + processing_class."""
    import inspect

    from trl import SFTTrainer

    return "dataset_text_field" in inspect.signature(SFTTrainer.__init__).parameters


def _make_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    train_cfg: dict,
    *,
    per_device_train_batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    warmup_steps: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
):
    """
    Instancie SFTTrainer en fonction de la version de TRL / Transformers installée (ex. Kaggle vs requirements.txt).
    """
    from trl import SFTTrainer

    bs = per_device_train_batch_size if per_device_train_batch_size is not None else train_cfg["batch_size"]
    lr = learning_rate if learning_rate is not None else train_cfg["learning_rate"]
    ws = warmup_steps if warmup_steps is not None else train_cfg["warmup_steps"]
    gas = (
        gradient_accumulation_steps
        if gradient_accumulation_steps is not None
        else train_cfg["gradient_accumulation_steps"]
    )

    common = {
        "output_dir": output_dir,
        "num_train_epochs": train_cfg["num_epochs"],
        "per_device_train_batch_size": bs,
        "learning_rate": lr,
        "warmup_steps": ws,
        "gradient_accumulation_steps": gas,
        "fp16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        **_training_args_eval_kw(eval_dataset),
    }

    if _sft_trainer_accepts_legacy_dataset_kwargs():
        from transformers import TrainingArguments

        training_args = TrainingArguments(**common)
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=train_cfg["max_seq_length"],
        )

    from trl import SFTConfig

    sft_field_names = {f.name for f in dataclasses.fields(SFTConfig)}
    seq_kw: Dict[str, int] = {}
    # TRL récent : `max_length` ; TRL 0.13 : `max_seq_length`
    if "max_length" in sft_field_names:
        seq_kw["max_length"] = train_cfg["max_seq_length"]
    elif "max_seq_length" in sft_field_names:
        seq_kw["max_seq_length"] = train_cfg["max_seq_length"]

    sft_args = SFTConfig(
        **common,
        dataset_text_field="text",
        **seq_kw,
    )
    return SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        logger.info("=" * 60)
        logger.info("FINE-TUNING LoRA")
        logger.info("=" * 60)

        # Charger le modèle et tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map=_finetune_device_map(),
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

        train_cfg = self.config["training"]
        output_path = self.output_dir / "lora"
        trainer = _make_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(output_path),
            train_cfg=train_cfg,
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

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

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config,
                device_map=_finetune_device_map(),
                trust_remote_code=True,
            )
        except Exception as e:
            if _is_bnb_cuda_kernel_failure(e) and os.getenv(
                "FINETUNE_QLORA_FALLBACK_LORA", ""
            ).strip().lower() in ("1", "true", "yes"):
                logger.warning(
                    "QLoRA indisponible sur ce runtime (%s) — repli automatique en LoRA FP16.",
                    type(e).__name__,
                )
                return self.train_lora(train_dataset, eval_dataset=eval_dataset)
            if _is_bnb_cuda_kernel_failure(e):
                raise RuntimeError(_BNB_QLORA_FAIL_HINT) from e
            raise
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
        trainer = _make_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(output_path),
            train_cfg=train_cfg,
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("=" * 60)
        logger.info("FULL FINE-TUNING")
        logger.info("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map=_finetune_device_map(),
            trust_remote_code=True,
        )

        train_cfg = self.config["training"]
        output_path = self.output_dir / "full_ft"
        trainer = _make_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(output_path),
            train_cfg=train_cfg,
            per_device_train_batch_size=max(1, train_cfg["batch_size"] // 2),
            learning_rate=train_cfg["learning_rate"] / 10,
            warmup_steps=train_cfg["warmup_steps"] * 2,
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"] * 2,
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


def build_hf_sft_dataset(dataset_rows: List[Dict], base_model_id: str):
    """Construit un `datasets.Dataset` avec colonne `text` pour SFTTrainer."""
    from datasets import Dataset
    from transformers import AutoTokenizer

    formatter = DatasetFormatter()
    chat_rows = formatter.to_chat_format(dataset_rows)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    texts: List[str] = []
    for row in chat_rows:
        messages = row["messages"]
        if getattr(tokenizer, "chat_template", None):
            t = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            t = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        texts.append(t)
    return Dataset.from_dict({"text": texts})


def _train_val_split(ds, eval_ratio: float, seed: int):
    """Découpe train / eval ; eval vide si trop peu d’exemples."""
    n = len(ds)
    if n < 2:
        raise ValueError("Le fine-tuning requiert au moins 2 exemples.")
    test_n = max(1, min(int(round(n * eval_ratio)), n - 1))
    if test_n <= 0 or n - test_n < 1:
        return ds, None
    split = ds.train_test_split(test_size=test_n, seed=seed)
    return split["train"], split["test"]


def run_finetune_from_evaluation_dataset(
    dataset_rows: List[Dict],
    base_model_id: str,
    method: str = "qlora",
    eval_ratio: float = 0.1,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> FineTuningResult:
    """
    Charge les Q/R, formate en texte SFT, entraîne LoRA / QLoRA / full, sauvegarde sous ``models/``.
    """
    if max_samples is not None and max_samples > 0:
        dataset_rows = dataset_rows[:max_samples]

    if len(dataset_rows) < 2:
        raise ValueError(
            f"Pas assez d'exemples pour entraîner ({len(dataset_rows)}). "
            "Générez d'abord data/evaluation/dataset_evaluation.json (étape dataset)."
        )

    method = method.lower().strip()
    if method not in ("qlora", "lora", "full"):
        raise ValueError("FINETUNE_METHOD doit être qlora, lora ou full")

    ds = build_hf_sft_dataset(dataset_rows, base_model_id)
    train_ds, eval_ds = _train_val_split(ds, eval_ratio=eval_ratio, seed=seed)

    tuner = FineTuner(base_model_id)
    if method == "qlora":
        result = tuner.train_qlora(train_ds, eval_dataset=eval_ds)
    elif method == "lora":
        result = tuner.train_lora(train_ds, eval_dataset=eval_ds)
    else:
        result = tuner.train_full(train_ds, eval_dataset=eval_ds)

    save_finetune_manifest(
        result,
        extra={
            "train_examples": len(train_ds),
            "eval_examples": len(eval_ds) if eval_ds is not None else 0,
            "eval_ratio": eval_ratio,
            "seed": seed,
            "max_samples_applied": max_samples,
        },
    )
    return result


def save_finetune_manifest(result: FineTuningResult, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Écrit ``data/evaluation/finetune_manifest.json`` (suivi + reprise en local)."""
    path = EVALUATION_DIR / "finetune_manifest.json"
    mp = Path(result.model_path).resolve()
    base = BASE_DIR.resolve()
    try:
        adapter_rel = str(mp.relative_to(base))
    except ValueError:
        adapter_rel = str(mp)
    payload: Dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **asdict(result),
        "adapter_path_project_relative": adapter_rel,
        "continuer_en_local": {
            "copier_vers_projet": [
                f"Copier le dossier entier « models/ » (ou seulement {adapter_rel}) "
                "à la racine du clone local du projet.",
            ],
            "charger_ladaptateur": (
                "from transformers import AutoModelForCausalLM; from peft import PeftModel; "
                f'base = AutoModelForCausalLM.from_pretrained("{result.base_model}", '
                "trust_remote_code=True, torch_dtype='auto', device_map='auto'); "
                f'model = PeftModel.from_pretrained(base, "{adapter_rel}")'
            ),
        },
    }
    if extra:
        payload["run"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Manifeste fine-tuning → %s", path)
    return path


def finetune_dry_run_summary(
    dataset_rows: List[Dict],
    base_model_id: str,
) -> Dict[str, Any]:
    """Résumé sans GPU (mode par défaut de l’étape finetune)."""
    return {
        "mode": "dry_run",
        "base_model": base_model_id,
        "num_examples": len(dataset_rows),
        "hint": "Pour entraîner : FINETUNE_RUN=1 python run.py --step finetune (GPU requis pour qlora/lora).",
    }
