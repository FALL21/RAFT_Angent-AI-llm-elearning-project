"""
Persistance des paramètres dans .env et application à chaud sur le module src.config.
Les secrets vides en entrée ne remplacent pas les valeurs existantes.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import set_key

import src.config as cfg

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

def _ensure_env_file() -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_PATH.is_file():
        ENV_PATH.write_text("", encoding="utf-8")


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "••••"
    return "••••••" + value[-4:]


def get_public_settings() -> Dict[str, Any]:
    """État courant (pas de clés en clair)."""
    oa = (cfg.OPENAI_API_KEY or "").strip()
    hf = (cfg.HUGGINGFACE_TOKEN or "").strip()
    bench = {
        name: entry.get("model_id", "")
        for name, entry in cfg.LLM_MODELS.items()
    }
    return {
        "openai_key_set": bool(oa),
        "openai_key_hint": _mask_secret(oa) if oa else "",
        "hf_token_set": bool(hf),
        "hf_token_hint": _mask_secret(hf) if hf else "",
        "llm_provider": (cfg.LLM_PROVIDER or "auto").strip().lower(),
        "hf_chat_model_id": (cfg.HF_CHAT_MODEL_ID or "").strip(),
        "openai_chat_model": (cfg.OPENAI_CHAT_MODEL or "gpt-4o-mini").strip(),
        "hf_router_base_url": (cfg.HF_ROUTER_BASE_URL or "").strip(),
        "rag_top_k": int(cfg.RAG_TOP_K),
        "chunk_size": int(cfg.CHUNK_SIZE),
        "benchmark_models": bench,
    }


def _write_env(key: str, value: str) -> None:
    _ensure_env_file()
    set_key(str(ENV_PATH), key, value, quote_mode="always")


def apply_settings(
    body: Dict[str, Any],
    *,
    allow_secrets: bool = True,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Applique les paramètres (fichier .env + os.environ + attributs cfg).
    Retourne (résumé pour JSON, message d'erreur ou None).
    """
    oa = body.get("openai_api_key")
    if allow_secrets and isinstance(oa, str) and oa.strip():
        v = oa.strip()
        _write_env("OPENAI_API_KEY", v)
        os.environ["OPENAI_API_KEY"] = v
        cfg.OPENAI_API_KEY = v

    hf = body.get("huggingface_token")
    if allow_secrets and isinstance(hf, str) and hf.strip():
        v = hf.strip()
        _write_env("HUGGINGFACE_TOKEN", v)
        os.environ["HUGGINGFACE_TOKEN"] = v
        cfg.HUGGINGFACE_TOKEN = v

    lp = body.get("llm_provider")
    if isinstance(lp, str) and lp.strip():
        v = lp.strip().lower()
        if v not in ("auto", "openai", "huggingface"):
            return {}, "llm_provider doit être auto, openai ou huggingface."
        _write_env("LLM_PROVIDER", v)
        os.environ["LLM_PROVIDER"] = v
        cfg.LLM_PROVIDER = v

    hf_mid = body.get("hf_chat_model_id")
    if isinstance(hf_mid, str) and hf_mid.strip():
        v = hf_mid.strip()
        _write_env("HF_CHAT_MODEL_ID", v)
        os.environ["HF_CHAT_MODEL_ID"] = v
        cfg.HF_CHAT_MODEL_ID = v

    ocm = body.get("openai_chat_model")
    if isinstance(ocm, str) and ocm.strip():
        v = ocm.strip()
        if not re.match(r"^[\w.\-:/]+$", v):
            return {}, "openai_chat_model : caractères non autorisés."
        _write_env("OPENAI_CHAT_MODEL", v)
        os.environ["OPENAI_CHAT_MODEL"] = v
        cfg.OPENAI_CHAT_MODEL = v

    if "rag_top_k" in body and body["rag_top_k"] is not None:
        try:
            k = int(body["rag_top_k"])
        except (TypeError, ValueError):
            return {}, "rag_top_k doit être un entier."
        if k < 1 or k > 20:
            return {}, "rag_top_k doit être entre 1 et 20."
        sv = str(k)
        _write_env("RAG_TOP_K", sv)
        os.environ["RAG_TOP_K"] = sv
        cfg.RAG_TOP_K = k

    chunk_changed = False
    if "chunk_size" in body and body["chunk_size"] is not None:
        try:
            cs = int(body["chunk_size"])
        except (TypeError, ValueError):
            return {}, "chunk_size doit être un entier."
        if cs < 128 or cs > 2048 or cs % 128 != 0:
            return {}, "chunk_size doit être entre 128 et 2048 (multiple de 128)."
        if cs != int(cfg.CHUNK_SIZE):
            chunk_changed = True
        sv = str(cs)
        _write_env("CHUNK_SIZE", sv)
        os.environ["CHUNK_SIZE"] = sv
        cfg.CHUNK_SIZE = cs

    note = None
    if chunk_changed:
        note = (
            "Chunk size modifié : relancez l’indexation (Documents → Indexer) pour "
            "recalculer les chunks avec la nouvelle taille."
        )

    return {
        "ok": True,
        "note": note,
        "invalidate_vector_cache": chunk_changed,
        "settings": get_public_settings(),
    }, None


def apply_benchmark_winner_to_chat(best_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Après un benchmark : utilise le modèle gagnant pour le chat.
    Met à jour HF_CHAT_MODEL_ID (ID Hub du slot LLM_MODELS) et, si un token HF est
    disponible, force LLM_PROVIDER=huggingface pour que le Chat et les pipelines
    passent par le routeur avec ce modèle.
    """
    name = best_model.get("name")
    if not name or not isinstance(name, str):
        return {"applied": False, "reason": "no_winner"}

    if name not in cfg.LLM_MODELS:
        return {"applied": False, "reason": "unknown_model_key", "name": name}

    model_id = (cfg.LLM_MODELS[name].get("model_id") or "").strip()
    if not model_id:
        return {"applied": False, "reason": "empty_model_id"}

    _write_env("HF_CHAT_MODEL_ID", model_id)
    os.environ["HF_CHAT_MODEL_ID"] = model_id
    cfg.HF_CHAT_MODEL_ID = model_id

    hf = (cfg.HUGGINGFACE_TOKEN or "").strip()
    if not hf:
        logger.warning(
            "Benchmark gagnant %s → %s enregistré pour le chat, mais aucun HUGGINGFACE_TOKEN : "
            "le Chat peut échouer tant que le token n’est pas configuré.",
            name,
            model_id,
        )
        return {
            "applied": True,
            "partial": True,
            "benchmark_winner_key": name,
            "hf_chat_model_id": model_id,
            "note": "HF_CHAT_MODEL_ID mis à jour ; token HF requis pour utiliser le routeur.",
        }

    _write_env("LLM_PROVIDER", "huggingface")
    os.environ["LLM_PROVIDER"] = "huggingface"
    cfg.LLM_PROVIDER = "huggingface"

    logger.info(
        "Chat aligné sur le benchmark : %s → %s (LLM_PROVIDER=huggingface)",
        name,
        model_id,
    )
    return {
        "applied": True,
        "benchmark_winner_key": name,
        "hf_chat_model_id": model_id,
        "llm_provider": "huggingface",
    }
