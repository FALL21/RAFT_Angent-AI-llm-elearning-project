"""
Instanciation unique du modèle de chat (OpenAI ou Hugging Face via le routeur /v1).
Lit src.config à chaque appel pour refléter les changements depuis l’UI / .env.
"""
from __future__ import annotations

import src.config as cfg


def _hf_router_chat_llm(api_key: str):
    """ChatOpenAI → endpoint compatible OpenAI du routeur Hugging Face."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=cfg.HF_CHAT_MODEL_ID,
        temperature=0.1,
        base_url=cfg.HF_ROUTER_BASE_URL,
        api_key=api_key,
    )


def get_default_chat_llm():
    """
    Retourne le LLM utilisé par Flask, run.py et les pipelines RAG/agents.

    - LLM_PROVIDER=openai : uniquement OpenAI (clé obligatoire).
    - LLM_PROVIDER=huggingface : routeur HF (token + HF_CHAT_MODEL_ID).
    - LLM_PROVIDER=auto : OpenAI si une clé est définie, sinon Hugging Face.
    """
    provider = (cfg.LLM_PROVIDER or "auto").strip().lower()
    okey = (cfg.OPENAI_API_KEY or "").strip()
    hf = (cfg.HUGGINGFACE_TOKEN or "").strip()
    omodel = (cfg.OPENAI_CHAT_MODEL or "gpt-4o-mini").strip()

    if provider == "huggingface":
        if not hf:
            raise RuntimeError(
                "LLM_PROVIDER=huggingface : renseignez HUGGINGFACE_TOKEN dans .env "
                "(https://huggingface.co/settings/tokens)."
            )
        return _hf_router_chat_llm(hf)

    if provider == "openai":
        if not okey:
            raise RuntimeError(
                "LLM_PROVIDER=openai : renseignez OPENAI_API_KEY dans .env."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=omodel, temperature=0.1)

    # auto
    if okey:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=omodel, temperature=0.1)
    if hf:
        return _hf_router_chat_llm(hf)
    raise RuntimeError(
        "Aucun LLM configuré : définissez OPENAI_API_KEY et/ou HUGGINGFACE_TOKEN dans .env, "
        "ou LLM_PROVIDER=huggingface avec HUGGINGFACE_TOKEN si le quota OpenAI est épuisé (erreur 429)."
    )
