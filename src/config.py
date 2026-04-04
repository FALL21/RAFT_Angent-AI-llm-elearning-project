"""
Configuration centrale du projet RAG-LLM Multi-Agent.
Toutes les variables d'environnement et paramètres sont centralisés ici.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Avant tout import tokenizers / sentence-transformers : évite l’avertissement fork (Flask reloader).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Chroma lit aussi ces variables au chargement de Settings ; on force False pour éviter PostHog bruyant.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")

# huggingface_hub lit HF_INFERENCE_ENDPOINT au premier import de `constants`.
# L’URL historique api-inference.huggingface.co renvoie 410 ; le routeur la remplace.
if not (os.getenv("HF_INFERENCE_ENDPOINT") or "").strip():
    os.environ["HF_INFERENCE_ENDPOINT"] = "https://router.huggingface.co"

# ============================================================
# CHEMINS DU PROJET
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"
MODELS_DIR = BASE_DIR / "models"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Création automatique des dossiers
for d in [RAW_PDF_DIR, PROCESSED_DIR, EVALUATION_DIR, MODELS_DIR, VECTORSTORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# CLÉS API (à renseigner dans le fichier .env)
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
# openai | huggingface | auto — auto : OpenAI si OPENAI_API_KEY est renseignée, sinon HF.
# Utilisez huggingface si OpenAI renvoie 429 (quota / billing).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
# Le routeur HF /v1/chat/completions n’accepte que des modèles « chat » ; Mistral-7B-Instruct-v0.3 peut renvoyer model_not_supported.
HF_CHAT_MODEL_ID = os.getenv(
    "HF_CHAT_MODEL_ID",
    "Qwen/Qwen2.5-7B-Instruct",
)
# API compatible OpenAI sur le routeur HF (remplace l’ancien /models/… de HuggingFaceEndpoint)
HF_ROUTER_BASE_URL = os.getenv(
    "HF_ROUTER_BASE_URL",
    "https://router.huggingface.co/v1",
).strip()
# Modèle OpenAI pour LLM_PROVIDER=openai / auto (clé présente)
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# ============================================================
# MODÈLES LLM À BENCHMARKER (Étape 2)
# ============================================================
LLM_MODELS = {
    "qwen-7b": {
        "provider": "huggingface",
        "model_id": os.getenv(
            "BENCHMARK_HF_MODEL_QWEN",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        "description": "Qwen 2.5 7B Instruct (routeur HF /v1/chat)",
        "max_tokens": 4096,
    },
    "mistral-7b": {
        "provider": "huggingface",
        # Le 3B est souvent « not supported by any provider you have enabled » alors que le 7B passe ;
        # défaut = 7B pour que les 3 cartes tournent. Différencier via BENCHMARK_HF_MODEL_MISTRAL.
        "model_id": os.getenv(
            "BENCHMARK_HF_MODEL_MISTRAL",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        "description": "Slot benchmark — défaut Qwen 2.5 7B (même base que qwen-7b si providers limités). Surcharger pour un autre modèle chat.",
        "max_tokens": 4096,
    },
    "llama3-8b": {
        "provider": "huggingface",
        # Sur le routeur /v1/chat, l’ID Hub « Llama-3.1-8B-Instruct » est reconnu ; « Meta-Llama-3.1-8B-Instruct » peut renvoyer model_not_found.
        "model_id": os.getenv(
            "BENCHMARK_HF_MODEL_LLAMA",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
        "description": "Llama 3.1 8B Instruct (gated Meta — accès Hub requis)",
        "max_tokens": 4096,
    },
}

# ============================================================
# PARAMÈTRES EMBEDDING & VECTORSTORE
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
VECTOR_DB_TYPE = "chroma"  # chroma | faiss

# ============================================================
# PARAMÈTRES RAG
# ============================================================
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # Nombre de documents récupérés
RAG_SCORE_THRESHOLD = 0.3        # Seuil de similarité
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============================================================
# PARAMÈTRES FINE-TUNING
# ============================================================
FINE_TUNING_CONFIG = {
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    },
    "qlora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1024,
    },
}

# ============================================================
# PARAMÈTRES RAFT (Retrieval Augmented Fine-Tuning)
# ============================================================
RAFT_CONFIG = {
    "num_distractors": 3,          # Documents distracteurs par question
    "oracle_probability": 0.8,     # Probabilité d'inclure le vrai document
    "chain_of_thought": True,      # Générer des raisonnements CoT
}

# ============================================================
# PARAMÈTRES AGENTS IA
# ============================================================
AGENT_CONFIG = {
    "max_iterations": 10,
    "tools": ["web_search", "pdf_reader", "calculator", "code_executor"],
    "search_engine": "duckduckgo",
    "agent_type": "react",          # react | plan_and_execute
}

# ============================================================
# PARAMÈTRES MULTI-AGENTS
# ============================================================
MULTI_AGENT_CONFIG = {
    "orchestrator": "supervisor",    # supervisor | hierarchical
    "agents": {
        "researcher": {
            "role": "Chercheur",
            "description": "Recherche des informations sur le web et dans les documents",
            "tools": ["web_search", "pdf_reader"],
        },
        "analyst": {
            "role": "Analyste",
            "description": "Analyse et synthétise les informations récupérées",
            "tools": ["calculator", "code_executor"],
        },
        "writer": {
            "role": "Rédacteur",
            "description": "Rédige des réponses claires et structurées",
            "tools": [],
        },
    },
}

# ============================================================
# PARAMÈTRES FLASK
# ============================================================
FLASK_CONFIG = {
    "SECRET_KEY": os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-prod"),
    "DEBUG": os.getenv("FLASK_DEBUG", "True").lower() == "true",
    "HOST": os.getenv("FLASK_HOST", "0.0.0.0"),
    "PORT": int(os.getenv("FLASK_PORT", 5000)),
}

# ============================================================
# PARAMÈTRES D'ÉVALUATION
# ============================================================
EVALUATION_CONFIG = {
    "num_questions": 300,
    "domains": ["e-learning"],
    "metrics": ["accuracy", "f1_score", "bleu", "rouge_l", "latency"],
    "test_split": 0.2,
    # Limite par défaut pour POST /api/dataset/generate (évite timeout navigateur ; CLI peut aller jusqu’à num_questions)
    "dataset_pdf_max_pairs": int(os.getenv("DATASET_PDF_MAX_PAIRS", "60")),
}

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
