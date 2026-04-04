# RAG-LLM Multi-Agent System

Dépôt GitHub : [**RAFT_Angent-AI-llm-elearning-project**](https://github.com/FALL21/RAFT_Angent-AI-llm-elearning-project) (`FALL21`)

**Projet Master Intelligence Artificielle & Big Data**

Système complet de Retrieval-Augmented Generation (RAG) avec fine-tuning (RAFT) et architecture multi-agents, déployé via Docker et Flask.

---

## Table des matières

1. [Présentation du projet](#présentation-du-projet)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration (.env)](#configuration-env)
5. [Utilisation](#utilisation)
6. [Étapes du projet](#étapes-du-projet)
7. [Déploiement Docker](#déploiement-docker)
8. [API Reference](#api-reference)
9. [Structure du projet](#structure-du-projet)

---

## Présentation du projet

Ce projet implémente un pipeline NLP complet en 8 étapes, allant de la génération d'un dataset d'évaluation jusqu'au déploiement d'un système multi-agents intelligent. Les thèmes du **README historique** couvrent la Finance, l’**E-learning** et la Médecine ; la configuration actuelle du dataset d’évaluation et des templates peut être centrée sur l’**e-learning** ou sur vos **PDFs** (voir [Étape 1](#étape-1--génération-du-dataset)).

### Objectifs

- Générer et évaluer un dataset de paires question-réponse (cible 300), **aligné sur les PDFs** lorsque ceux-ci sont indexés
- Benchmarker plusieurs LLMs (OpenAI, modèles Hugging Face via le routeur d’inférence)
- Implémenter un RAG simple puis avancé avec reranking, HyDE et recherche hybride
- Appliquer du fine-tuning (LoRA, QLoRA, Full) et la méthode RAFT
- Déployer un système multi-agents avec orchestrateur superviseur
- Fournir une interface web interactive via Flask (+ Docker optionnel)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACE FLASK                            │
│              (Upload PDFs · Chat · Dashboard)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │    ORCHESTRATEUR        │
              │    (Superviseur)        │
              └────┬───────┬───────┬────┘
                   │       │       │
          ┌────────▼──┐ ┌──▼─────┐ ┌▼────────┐
          │ CHERCHEUR │ │ANALYSTE│ │RÉDACTEUR │
          │  (RAG +   │ │(Véri-  │ │(Synthèse │
          │   Web)    │ │fication│ │& Réponse)│
          └─────┬─────┘ └────────┘ └──────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───▼───┐ ┌────▼────┐ ┌────▼────┐
│Vector │ │Recherche│ │  PDF    │
│ Store │ │  Web    │ │ Reader  │
│(Chroma│ │(DDG)    │ │         │
└───────┘ └─────────┘ └─────────┘
```

### Pipelines disponibles

| Pipeline           | Description        | Composants                                      |
| ------------------ | ------------------ | ----------------------------------------------- |
| RAG Simple         | Retrieval basique  | LLM + VectorStore                               |
| RAG Avancé         | Retrieval optimisé | + Reranking + HyDE + BM25 hybride + Multi-Query |
| RAG + Agent        | Agent autonome     | + Outils (web, calcul, PDF)                     |
| RAFT + Multi-Agent | Système complet    | + Fine-tuning RAFT + 3 agents spécialisés       |

---

## Installation

### Prérequis

- **Python 3.11 ou 3.12** recommandé (certaines roues, ex. `bitsandbytes`, peuvent manquer sous Python 3.13 selon la plateforme)
- GPU recommandé pour le fine-tuning (optionnel pour RAG et embeddings CPU)
- Au moins une clé utilisable : **OpenAI** et/ou **Hugging Face** (token + modèle chat compatible routeur)

### Installation locale

```bash
# 1. Cloner le projet
git clone <url-du-repo>
cd rag-llm-elearning-project

# 2. Créer un environnement virtuel
python3.12 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env (voir section Configuration)

# 5. Placer vos PDFs dans data/raw_pdfs/

# 6. Lancer l'application
python run.py --step serve
```

L’URL d’écoute dépend de **`FLASK_PORT`** dans `.env` (par défaut souvent **5001** sur macOS si le port 5000 est pris par AirPlay).

### Installation Docker (recommandé)

```bash
# 1. Configurer .env
cp .env.example .env

# 2. Lancer avec Docker Compose
cd docker
docker-compose up --build

# 3. Accéder à l'interface (port défini dans docker-compose / FLASK)
# http://localhost:5000
```

---

## Configuration (.env)

Principales variables (voir **`.env.example`** pour la liste complète) :

| Variable                                                                              | Rôle                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY`                                                                      | Chat / agents en mode `LLM_PROVIDER=openai` (défaut modèle : `gpt-4o-mini` dans `llm_factory.py`)                                                                                                                                                                                       |
| `HUGGINGFACE_TOKEN`                                                                   | Authentification Hugging Face (inférence routeur, Hub)                                                                                                                                                                                                                                  |
| `LLM_PROVIDER`                                                                        | `auto` (OpenAI si clé présente, sinon HF), `openai`, ou `huggingface` — utile si quota OpenAI **429**                                                                                                                                                                                   |
| `HF_CHAT_MODEL_ID`                                                                    | Modèle **chat** pour le routeur HF (ex. `Qwen/Qwen2.5-7B-Instruct`) — doit être pris en charge en `/v1/chat/completions`                                                                                                                                                                |
| `OPENAI_CHAT_MODEL`                                                                   | Modèle OpenAI pour le chat (défaut `gpt-4o-mini`)                                                                                                                                                                                                                                       |
| `RAG_TOP_K` / `CHUNK_SIZE`                                                            | Récupération RAG et découpe PDF (1–20 / 128–2048, multiple de 128) — aussi éditables dans l’UI Paramètres                                                                                                                                                                               |
| `HF_ROUTER_BASE_URL`                                                                  | Défaut : `https://router.huggingface.co/v1` (API compatible OpenAI)                                                                                                                                                                                                                     |
| `HF_INFERENCE_ENDPOINT`                                                               | Défaut géré dans `src/config.py` : `https://router.huggingface.co` (remplace l’ancien `api-inference.huggingface.co`, **410 Gone**)                                                                                                                                                     |
| `FLASK_PORT` / `FLASK_HOST`                                                           | Port et hôte du serveur de développement                                                                                                                                                                                                                                                |
| `DATASET_PDF_MAX_PAIRS`                                                               | Plafond par défaut pour **`POST /api/dataset/generate`** (évite les timeouts navigateur ; la CLI `dataset` peut aller jusqu’à 300)                                                                                                                                                      |
| `BENCHMARK_HF_MODEL_QWEN` / `BENCHMARK_HF_MODEL_MISTRAL` / `BENCHMARK_HF_MODEL_LLAMA` | IDs Hub pour **qwen-7b**, **mistral-7b**, **llama3-8b** (routeur `/v1/chat`). Défaut **mistral-7b** = **Qwen2.5-7B** (comme qwen-7b) si le **3B** n’est servi par aucun _Inference Provider_ activé. Défaut **llama** = `meta-llama/Llama-3.1-8B-Instruct` (ID reconnu par le routeur). |

Le choix du LLM côté code passe par **`src/llm_factory.py`**.

---

## Utilisation

### Interface Web

1. Ouvrir **`http://127.0.0.1:<FLASK_PORT>`** (souvent **5001** en local si défini dans `.env`).
2. Onglet **Documents** : déposer ou uploader des PDFs, puis **Indexer** (construit / met à jour Chroma sous `vectorstore/`).
3. Onglet **Chat** : choisir une pipeline et poser des questions (l’index est rechargé ou **réindexé automatiquement** si les PDFs ont changé — empreinte des fichiers dans `vectorstore/.pdf_sources_fingerprint`).
4. Onglet **Dataset** : **Générer depuis les PDFs** — le LLM crée des Q/R à partir des **chunks indexés** ; nombre de paires réglable (plafond côté API via `DATASET_PDF_MAX_PAIRS` ou corps JSON). Sans fichier JSON, l’API renvoie un dataset vide jusqu’à génération.
5. Onglet **Benchmark** : résultats comparatifs après exécution du benchmark.

### Kaggle — pipeline expérimental (0–8) et sauvegarde des résultats

Voir **[docs/KAGGLE_EXPERIMENTS.md](docs/KAGGLE_EXPERIMENTS.md)** : script `scripts/kaggle_full_experiment.py` (base documentaire, Q/R, benchmark LLM, RAG simple / avancé / agent / multi-agent, fine-tuning, RAFT), fichiers JSON dans `data/evaluation/`.

### Ligne de commande

```bash
# Étape 1 : Dataset (PDFs + LLM si chunks disponibles, sinon templates)
python run.py --step dataset

# Étape 2 : Benchmark LLMs
python run.py --step benchmark

# Étape 3 : RAG Simple
python run.py --step rag

# Étape 4 : RAG Avancé
python run.py --step rag-adv

# Étape 5 : Fine-Tuning
python run.py --step finetune

# Étape 6 : RAFT
python run.py --step raft

# Étape 7 : RAG + Agent
python run.py --step agent

# Étape 8 : Multi-Agent
python run.py --step multi-agent

# Serveur web Flask
python run.py --step serve

# Toutes les étapes (sans serveur)
python run.py --step all
```

---

## Étapes du projet

### Étape 1 — Génération du Dataset

Fichiers : `src/dataset_generator.py`, `src/index_fingerprint.py` (cohérence index / PDFs côté app).

- **Mode principal** : à partir des **chunks des PDFs** (`data/raw_pdfs/`), le LLM génère des paires question/réponse fidèles au passage (domaine métadonnée `documents`). Déclenché par **`POST /api/dataset/generate`** ou par **`python run.py --step dataset`** si des chunks existent.
- **Repli** : si aucun PDF n’est disponible, génération **locale par templates** (thème e-learning historique), pour le développement sans corpus.

Fichier de sortie : **`data/evaluation/dataset_evaluation.json`**.

### Étape 2 — Benchmark LLM

Fichier : `src/llm_benchmark.py`

Compare les modèles configurés avec des métriques type Accuracy, F1, BLEU, ROUGE-L, latence, etc.

### Étape 3 — RAG Simple

Fichier : `src/rag_simple.py`

Pipeline : Question → recherche vectorielle (Chroma/FAISS) → prompt augmenté → LLM → Réponse.

### Étape 4 — RAG Avancé

Fichier : `src/rag_advanced.py`

- **Reranking** (Cross-Encoder)
- **HyDE**
- **Recherche hybride** (dense + BM25, fusion RRF)
- **Multi-Query**

### Étape 5 — Fine-Tuning

Fichier : `src/fine_tuning.py`

LoRA, QLoRA, fine-tuning complet (ressources GPU selon la méthode).

### Étape 6 — RAFT

Fichier : `src/raft.py`

RAG + fine-tuning : oracles, distracteurs, raisonnement.

### Étape 7 — RAG + Agent IA

Fichier : `src/rag_agent.py`

Agent ReAct avec outils (base vectorielle, web DuckDuckGo, exécution de code selon configuration).

### Étape 8 — RAFT + Multi-Agent IA

Fichier : `src/raft_multi_agent.py`

Orchestration Chercheur / Analyste / Rédacteur avec superviseur.

---

## Déploiement Docker

### Commandes essentielles

```bash
cd docker
docker-compose up --build -d
docker-compose logs -f app
docker-compose down
docker-compose restart app
docker-compose down -v   # ⚠ supprime aussi les volumes
```

### Services

| Service  | Port                 | Description                             |
| -------- | -------------------- | --------------------------------------- |
| app      | 5000 (selon compose) | Application Flask                       |
| chromadb | 8000                 | ChromaDB (si utilisé en service séparé) |
| redis    | 6379                 | Cache (optionnel)                       |

---

## API Reference

| Endpoint                | Méthode | Description                                                                                                                                                                                                                         |
| ----------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/api/health`           | GET     | État de l’app ; champs utiles : `pdfs_available`, `index_persisted`, `vector_store_in_memory`                                                                                                                                       |
| `/api/settings`         | GET     | Paramètres effectifs (clés masquées, IDs benchmark, `RAG_TOP_K`, etc.)                                                                                                                                                              |
| `/api/settings`         | POST    | Enregistre dans `.env` et met à jour la config en mémoire ; corps JSON : `llm_provider`, `hf_chat_model_id`, `openai_chat_model`, `rag_top_k`, `chunk_size`, et optionnellement `openai_api_key` / `huggingface_token` si non vides |
| `/api/upload`           | POST    | Upload de PDFs                                                                                                                                                                                                                      |
| `/api/pdfs`             | GET     | Liste des PDFs dans `data/raw_pdfs/`                                                                                                                                                                                                |
| `/api/index`            | POST    | Indexation / réindexation complète (réinitialise le stockage vectoriel persisté pour ce déploiement)                                                                                                                                |
| `/api/query`            | POST    | Question + `pipeline` (`rag_simple`, `rag_advanced`, `rag_agent`, `multi_agent`)                                                                                                                                                    |
| `/api/dataset`          | GET     | Lit `dataset_evaluation.json` ou renvoie `data: []` si absent                                                                                                                                                                       |
| `/api/dataset/generate` | POST    | Génère le dataset depuis les **chunks PDF** + LLM ; corps JSON optionnel : `{"max_pairs": 60}`                                                                                                                                      |
| `/api/benchmark`        | GET     | Résultats du benchmark (`benchmark_results.json`)                                                                                                                                                                                   |
| `/api/benchmark/run`    | POST    | Lance le benchmark sur un échantillon du dataset ; JSON : `{"max_questions": 15}` (max 100)                                                                                                                                         |
| `/api/pipelines`        | GET     | Pipelines disponibles                                                                                                                                                                                                               |

### Exemples

```bash
PORT=5001
curl -s "http://127.0.0.1:${PORT}/api/health"

curl -X POST "http://127.0.0.1:${PORT}/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Qu est-ce que le RAG ?", "pipeline": "rag_simple"}'

curl -X POST "http://127.0.0.1:${PORT}/api/dataset/generate" \
  -H "Content-Type: application/json" \
  -d '{"max_pairs": 40}'
```

---

## Structure du projet

```
rag-llm-elearning-project/
├── app/
│   ├── app.py                  # Flask, routes API, sync index / dataset
│   ├── templates/index.html
│   └── static/                 # CSS & JavaScript
├── src/
│   ├── config.py               # Configuration centralisée
│   ├── llm_factory.py          # Instanciation OpenAI / routeur HF (chat)
│   ├── index_fingerprint.py    # Empreinte des PDFs → réindexation si changement
│   ├── dataset_generator.py    # Dataset (PDF + LLM ou templates)
│   ├── llm_benchmark.py
│   ├── pdf_processor.py
│   ├── vector_store.py         # Chroma / FAISS + clear_persisted_vectorstore
│   ├── rag_simple.py
│   ├── rag_advanced.py
│   ├── fine_tuning.py
│   ├── raft.py
│   ├── rag_agent.py
│   ├── raft_multi_agent.py
│   └── evaluator.py
├── data/
│   ├── raw_pdfs/               # PDFs source
│   ├── processed/
│   └── evaluation/             # dataset_evaluation.json, benchmark, etc.
├── vectorstore/                # Persistance Chroma (gitignore courant)
├── docker/
├── docs/
├── requirements.txt
├── .env.example
└── run.py
```

---

## PDFs recommandés pour la base de connaissances

### Finance

- Investopedia, BIS Papers, SSRN

### E-learning

- UNESCO ICT in Education, EdTech, MIT OpenCourseWare

### Médecine

- PubMed Central, WHO, arXiv (NLP / IA santé)

---

## Auteur

Projet réalisé dans le cadre du Master Intelligence Artificielle & Big Data.

## Licence

Ce projet est à usage académique.
