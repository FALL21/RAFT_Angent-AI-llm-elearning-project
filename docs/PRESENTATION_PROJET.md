# Présentation complète du projet

## RAG–LLM Multi-Agent (e-learning & documents)

**Contexte :** projet Master Intelligence Artificielle & Big Data (NLP / systèmes conversationnels sur corpus documentaire).

Ce document peut servir de **support de soutenance**, de **synthèse exécutive** ou de **plan de diapositives**. Le détail technique (pipelines, schémas Mermaid, résultats chiffrés) est approfondi dans les autres fichiers du dossier `docs/` et dans le `README.md` à la racine.

---

## Table des matières

1. [Contexte et problématique](#1-contexte-et-problématique)
2. [Objectifs du projet](#2-objectifs-du-projet)
3. [Solution proposée en une phrase](#3-solution-proposée-en-une-phrase)
4. [Périmètre fonctionnel](#4-périmètre-fonctionnel)
5. [Stack technique](#5-stack-technique)
6. [Architecture générale](#6-architecture-générale)
7. [Chaîne de traitement des données](#7-chaîne-de-traitement-des-données)
8. [Huit étapes du pipeline (CLI)](#8-huit-étapes-du-pipeline-cli)
9. [Quatre modes de question-réponse (interface)](#9-quatre-modes-de-question-réponse-interface)
10. [Interface web et API](#10-interface-web-et-api)
11. [Résultats obtenus (instantané dépôt)](#11-résultats-obtenus-instantané-dépôt)
12. [Installation et déploiement](#12-installation-et-déploiement)
13. [Documentation associée](#13-documentation-associée)
14. [Limites et perspectives](#14-limites-et-perspectives)

---

## 1. Contexte et problématique

Les modèles de langage **ne connaissent pas** les documents privés ou récents (cours, rapports, PDFs de formation). Les **hallucinations** et les réponses hors sujet sont fréquentes si l’on s’en remet au seul modèle pré-entraîné.

**Problématique :** comment permettre à un utilisateur d’**interroger un corpus PDF** (par exemple du matériel pédagogique M2 / e-learning) avec des réponses **ancrées dans les sources**, tout en explorant des stratégies avancées (retrieval enrichi, agents, orchestration multi-agents) et une **évaluation comparative** des modèles ?

---

## 2. Objectifs du projet

| Objectif                        | Réalisation dans le dépôt                                                               |
| ------------------------------- | --------------------------------------------------------------------------------------- |
| Ingérer et indexer des PDFs     | Découpage, embeddings, Chroma/FAISS persistant                                          |
| Répondre par RAG                | Quatre pipelines au choix dans le chat                                                  |
| Améliorer le retrieval          | RAG avancé : HyDE, multi-query, BM25 + fusion RRF, reranking                            |
| Aller au-delà du seul corpus    | Agent ReAct (web DuckDuckGo, calcul Python) ; multi-agent avec complément web           |
| Évaluer les LLMs                | Dataset Q/R + benchmark (accuracy, F1, BLEU, ROUGE-L, latence)                          |
| Préparer l’adaptation du modèle | Fine-tuning (LoRA/QLoRA) ; construction dataset **RAFT** (oracles / distracteurs)       |
| Rendre le système utilisable    | Interface Flask (upload, indexation, chat, dataset, benchmark) ; Docker optionnel       |
| Industrialiser la config        | `.env`, API `/api/settings`, alignement automatique du chat sur le gagnant du benchmark |

---

## 3. Solution proposée en une phrase

Un **système modulaire** qui transforme des **PDFs** en **base vectorielle**, expose plusieurs **stratégies RAG / agents** via une **API REST** et une **interface web**, et intègre un **cycle d’évaluation** (dataset + benchmark) pour comparer et configurer les LLM.

---

## 4. Périmètre fonctionnel

- **Entrées :** fichiers PDF dans `data/raw_pdfs/` (upload possible depuis l’UI).
- **Sorties :** réponses textuelles avec traçabilité des sources (selon pipeline), métriques de benchmark, fichiers JSON d’évaluation.
- **Domaines cibles :** initialement orienté **e-learning** en mode template ; en production sur ce dépôt, le dataset d’évaluation peut être généré depuis les **chunks PDF** (domaine métadonnée `documents`).
- **Utilisateurs visés :** étudiant / chercheur / enseignant testant des corpus pédagogiques ; développeur étendant les pipelines.

---

## 5. Stack technique

| Couche            | Technologies                                                                       |
| ----------------- | ---------------------------------------------------------------------------------- |
| Langage           | Python 3.11 / 3.12 recommandé                                                      |
| Framework web     | Flask                                                                              |
| Orchestration NLP | LangChain (prompts, retrievers, agents ReAct)                                      |
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)                            |
| Vector store      | ChromaDB (défaut) ou FAISS                                                         |
| Retrieval avancé  | `rank_bm25`, Cross-Encoder (MS MARCO)                                              |
| LLM               | OpenAI (ex. `gpt-4o-mini`) et/ou **routeur Hugging Face** (`/v1/chat/completions`) |
| Agents            | Outils DuckDuckGo, exécution Python (agent)                                        |
| Conteneurs        | Docker Compose (dossier `docker/`)                                                 |

---

## 6. Architecture générale

Vue **macro** : l’utilisateur passe par l’**interface Flask**, qui route vers une **pipeline** ; toutes s’appuient sur le **même index vectoriel** (et sur la liste des **chunks** pour le mode avancé — BM25).

```
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACE FLASK                            │
│              (Upload PDFs · Chat · Dataset · Benchmark)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   ROUTEUR DE PIPELINE   │
              │  (rag_simple … multi)   │
              └────┬───────┬───────┬────┘
                   │       │       │
          ┌────────▼──┐ ┌──▼─────┐ ┌▼──────────┐
          │ RAG       │ │ RAG    │ │ Agent /   │
          │ simple    │ │ avancé │ │ Multi-ag. │
          └─────┬─────┘ └──┬─────┘ └─────┬─────┘
                │          │             │
                └──────────┼─────────────┘
                           │
                ┌──────────▼──────────┐
                │    VECTOR STORE     │
                │  (Chroma / FAISS)   │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │   PDF PROCESSOR   │
                │  (chunks + meta)    │
                └─────────────────────┘
```

Pour un **schéma Mermaid** et le détail par pipeline, voir [PIPELINES_ET_ARCHITECTURE.md](PIPELINES_ET_ARCHITECTURE.md).

**Multi-agents (vue métier)** : orchestration type Chercheur (RAG ± web) → Analyste → Rédacteur, avec boucles de retry si la confiance est faible (détail dans le même document technique).

---

## 7. Chaîne de traitement des données

1. **Ingestion** : lecture PDF (PyPDF), nettoyage, découpage récursif (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
2. **Métadonnées** : `chunk_id`, `source_file` pour citation et filtrage.
3. **Indexation** : embedding de chaque chunk, stockage persistant sous `vectorstore/`.
4. **Cohérence** : empreinte des fichiers sources (`index_fingerprint`) ; réindexation si les PDFs changent.
5. **Requête** : question utilisateur → retrieval (et éventuellement expansions / hybride / rerank) → prompt → LLM.

---

## 8. Huit étapes du pipeline (CLI)

Fichier d’orchestration : `run.py`.

| #   | Commande             | Rôle                                                                       |
| --- | -------------------- | -------------------------------------------------------------------------- |
| 1   | `--step dataset`     | Génère `dataset_evaluation.json` (PDF + LLM ou templates)                  |
| 2   | `--step benchmark`   | Compare les LLM, enregistre `benchmark_results.json`, peut aligner le chat |
| 3   | `--step rag`         | Test RAG simple en console                                                 |
| 4   | `--step rag-adv`     | Test RAG avancé en console                                                 |
| 5   | `--step finetune`    | Préparation fine-tuning LoRA/QLoRA                                         |
| 6   | `--step raft`        | Construction d’un dataset **RAFT** (oracles / distracteurs)                |
| 7   | `--step agent`       | Test agent ReAct en console                                                |
| 8   | `--step multi-agent` | Test superviseur multi-agents en console                                   |
| —   | `--step serve`       | Lance l’application web                                                    |
| —   | `--step all`         | Enchaîne les étapes 1–8 **sans** le serveur                                |

---

## 9. Quatre modes de question-réponse (interface)

Valeur du champ JSON `pipeline` pour `POST /api/query` :

| Mode            | Idée clé                                   | Coût / complexité typique                 |
| --------------- | ------------------------------------------ | ----------------------------------------- |
| **RAG simple**  | Top-k vectoriel → prompt → réponse         | Faible                                    |
| **RAG avancé**  | Multi-query, HyDE, BM25+RRF, reranking     | Plus d’appels LLM et calcul cross-encoder |
| **RAG + agent** | ReAct : outils base, web, Python           | Variable (boucle)                         |
| **Multi-agent** | Chercheur → Analyste → Rédacteur (+ retry) | Élevé (plusieurs passes LLM)              |

_Note :_ le libellé « RAFT + Multi-Agent » dans l’UI fait référence au **positionnement pédagogique** du projet ; à l’**inférence**, le module multi-agent est un **workflow supervisé** ; le **RAFT** au sens article (dataset avec distracteurs) correspond surtout à l’**étape 6** `raft`. Voir la clarification dans [PIPELINES_ET_ARCHITECTURE.md](PIPELINES_ET_ARCHITECTURE.md#64-multi_agent--orchestration-chercheur--analyste--rédacteur).

---

## 10. Interface web et API

### Fonctionnalités principales (UI)

- **Documents :** dépôt / liste des PDF, **indexation** (construction ou mise à jour Chroma).
- **Chat :** choix de pipeline, questions, affichage des sources (selon réponse API).
- **Paramètres :** provider LLM, modèles, `RAG_TOP_K`, taille de chunks (persistance `.env`).
- **Dataset :** génération de paires Q/R depuis les chunks (plafond configurable pour éviter les timeouts).
- **Benchmark :** lancement depuis l’UI, consultation des résultats.

### API REST (aperçu)

| Endpoint                                      | Rôle                             |
| --------------------------------------------- | -------------------------------- |
| `GET /api/health`                             | Santé de l’app, index en mémoire |
| `POST /api/query`                             | Question + `pipeline`            |
| `POST /api/index`                             | Réindexation                     |
| `GET/POST /api/settings`                      | Lecture / écriture configuration |
| `POST /api/upload`                            | Upload PDF                       |
| `GET/POST /api/dataset` · `POST .../generate` | Dataset évaluation               |
| `GET/POST /api/benchmark` · `POST .../run`    | Benchmark                        |
| `GET /api/pipelines`                          | Liste des pipelines              |

---

## 11. Résultats obtenus (instantané dépôt)

Les fichiers sous `data/evaluation/` sont des **instantanés** : ils évoluent après chaque régénération.

**Dataset (exemple enregistré) :** 22 questions, génération `pdf_chunks`, corpus issu de **10 PDFs** (thèmes Spark, NLP, deep learning, veille techno, projets).

**Benchmark LLM (exemple enregistré) :** trois modèles (`llama3-8b`, `qwen-7b`, `mistral-7b`) ; **meilleur score composite** : **Llama 3 8B** ; latence la plus faible observée sur cet essai également côté Llama 3 8B. Les exactitudes strictes restent modestes sur 22 questions ; les métriques F1 / BLEU / ROUGE-L complètent la lecture.

---

## 12. Installation et déploiement

**Locale (résumé) :**

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # éditer les clés OpenAI / Hugging Face
# Placer des PDF dans data/raw_pdfs/
python run.py --step serve
```

**Docker :** `cd docker && docker-compose up --build` (ports selon `docker-compose`).

Détails matériels, NLTK, variantes : [guide_installation.md](guide_installation.md) et [README.md — Installation](../README.md#installation).

---

## 13. Documentation associée

| Document                                                     | Contenu                                                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| [README.md](../README.md)                                    | Installation, `.env`, utilisation, API, structure des dossiers                             |
| [PIPELINES_ET_ARCHITECTURE.md](PIPELINES_ET_ARCHITECTURE.md) | Architecture détaillée, schémas Mermaid par pipeline, résultats chiffrés, fichiers sources |
| [architecture.md](architecture.md)                           | Vue technique condensée (diagramme ASCII, modules)                                         |
| [guide_installation.md](guide_installation.md)               | Prérequis, déploiement, dépannage                                                          |

---

## 14. Limites et perspectives

- **Sécurité :** l’outil « calcul Python » de l’agent exécute du code ; à isoler ou désactiver en production ouverte sur Internet.
- **Quota / fournisseurs :** les appels au routeur Hugging Face peuvent être limités (HTTP 402) ; le code inclut des garde-fous pour limiter les boucles coûteuses.
- **Évaluation pipelines RAG :** comparaison systématique `rag_simple` vs `rag_advanced` vs agents peut s’appuyer sur `src/evaluator.py` (intégration manuelle ou future étape CLI).
- **Pistes :** augmentation du dataset, métriques sémantiques (embedding similarity), filtres par `source_file`, garde-fous sur la recherche web, déploiement HTTPS et secrets managés.

---

**Synthèse finale :** le projet démontre une **chaîne complète** « corpus PDF → index sémantique → réponses contextualisées », avec **plusieurs niveaux de sophistication** (RAG classique à multi-agents) et un **cadre d’évaluation** pour les LLM, livré avec une **interface opérationnelle** et une documentation technique structurée.
