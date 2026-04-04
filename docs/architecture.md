# Documentation Technique — Architecture du Système

## 1. Vue d'ensemble

Le système RAG-LLM Multi-Agent est une architecture modulaire en 8 couches progressives,
chacune ajoutant des capacités supplémentaires au système de question-réponse.

## 2. Diagramme d'architecture

```
                        ┌──────────────────────┐
                        │   UTILISATEUR        │
                        │   (Interface Web)    │
                        └──────────┬───────────┘
                                   │ HTTP/REST
                        ┌──────────▼───────────┐
                        │   FLASK SERVER       │
                        │   (API Gateway)      │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    ROUTER DE PIPELINE       │
                    │  (Sélection de la stratégie) │
                    └──┬───────┬───────┬───────┬──┘
                       │       │       │       │
              ┌────────▼─┐ ┌──▼────┐ ┌▼─────┐ ┌▼──────────┐
              │RAG Simple│ │RAG    │ │Agent │ │Multi-Agent│
              │          │ │Avancé │ │ReAct │ │RAFT+Super │
              └────┬─────┘ └──┬────┘ └──┬───┘ └──┬────────┘
                   │          │         │         │
              ┌────▼──────────▼─────────▼─────────▼────┐
              │          VECTOR STORE                   │
              │    (ChromaDB / FAISS)                   │
              │    Embeddings: all-MiniLM-L6-v2         │
              └────────────────┬───────────────────────┘
                               │
              ┌────────────────▼───────────────────────┐
              │        PDF PROCESSOR                   │
              │   (PyPDF → Chunking → Nettoyage)       │
              └────────────────────────────────────────┘
```

## 3. Description des modules

### 3.1. PDF Processor (`pdf_processor.py`)

Responsable de l'ingestion des documents :
- Chargement via PyPDF (supporte les PDFs multi-pages)
- Nettoyage du texte (suppression headers/footers, normalisation)
- Découpage en chunks via RecursiveCharacterTextSplitter
- Paramètres : chunk_size=512, chunk_overlap=50
- Enrichissement des métadonnées (source, chunk_id, page)

### 3.2. Vector Store (`vector_store.py`)

Base de données vectorielle pour la recherche sémantique :
- **Embeddings** : sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Backends** : ChromaDB (par défaut) ou FAISS
- **Opérations** : création, chargement, recherche, ajout
- **Persistance** : sauvegarde sur disque dans `vectorstore/`

### 3.3. RAG Simple (`rag_simple.py`)

Pipeline basique en 3 étapes :
1. **Retrieval** : Top-K documents par similarité cosinus
2. **Augmentation** : Construction du prompt avec le contexte
3. **Generation** : Le LLM génère la réponse

### 3.4. RAG Avancé (`rag_advanced.py`)

4 optimisations au-dessus du RAG simple :

**a) Reranking (Cross-Encoder)**
- Modèle : ms-marco-MiniLM-L-6-v2
- Réordonne les résultats par pertinence fine
- Réduit le bruit dans le contexte du LLM

**b) HyDE (Hypothetical Document Embeddings)**
- Le LLM génère un document hypothétique à partir de la question
- Ce document est utilisé comme requête de recherche
- Améliore la recherche pour les questions complexes

**c) Recherche hybride (Dense + BM25)**
- Dense : recherche vectorielle classique
- Sparse : BM25 (correspondance lexicale)
- Fusion par Reciprocal Rank Fusion (RRF)

**d) Multi-Query Retriever**
- Reformule la question en 3 variantes
- Recherche pour chaque variante
- Dédoublonne et combine les résultats

### 3.5. Fine-Tuning (`fine_tuning.py`)

3 méthodes de fine-tuning sur un LLM open-source :

| Méthode | Paramètres entraînés | Mémoire GPU | Qualité |
|---------|---------------------|-------------|---------|
| LoRA | ~0.1% (rang 16) | ~8 GB | Bonne |
| QLoRA | ~0.1% + quant 4-bit | ~4 GB | Bonne |
| Full | 100% | ~32 GB+ | Meilleure |

Framework : HuggingFace PEFT + TRL (SFTTrainer).

### 3.6. RAFT (`raft.py`)

Retrieval Augmented Fine-Tuning — combine RAG et FT :
- Construction d'un dataset spécialisé avec documents oracle et distracteurs
- Le modèle apprend à distinguer les bons documents des mauvais
- Chain-of-Thought intégré dans les réponses d'entraînement
- Probabilité oracle = 80% (paramétrable)

### 3.7. Agent IA (`rag_agent.py`)

Agent ReAct (Reasoning + Acting) avec outils :
- `base_connaissances` : recherche dans le VectorStore
- `recherche_web` : DuckDuckGo Search
- `calculateur_python` : exécution de code Python
- Boucle : Thought → Action → Observation → ... → Final Answer
- Max 10 itérations

### 3.8. Multi-Agent (`raft_multi_agent.py`)

3 agents spécialisés + superviseur :
- **Chercheur** : RAG + recherche web automatique si confiance < 70%
- **Analyste** : vérification de cohérence, identification des lacunes
- **Rédacteur** : synthèse finale structurée avec sources
- **Superviseur** : orchestre le workflow, peut relancer une recherche

## 4. Flux de données

```
PDFs → pdf_processor → chunks → vector_store → [RAG|Agent|Multi-Agent] → réponse
                                     ↑
                                embeddings
                          (all-MiniLM-L6-v2)
```

## 5. Technologies utilisées

| Catégorie | Technologie | Rôle |
|-----------|-------------|------|
| Orchestration | LangChain | Chaînes RAG, agents, prompts |
| Embeddings | Sentence-Transformers | Vectorisation sémantique |
| Vector DB | ChromaDB / FAISS | Stockage et recherche vectorielle |
| LLM | OpenAI / Mistral / LLaMA | Génération de texte |
| Fine-Tuning | PEFT + TRL | Adaptation au domaine |
| Reranking | Cross-Encoder | Réordonnancement des résultats |
| Web | Flask | Interface utilisateur |
| Déploiement | Docker + Compose | Containerisation |
| Recherche | DuckDuckGo | Recherche web complémentaire |

## 6. Métriques d'évaluation

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| Accuracy | % de réponses correctes | > 70% = bon |
| F1-Score | Harmonie précision/rappel | > 0.5 = bon |
| BLEU | Correspondance n-grams | > 0.3 = bon |
| ROUGE-L | Plus longue sous-séquence commune | > 0.4 = bon |
| Latence | Temps de réponse en ms | < 3000ms = bon |
