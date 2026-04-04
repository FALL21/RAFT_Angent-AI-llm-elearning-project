# Expériences sur Kaggle (étapes 0–8)

Ce flux correspond à ton plan : base documentaire → Q/R → benchmarks LLM et RAG → fine-tuning → RAFT → agents. Tous les **résultats** sont écrits sous `data/evaluation/` pour téléchargement vers ton poste.

## Commande unique

À la racine du projet (après `git pull`, PDFs dans `data/raw_pdfs/`) :

- **Secrets Kaggle** : créez un secret nommé exactement **`HUGGINGFACE_TOKEN`** (ou **`HF_TOKEN`**) dans *Add-ons → Secrets*. Le script `kaggle_full_experiment.py` les lit automatiquement au démarrage (même avec `!python`, qui ne voit pas les variables d’une autre cellule Python).
- Sinon : `export HUGGINGFACE_TOKEN=...` dans le même shell que celui qui lance Python.

```bash
export KAGGLE_MAX_QUESTIONS_PER_EVAL=15
# Optionnel : entraîner (long, GPU)
# export FINETUNE_RUN=1
# export FINETUNE_METHOD=qlora

python scripts/kaggle_full_experiment.py
```

Régénérer le dataset Q/R (LLM) même si le JSON existe déjà :

```bash
python scripts/kaggle_full_experiment.py --regen-dataset
# ou : export KAGGLE_DATASET_REGEN=1
```

## Fichiers produits

| Étape | Fichier(s) |
|-------|------------|
| 0 — Base de connaissances | `kaggle_step00_knowledge_base.json` |
| 1 — Q/R depuis PDFs | `kaggle_step01_qa_dataset.json`, `dataset_evaluation.json` |
| 2 — LLM seuls | `benchmark_results.json`, `kaggle_step02_llm_only.json` |
| 3 — RAG simple | `kaggle_step03_rag_simple.json` |
| 4 — RAG optimisé | `kaggle_step04_rag_advanced.json` |
| 5 — Fine-tuning | `kaggle_step05_finetune.json`, `kaggle_step05a_finetune_chat_format.jsonl`, évent. `finetune_manifest.json` + dossier `models/` |
| 6 — RAFT | `kaggle_step06_raft.json`, `raft_dataset.jsonl` |
| 7 — RAG + agent | `kaggle_step07_rag_agent.json` |
| 8 — Multi-agent | `kaggle_step08_rag_multi_agent.json` |
| Comparaison RAG | `kaggle_rag_family_report.json` |
| **Index** | `kaggle_experiment_index.json` (liste de tout ce qui a été écrit) |

## Sauter des étapes (quota / temps)

Exemple : ne faire que corpus + dataset + benchmark LLM :

```bash
export KAGGLE_SKIP_STEPS=3,4,5,6,7,8
python scripts/kaggle_full_experiment.py
```

Indices : `0` … `8` comme dans le tableau ci-dessus (alignés sur `_STEP_NAMES` du script).

## Étape 6 « RAG + fine-tuning RAFT »

Le script **construit** `raft_dataset.jsonl` et peut lancer l’entraînement RAFT si :

```bash
export KAGGLE_RUN_RAFT_TRAIN=1
export RAFT_TRAIN_METHOD=qlora   # optionnel
```

Sinon, le JSON d’étape 6 indique que l’évaluation end-to-end « modèle RAFT + RAG » se fait en **local** en chargeant l’adaptateur et en injectant le contexte retrieval (hors API routeur).

## Reprise sur ta machine

1. Télécharger `data/evaluation/*.json`, `*.jsonl`, et si besoin tout `models/`.
2. Les placer au même chemin dans ton clone local.
3. Lire `finetune_manifest.json` pour recharger les adaptateurs PEFT.
