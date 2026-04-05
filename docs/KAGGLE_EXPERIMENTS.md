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

## Fine-tuning (entraînement QLoRA / LoRA) sur Kaggle

L’étape 5 **prépare** toujours le JSONL ; l’**entraînement** ne part que si `FINETUNE_RUN=1`.

### Prérequis

1. **GPU** : dans le notebook Kaggle, *Settings* → **Accelerator** → **GPU** (idéalement **T4** x2). Sans GPU, QLoRA/LoRA échouera ou sera extrêmement lent.
2. **Token Hugging Face** : secret **`HUGGINGFACE_TOKEN`** (téléchargement de `Qwen/Qwen2.5-7B-Instruct` et dépendances).
3. **Dataset** : fichier `data/evaluation/dataset_evaluation.json` présent (généré par l’étape 1 ou copié depuis ton dépôt / une sortie précédente).

### Lancer tout le pipeline avec entraînement

```bash
cd /kaggle/working/RAFT_Angent-AI-llm-elearning-project   # ou le nom de ton clone
export FINETUNE_RUN=1
export FINETUNE_METHOD=qlora    # défaut si omis ; alternatives : lora, full (très lourd)
# Optionnel : autre base (défaut = Qwen 2.5 7B Instruct)
# export FINETUNE_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
python scripts/kaggle_full_experiment.py
```

### Uniquement le fine-tuning (sans RAG / benchmark — économise quota API)

Si `dataset_evaluation.json` est déjà là :

```bash
export FINETUNE_RUN=1
export KAGGLE_FINETUNE_ONLY=1
python scripts/kaggle_full_experiment.py
```

Équivalent : `export KAGGLE_SKIP_STEPS=0,1,2,3,4,6,7,8` avec `FINETUNE_RUN=1`.

### Après l’entraînement

- Adaptateur : dossier `models/qlora/final/` (ou `models/lora/final/`).
- Métadonnées : `data/evaluation/finetune_manifest.json` et `kaggle_step05_finetune.json` (champ `training` si l’entraînement a tourné).
- **Télécharge** `models/` et les JSON depuis l’onglet *Output* ou le panneau fichiers pour les réutiliser en local.

### Si mémoire GPU insuffisante (OOM)

Réduire la charge dans le code (`src/config.py`, bloc `FINE_TUNING_CONFIG` → `training.batch_size`, `max_seq_length`) ou tester `FINETUNE_METHOD=lora` sur une machine avec plus de VRAM. Sur T4 16 Go, **QLoRA + Qwen2.5-7B** est en général le bon compromis.

### Plusieurs GPU (Kaggle T4×2) — erreur `cuda:0` / `cuda:1`

Avec `device_map="auto"`, le modèle peut être réparti sur deux cartes et la loss PEFT/QLoRA plante (`Expected all tensors to be on the same device`). Par défaut, `src/fine_tuning.py` charge tout sur **`cuda:0`** dès qu’il détecte plusieurs GPU. Pour revenir à l’ancien comportement : `FINETUNE_DEVICE_MAP=auto` (ou `FINETUNE_MULTI_GPU=1`).

Avec **T4×2** et un seul processus Python, le `Trainer` peut en plus enrouler le modèle en **`torch.nn.DataParallel`**, ce qui casse PEFT (ex. `RuntimeError: chunk expects at least a 1-dimensional tensor`). Le code force **`args._n_gpu = 1`** avant l’entraînement pour éviter ce chemin. Pour réactiver l’ancien comportement (déconseillé ici) : `FINETUNE_ALLOW_DATAPARALLEL=1`.

### `no kernel image` / bitsandbytes (QLoRA)

Si le chargement 4-bit échoue avec **`CUDA error: no kernel image is available for execution on the device`**, le paquet **bitsandbytes** livré avec l’image ne correspond pas au GPU / au CUDA du notebook.

1. **Mettre à jour bitsandbytes** dans une cellule **avant** l’entraînement :

```python
!pip install -U 'bitsandbytes>=0.45.0'
```

Puis redémarrer le kernel si Kaggle le demande, ou relancer la session.

2. **Sans QLoRA** : `FINETUNE_METHOD=lora` (modèle en FP16 — plus gourmand en VRAM ; réduire `batch_size` / `max_seq_length` dans `src/config.py` si OOM).

3. **Repli auto** : `FINETUNE_QLORA_FALLBACK_LORA=1` avec `FINETUNE_METHOD=qlora` : en cas d’échec 4-bit, le script enchaîne en **LoRA FP16** (même dataset, adaptateur sous `models/lora/final/`).

### Tesla P100 (sm_60) — incompatible avec PyTorch CUDA 12 (Kaggle)

Kaggle attribue parfois un **Tesla P100** (capability **6.0**). Les images récentes embarquent **PyTorch 2.x + CUDA 12**, qui ne ciblent que **sm_70+** (T4, V100, A100…). Tu verras des avertissements du type *« P100 … is not compatible with the current PyTorch installation »*, puis souvent **`named symbol not found`** dans bitsandbytes ou **`no kernel image`**.

**Ce n’est pas corrigible** par `pip install -U bitsandbytes` : le problème vient de PyTorch + GPU.

**À faire :**

- Relancer une session / un autre notebook en espérant un **GPU T4** (ou utiliser un service avec GPU Volta+).
- Ou entraîner **en local / ailleurs** avec un GPU récent.
- Option expérimentale : réinstaller un **PyTorch ancien + CUDA 11.x** incluant sm_60 (lourd, fragile sur Kaggle).

Le script `run_finetune_from_evaluation_dataset` **refuse** désormais ce cas (avant téléchargement lourd du modèle) avec un message explicite. Pour ignorer le garde-fou (échec probable) : `FINETUNE_ALLOW_SM60=1`.

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

### Quota Hugging Face (HTTP 402)

Si le routeur renvoie **402 Payment Required** (« monthly included credits depleted »), les appels LLM cessent d’être servis : le benchmark et le RAG **avancé** (plus d’appels par question) échouent vite, alors que le RAG **simple** peut encore partiellement réussir avant la coupure.

**Pistes :** crédits prépayés ou abonnement sur [hf.co](https://huggingface.co), réduire `KAGGLE_MAX_QUESTIONS_PER_EVAL`, étaler les étapes sur plusieurs jours, ou utiliser **OpenAI** (`OPENAI_API_KEY` + `LLM_PROVIDER=openai` si tu as du quota).

### Journaux pypdf (« Ignoring wrong pointing object »)

Certains PDF exportés ont une table des objets imparfaite ; **pypdf** émet des avertissements mais l’extraction continue. Au démarrage, `kaggle_full_experiment.py` remonte le niveau des loggers `pypdf` / `pypdf._reader` à **ERROR** pour limiter le bruit dans la sortie Kaggle.

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
