# Guide d'Installation et de Déploiement

## 1. Prérequis

### Matériel minimum
- CPU : 4 cœurs
- RAM : 8 Go (16 Go recommandé)
- Stockage : 10 Go libres
- GPU : optionnel (requis uniquement pour le fine-tuning, étape 5)

### Logiciels
- Python 3.10 ou supérieur
- Docker + Docker Compose (pour le déploiement containerisé)
- Git

### Clés API
Au moins une des clés suivantes est nécessaire :
- **OpenAI API Key** : pour GPT-4o-mini (recommandé pour commencer)
- **HuggingFace Token** : pour Mistral 7B et LLaMA 3 (modèles open-source)

## 2. Installation locale (développement)

```bash
# Cloner le projet
git clone <url-du-repo>
cd rag-llm-project

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK (pour l'évaluation)
python -c "import nltk; nltk.download('punkt')"

# Configurer les variables d'environnement
cp .env.example .env
nano .env                       # Ajouter vos clés API
```

## 3. Préparation des données

### Ajout des PDFs
Placez vos fichiers PDF dans le dossier `data/raw_pdfs/`. Le système accepte
tout PDF lisible. Voici des sources recommandées par domaine :

**Finance :**
- Télécharger des rapports annuels depuis les sites des entreprises cotées
- Articles de recherche depuis SSRN (ssrn.com)
- Rapports de la Banque de France ou de la BCE

**E-learning :**
- Publications UNESCO sur l'éducation numérique
- Articles depuis Google Scholar sur "adaptive learning" ou "MOOC"
- Rapports EdTech de McKinsey ou Deloitte

**Médecine :**
- Articles en libre accès depuis PubMed Central (ncbi.nlm.nih.gov/pmc)
- Rapports techniques de l'OMS (who.int)
- Publications sur l'IA en santé depuis arXiv

### Génération du dataset d'évaluation
```bash
python run.py --step dataset
```
Cela crée `data/evaluation/dataset_evaluation.json` avec 300 paires Q/R.

## 4. Exécution du projet

### Mode séquentiel (toutes les étapes)
```bash
python run.py --step all
```

### Étape par étape
```bash
python run.py --step dataset      # Étape 1
python run.py --step benchmark    # Étape 2
python run.py --step rag          # Étape 3
python run.py --step rag-adv      # Étape 4
python run.py --step finetune     # Étape 5
python run.py --step raft         # Étape 6
python run.py --step agent        # Étape 7
python run.py --step multi-agent  # Étape 8
```

### Lancer l'interface web
```bash
python run.py --step serve
# Ouvrir http://localhost:5000
```

## 5. Déploiement Docker

### Construction et lancement
```bash
cd docker
docker-compose up --build -d
```

### Vérification
```bash
# Vérifier que les services tournent
docker-compose ps

# Tester l'API
curl http://localhost:5000/api/health

# Voir les logs
docker-compose logs -f app
```

### Arrêt
```bash
docker-compose down          # Arrêter les services
docker-compose down -v       # Arrêter + supprimer les volumes
```

## 6. Utilisation de l'interface web

### Workflow recommandé

1. **Uploader les PDFs** : onglet Documents → glisser-déposer vos PDFs
2. **Indexer** : cliquer le bouton "Indexer les documents"
3. **Choisir la pipeline** : sélecteur en haut du chat
4. **Poser des questions** : taper dans le champ de saisie

### Choix de la pipeline

| Vous voulez... | Pipeline recommandée |
|---------------|---------------------|
| Réponse rapide et simple | RAG Simple |
| Meilleure précision | RAG Avancé |
| Compléter avec le web | RAG + Agent IA |
| Réponse complète et vérifiée | RAFT + Multi-Agent |

## 7. Résolution de problèmes

### Erreur "Aucun PDF trouvé"
Vérifiez que vos PDFs sont bien dans `data/raw_pdfs/` et qu'ils sont lisibles.

### Erreur "API key invalid"
Vérifiez votre fichier `.env` et assurez-vous que les clés sont correctes.

### Erreur de mémoire lors du fine-tuning
Utilisez QLoRA au lieu de LoRA ou Full FT. QLoRA nécessite environ 4 Go de VRAM.

### Erreur Docker "port already in use"
```bash
# Trouver le processus qui utilise le port
lsof -i :5000
# Changer le port dans docker-compose.yml
```
