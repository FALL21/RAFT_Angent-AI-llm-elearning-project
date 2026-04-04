# ════════════════════════════════════════════════════════════
# RAG-LLM Multi-Agent — Makefile
# ════════════════════════════════════════════════════════════

.PHONY: help install run serve docker-up docker-down clean dataset benchmark

help: ## Afficher l'aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Installer les dépendances
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt')"
	cp -n .env.example .env || true

dataset: ## Générer le dataset d'évaluation (300 Q/R)
	python run.py --step dataset

benchmark: ## Benchmarker les 3 LLMs
	python run.py --step benchmark

rag: ## Lancer le RAG Simple
	python run.py --step rag

agent: ## Lancer l'Agent RAG
	python run.py --step agent

serve: ## Lancer l'interface web Flask
	python run.py --step serve

all: ## Exécuter toutes les étapes
	python run.py --step all

download-pdfs: ## Télécharger les PDFs de démonstration
	python scripts/download_pdfs.py

docker-up: ## Lancer avec Docker Compose
	cd docker && docker-compose up --build -d

docker-down: ## Arrêter Docker Compose
	cd docker && docker-compose down

docker-logs: ## Voir les logs Docker
	cd docker && docker-compose logs -f app

clean: ## Nettoyer les fichiers temporaires
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf vectorstore/* models/* data/processed/*
