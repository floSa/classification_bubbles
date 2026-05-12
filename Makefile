# Bubble Project — raccourcis d'orchestration
# Voir README.md pour le détail des deux modes.

.DEFAULT_GOAL := help

.PHONY: help train run stop clean logs status

help:  ## Affiche cette aide
	@echo "Bubble Project — commandes disponibles :"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""

train:  ## Mode 1 : reset complet (volumes + modèle) + génération + entraînement
	@echo "♻️  Reset complet des volumes et du modèle..."
	docker compose --profile training down -v
	rm -f models/*.pth models/*.json
	@echo "🚀 Démarrage avec génération + training..."
	docker compose --profile training up -d --build
	@echo ""
	@echo "✅ Services lancés. Suivi de la progression :"
	@echo "   - Dashboard : http://localhost:8501"
	@echo "   - Logs training : make logs"

run:  ## Mode 2 : monitoring uniquement (nécessite données + modèle déjà présents)
	docker compose up -d --build
	@echo ""
	@echo "✅ Services lancés (sans training)."
	@echo "   - Dashboard : http://localhost:8501"
	@echo "   Si le dashboard affiche 'Configuration initiale requise',"
	@echo "   lance d'abord 'make train' pour générer les données et entraîner."

stop:  ## Arrête tous les conteneurs (préserve les volumes et le modèle)
	docker compose --profile training down

clean:  ## Suppression complète : volumes + modèle + fichiers de progression
	docker compose --profile training down -v
	rm -f models/*.pth models/*.json

logs:  ## Suit les logs du service de training en direct
	docker logs -f bubble_training

status:  ## État des conteneurs et résumé rapide
	@docker compose ps
	@echo ""
	@echo "=== Modèle ==="
	@ls -lh models/*.pth 2>/dev/null || echo "  Pas de modèle entraîné."
	@echo ""
	@echo "=== Progression training ==="
	@cat models/training_progress.json 2>/dev/null || echo "  Pas de fichier de progression."
