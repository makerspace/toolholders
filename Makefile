backend:
	uv run python3 -m backend.main

frontend:
	cd frontend && npm run build

.PHONY: backend frontend