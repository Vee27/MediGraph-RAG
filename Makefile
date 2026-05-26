.PHONY: run docker-up docker-down test lint

# Run the app locally (without Docker)
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start everything in Docker
docker-up:
	docker compose up --build

# Stop all Docker containers
docker-down:
	docker compose down

# Run tests
test:
	pytest tests/ -v

# Check code style
lint:
	ruff check app/
