.PHONY: run dev docker-up docker-down test lint eval

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

test:
	pytest tests/ -v

lint:
	ruff check app/ tests/

eval:
	python -m app.evaluation.benchmark