# MediGraph-RAG

An agentic RAG chatbot for medical charts, built with FastAPI, LangGraph, and Ollama.

## What it does

- Upload patient charts as PDF
- Ask physician-style questions grounded in the chart
- Generate SOAP-style clinical summaries
- Detect medications and build patient timelines
- Runs fully locally using Ollama — no external API needed

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Agent framework | LangGraph |
| LLM runtime | Ollama (phi3:mini) |
| Embeddings | nomic-embed-text |
| Vector DB | ChromaDB |
| Frontend | React + Vite |
| Evaluation | RAGAS |
| Deployment | Docker |

## Quickstart

### Prerequisites

- Python 3.11+
- Ollama installed and running
- Docker (optional)

### 1. Clone the repo

git clone https://github.com/YOUR_USERNAME/MediGraph-RAG.git
cd MediGraph-RAG

### 2. Create virtual environment

python -m venv .venv
source .venv/Scripts/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Set up environment

Copy .env.example to .env and edit with your settings

### 5. Pull Ollama models

ollama pull phi3:mini
ollama pull nomic-embed-text

### 6. Run the app

make run

or

uvicorn app.main:app --reload

### 7. Open the API docs

Visit http://localhost:8000/docs

## Make commands

| Command | What it does |
|---|---|
| make run | Start FastAPI dev server |
| make docker-up | Start all services in Docker |
| make docker-down | Stop all Docker services |
| make test | Run pytest |
| make eval | Run RAGAS evaluation |

## Project structure

MediGraph-RAG/
├── app/
│   ├── api/
│   │   └── routes/          # FastAPI endpoints
│   ├── agents/              # LangGraph agents
│   ├── rag/                 # Ingest, retrieval, prompts
│   ├── models/              # Ollama client, response models
│   ├── tools/               # Timeline, medication tools
│   ├── memory/              # Session memory
│   ├── evaluation/          # RAGAS eval, hallucination checks
│   └── config/              # Settings, logging
│
├── frontend/
│   └── react-app/           # React + Vite UI
│
├── datasets/                # Sample charts, evaluation QA pairs
├── tests/                   # Pytest test suite
└── docs/                    # Architecture documentation

