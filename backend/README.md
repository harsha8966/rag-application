#  RAG Backend

FastAPI backend for the Enterprise RAG Assistant.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

```
app/
├── main.py              # FastAPI application entry
├── config.py            # Configuration management
│
├── api/
│   ├── routes/          # API endpoints
│   │   ├── upload.py    # Document upload
│   │   ├── ask.py       # Question answering
│   │   └── feedback.py  # User feedback
│   └── schemas/         # Pydantic models
│
├── core/
│   ├── ingestion/       # Document processing
│   │   ├── parser.py    # PDF/TXT extraction
│   │   ├── chunker.py   # Text splitting
│   │   └── metadata.py  # Metadata management
│   │
│   ├── embeddings/      # Vector operations
│   │   ├── gemini_embeddings.py
│   │   └── vector_store.py
│   │
│   ├── retrieval/       # Search strategies
│   │   ├── retriever.py
│   │   └── reranker.py
│   │
│   ├── reasoning/       # LLM interaction
│   │   ├── prompt_templates.py
│   │   ├── llm_client.py
│   │   └── confidence.py
│   │
│   └── feedback/        # Feedback storage
│       └── feedback_store.py
│
└── utils/
    ├── exceptions.py    # Custom exceptions
    └── logging.py       # Structured logging
```

## Key Components

### Document Ingestion
- Parses PDF and TXT files
- Chunks text using RecursiveCharacterTextSplitter
- Preserves metadata for citations

### Vector Store
- FAISS for similarity search
- Gemini embeddings (768 dimensions)
- Supports MMR for diverse retrieval

### LLM Integration
- Gemini 1.5 Pro for answer generation
- Strict prompts prevent hallucination
- Confidence scoring based on retrieval quality

### Feedback Loop
- JSON-based storage
- Tracks questions, answers, and user ratings
- Enables quality monitoring without retraining

## Testing

```bash
pytest tests/ -v
```

## Environment Variables

See `.env.example` for all configuration options.
