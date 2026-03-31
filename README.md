# Enterprise RAG-Powered AI Assistant

A production-grade Retrieval Augmented Generation (RAG) system that answers questions **ONLY** from uploaded internal documents while minimizing hallucinations.

![RAG Architecture](https://img.shields.io/badge/Architecture-RAG-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![React](https://img.shields.io/badge/React-18-blue)
![Gemini](https://img.shields.io/badge/LLM-Gemini_1.5_Pro-orange)

## 🎯 Key Features

- **Document Upload**: Support for PDF and TXT files
- **Strict RAG**: Answers ONLY from uploaded documents
- **Hallucination Prevention**: Explicit prompting forces "I don't know" responses
- **Confidence Scores**: Every answer includes reliability indicators
- **Source Citations**: Trace every answer back to source documents
- **Feedback Loop**: Collect user feedback for quality monitoring
- **Enterprise-Ready**: Production-quality code with proper error handling

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE (React)                         │
│  • Chat Interface  • File Upload  • Confidence Display  • Sources   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                                │
├─────────────────────────────────────────────────────────────────────┤
│  INGESTION       │  RETRIEVAL        │  REASONING                   │
│  • PDF Parser    │  • FAISS Search   │  • Strict Prompts            │
│  • Chunker       │  • MMR Diversity  │  • Gemini LLM                │
│  • Metadata      │  • Score Filter   │  • Confidence Calc           │
├─────────────────────────────────────────────────────────────────────┤
│  EMBEDDINGS (Gemini text-embedding-004)  │  FEEDBACK STORAGE        │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Google Cloud API Key (for Gemini)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Run the server
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📁 Project Structure

```
enterprise-rag-assistant/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes and schemas
│   │   ├── core/
│   │   │   ├── ingestion/    # Document parsing and chunking
│   │   │   ├── embeddings/   # Gemini embeddings + FAISS
│   │   │   ├── retrieval/    # Similarity search + MMR
│   │   │   ├── reasoning/    # LLM client + prompts
│   │   │   └── feedback/     # User feedback storage
│   │   └── utils/            # Logging, exceptions
│   ├── data/                 # Persistent storage
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── components/       # React UI components
    │   ├── services/         # API client
    │   └── styles/           # Tailwind CSS
    └── package.json
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload and index a document |
| `/upload/documents` | GET | List indexed documents |
| `/upload/documents/{id}` | DELETE | Remove a document |
| `/ask` | POST | Ask a question |
| `/feedback` | POST | Submit feedback |
| `/feedback/stats` | GET | Get feedback statistics |
| `/health` | GET | Health check |

## 🛡️ Hallucination Prevention

This system uses multiple strategies to prevent hallucinations:

### 1. Strict Prompting
```
You may ONLY use information explicitly stated in the CONTEXT section.
Do NOT use your training knowledge, even if you're confident it's correct.
If the answer is NOT in the context, say "I don't have enough information."
```

### 2. Retrieval Grounding
- Every answer is generated from retrieved document chunks
- Source citations are required for every factual claim
- Low-relevance chunks are filtered out

### 3. Confidence Scoring
- Combines retrieval scores, chunk agreement, and coverage
- Low confidence answers are clearly flagged
- Users can assess reliability before acting

### 4. User Feedback
- Collect feedback on answer quality
- Identify problem areas without retraining
- Continuous quality monitoring

## 📊 Configuration

Key settings in `.env`:

```env
# LLM Settings
GEMINI_MODEL=gemini-1.5-pro
TEMPERATURE=0.1           # Low for factual responses

# Retrieval Settings
TOP_K_RESULTS=5           # Chunks per query
SIMILARITY_THRESHOLD=0.7  # Minimum relevance
MMR_DIVERSITY_SCORE=0.3   # Balance relevance/diversity

# Chunking Settings
CHUNK_SIZE=750            # ~700-800 tokens
CHUNK_OVERLAP=100         # Context continuity
```

## 🧪 Testing

```bash
cd backend
pytest tests/
```

## 📖 Interview-Ready Explanations

### Why FAISS over cloud vector databases?
- **No external dependencies**: Runs locally, no network latency
- **Data sovereignty**: Enterprise data stays on-premises
- **Cost-effective**: No per-query pricing
- **Fast**: Sub-millisecond search on millions of vectors

### Why chunking with overlap?
- **Embedding limits**: Models have token limits (512-8192)
- **Retrieval precision**: Small chunks allow targeted retrieval
- **Context continuity**: Overlap prevents information loss at boundaries

### Why MMR over pure similarity?
- **Diversity**: Avoids redundant results from same section
- **Coverage**: Better for multi-aspect questions
- **Quality**: More perspectives in the context

### Why strict prompting even with RAG?
- **LLMs can ignore context**: Training knowledge bleeds through
- **Gap filling**: Models synthesize plausible but wrong info
- **Explicit constraints**: Clear rules improve compliance

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

Built with ❤️ for enterprise AI applications
