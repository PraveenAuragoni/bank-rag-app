# Bank Policy RAG API

A production-grade RAG (Retrieval Augmented Generation) system for banking policy Q&A.
Built with **LangChain 1.3.2**, **ChromaDB**, **FastAPI**, and **Claude AI**.

---

## What It Does

- Upload any bank policy PDF
- Ask questions in plain English
- Get accurate answers with source references (document name + page number)
- Smart routing — searches the most relevant document type automatically

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Anthropic Claude (claude-opus-4-5) |
| Framework | LangChain 1.3.2 (LCEL) |
| Vector DB | ChromaDB (persisted on disk) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (free, local) |
| API | FastAPI |
| PDF Loading | LangChain PyPDFLoader |

---

## Project Structure

```
bank-rag-app/
├── main.py                  # FastAPI app — all endpoints
├── rag_engine.py            # RAG logic — index, search, answer
├── models.py                # Pydantic request/response models
├── create_sample_pdfs.py    # Creates test PDFs
├── requirements.txt
├── .env.example
├── .gitignore
├── bank_docs/               # Put your PDF files here
└── chroma_db/               # Auto-created — vector database
```

---

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/your-username/bank-rag-app.git
cd bank-rag-app
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

Get your free key at: https://console.anthropic.com

### 5. Create sample PDFs (optional)

```bash
python create_sample_pdfs.py
```

### 6. Run the server

```bash
uvicorn main:app --reload --port 8000
```

### 7. Open in browser

```
http://localhost:8000          # Web UI
http://localhost:8000/docs     # Interactive API docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Health check + doc count |
| POST | `/ask` | Ask a question |
| POST | `/upload-pdf` | Upload new PDF |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{filename}` | Delete a document |
| GET | `/collections` | ChromaDB info |

---

## Example API Usage

### Ask a question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the minimum balance for savings account?"}'
```

Response:
```json
{
  "question": "What is the minimum balance for savings account?",
  "answer": "The minimum balance is INR 10,000 for metro cities...",
  "sources": ["savings_account.pdf pg.0"],
  "routed_to": "savings_account",
  "confidence": "high"
}
```

### Upload a PDF

```bash
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@my_policy.pdf"
```

### Filter to specific document

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the rates?", "doc_filter": "fixed_deposit"}'
```

---

## Call from Java Spring Boot

```java
WebClient client = WebClient.create("http://localhost:8000");

Map<String, String> request = Map.of("question", "What is KYC?");

Map response = client.post()
    .uri("/ask")
    .bodyValue(request)
    .retrieve()
    .bodyToMono(Map.class)
    .block();

String answer = (String) response.get("answer");
```

---

## How RAG Works

```
User Question
     ↓
Smart Router (keyword match → LLM fallback)
     ↓
ChromaDB search (top 3 matching chunks)
     ↓
Format context: [filename pg.X] + content
     ↓
Claude answers using ONLY the context
     ↓
Return answer + sources + confidence
```

---

## Portfolio Notes

This project demonstrates:
- **RAG architecture** — production pattern for enterprise AI
- **LangChain 1.3.2 LCEL** — modern chain composition
- **Vector search** — semantic search via ChromaDB
- **Smart routing** — keyword + LLM hybrid routing
- **REST API** — FastAPI with validation and error handling
- **Java integration** — callable from Spring Boot microservices

---

## Author

Praveen — Senior Java/AWS Engineer | GenAI Applications
