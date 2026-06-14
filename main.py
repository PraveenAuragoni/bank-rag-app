"""
Bank Policy RAG API
FastAPI + LangChain 1.3.2 + ChromaDB + Claude
"""

import os
import time
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from logger_config import setup_logging, get_logger
from models import (
    QuestionRequest, RAGResponse,
    UploadResponse, DocumentListResponse,
    HealthResponse, ChatRequest, ChatResponse
)
#from rag_engine import RAGEngine
from rag_engine_bedrock import RAGEngineBedrock

# ── Setup logging FIRST before anything else ─────────────
setup_logging(level="INFO")
logger = get_logger(__name__)

# ── App startup / shutdown ────────────────────────────────
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("=" * 60)
    logger.info("Bank Policy RAG API — Starting up")
    logger.info("=" * 60)
    engine = RAGEngineBedrock()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="Bank Policy RAG API",
    description="Ask questions about bank policy documents using AI",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"→ {request.method} {request.url.path} | ip={client_ip}")

    response = await call_next(request)
    elapsed = time.time() - start
    status = response.status_code

    if status >= 500:
        logger.error(f"← {status} {request.url.path} | {elapsed:.3f}s")
    elif status >= 400:
        logger.warning(f"← {status} {request.url.path} | {elapsed:.3f}s")
    else:
        logger.info(f"← {status} {request.url.path} | {elapsed:.3f}s")

    return response


# ════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def home():
    """Simple web UI to test the RAG system"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Bank Policy RAG</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f0f4ff;
               display: flex; justify-content: center; padding: 40px 20px; }
        .container { width: 100%; max-width: 720px; }
        h1 { color: #0C447C; margin-bottom: 6px; }
        p.sub { color: #666; margin-bottom: 24px; font-size: 14px; }
        textarea { width: 100%; padding: 12px; border: 1px solid #ccc;
                   border-radius: 8px; font-size: 15px; resize: vertical; }
        button { margin-top: 12px; padding: 10px 28px; background: #185FA5;
                 color: white; border: none; border-radius: 8px;
                 font-size: 15px; cursor: pointer; }
        button:hover { background: #0C447C; }
        .answer-box { margin-top: 20px; padding: 16px; background: white;
                      border-left: 4px solid #185FA5; border-radius: 8px;
                      display: none; }
        .answer-box h3 { color: #185FA5; margin-bottom: 8px; }
        .answer-box p  { line-height: 1.6; color: #333; }
        .sources { margin-top: 12px; font-size: 13px; color: #666; }
        .meta { display: flex; gap: 16px; margin-top: 10px; font-size: 12px; color: #999; }
        .loading { color: #185FA5; font-style: italic; }
        .upload-section { margin-top: 32px; padding: 16px; background: white;
                          border-radius: 8px; border: 1px dashed #ccc; }
        .upload-section h3 { color: #0C447C; margin-bottom: 12px; }
        input[type=file] { font-size: 14px; }
        .upload-btn { background: #0F6E56; margin-top: 10px; }
        .upload-btn:hover { background: #085041; }
        .msg { margin-top: 10px; font-size: 13px; padding: 8px;
               border-radius: 6px; display: none; }
        .msg.success { background: #E1F5EE; color: #0F6E56; display: block; }
        .msg.error   { background: #FAECE7; color: #993C1D; display: block; }
    </style>
</head>
<body>
<div class="container">
    <h1>Bank Policy RAG System</h1>
    <p class="sub">Ask any question about bank policies — powered by Claude AI</p>

    <textarea id="question" rows="3"
        placeholder="e.g. What is the minimum balance for savings account?"></textarea>
    <br>
    <button onclick="askQuestion()">Ask Question</button>

    <div class="answer-box" id="answerBox">
        <h3>Answer</h3>
        <p id="answerText"></p>
        <div class="sources" id="sourcesText"></div>
        <div class="meta">
            <span id="routedTo"></span>
            <span id="confidence"></span>
        </div>
    </div>

    <div class="upload-section">
        <h3>Upload New Policy Document</h3>
        <input type="file" id="pdfFile" accept=".pdf">
        <br>
        <button class="upload-btn" onclick="uploadPDF()">Upload PDF</button>
        <div class="msg" id="uploadMsg"></div>
    </div>
</div>

<script>
async function askQuestion() {
    const q   = document.getElementById('question').value.trim();
    const box = document.getElementById('answerBox');
    if (!q) return;
    document.getElementById('answerText').innerHTML = '<span class="loading">Thinking...</span>';
    box.style.display = 'block';
    document.getElementById('sourcesText').textContent = '';
    document.getElementById('routedTo').textContent    = '';
    document.getElementById('confidence').textContent  = '';
    try {
        const res  = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q })
        });
        const data = await res.json();
        document.getElementById('answerText').textContent  = data.answer;
        document.getElementById('sourcesText').textContent = 'Sources: ' + data.sources.join(' | ');
        document.getElementById('routedTo').textContent    = 'Searched: ' + data.routed_to;
        document.getElementById('confidence').textContent  = 'Confidence: ' + data.confidence;
    } catch(e) {
        document.getElementById('answerText').textContent = 'Error: ' + e.message;
    }
}
async function uploadPDF() {
    const file = document.getElementById('pdfFile').files[0];
    const msg  = document.getElementById('uploadMsg');
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    msg.className = 'msg'; msg.textContent = 'Uploading...'; msg.style.display = 'block';
    try {
        const res  = await fetch('/upload-pdf', { method: 'POST', body: form });
        const data = await res.json();
        msg.className   = 'msg success';
        msg.textContent = data.message;
    } catch(e) {
        msg.className   = 'msg error';
        msg.textContent = 'Upload failed: ' + e.message;
    }
}
document.getElementById('question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askQuestion(); }
});
</script>
</body>
</html>
"""


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check API health and document count"""
    logger.debug("Health check requested")
    return HealthResponse(
        status="healthy",
        docs_indexed=engine.get_doc_count(),
        version="1.0.0"
    )


@app.post("/ask", response_model=RAGResponse, tags=["RAG"])
def ask_question(request: QuestionRequest):
    """Ask a question about bank policies"""
    logger.info(f"New question: '{request.question[:80]}'")

    if not request.question.strip():
        logger.warning("Empty question received")
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > 2000:
        logger.warning(f"Question too long: {len(request.question)} chars")
        raise HTTPException(status_code=400, detail="Question too long. Max 2000 characters.")

    try:
        result = engine.answer(request.question, request.doc_filter)
        logger.info(
            f"Question answered | confidence={result['confidence']} | routed_to={result['routed_to']}"
        )
        return RAGResponse(**result)
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-pdf", response_model=UploadResponse, tags=["Documents"])
def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a new PDF policy document"""
    logger.info(f"PDF upload requested: {file.filename}")

    if not file.filename.endswith(".pdf"):
        logger.warning(f"Rejected non-PDF upload: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    if file.size and file.size > 50 * 1024 * 1024:
        logger.warning(f"Rejected oversized file: {file.filename} ({file.size} bytes)")
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

    save_path = f"bank_docs/{file.filename}"
    os.makedirs("bank_docs", exist_ok=True)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"PDF saved to disk: {save_path}")

        chunks_added = engine.index_pdf(save_path)
        logger.info(f"PDF indexed: {file.filename} | chunks={chunks_added}")

        return UploadResponse(
            message=f"{file.filename} uploaded and indexed successfully",
            filename=file.filename,
            chunks_indexed=chunks_added
        )
    except Exception as e:
        logger.error(f"PDF upload/index failed: {file.filename} | error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    """List all indexed PDF documents"""
    docs = []
    if os.path.exists("bank_docs"):
        docs = [f for f in os.listdir("bank_docs") if f.endswith(".pdf")]
    logger.debug(f"Documents listed: {docs}")
    return DocumentListResponse(documents=docs, total=len(docs))


@app.delete("/documents/{filename}", tags=["Documents"])
def delete_document(filename: str):
    """Delete a document from disk"""
    path = f"bank_docs/{filename}"
    if not os.path.exists(path):
        logger.warning(f"Delete failed — file not found: {filename}")
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(path)
    logger.info(f"Document deleted: {filename}")
    return {"message": f"{filename} deleted"}


@app.get("/collections", tags=["System"])
def list_collections():
    """Show ChromaDB collection info"""
    info = engine.get_collection_info()
    logger.debug(f"Collection info requested: {info}")
    return info


@app.get("/logs", tags=["System"])
def get_logs(lines: int = 50):
    """
    View last N lines of the log file.
    Usage: /logs?lines=100
    """
    log_file = "logs/app.log"
    if not os.path.exists(log_file):
        logger.warning("Log file not found when requested via /logs")
        return {"logs": [], "message": "No log file found yet"}

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        last_lines = all_lines[-lines:]
        logger.debug(f"Log tail requested: last {lines} lines")
        return {
            "total_lines": len(all_lines),
            "showing_last": lines,
            "logs": [line.rstrip() for line in last_lines]
        }
    except Exception as e:
        logger.error(f"Failed to read log file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
