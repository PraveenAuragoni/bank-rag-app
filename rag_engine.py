"""
RAG Engine
Core logic: indexing PDFs, searching ChromaDB, answering with Claude
LangChain 1.3.2 — LCEL approach, no keyword routing, no hardcoding
Dynamic chunk size — auto-adapts to every PDF uploaded
Production-safe reindex — blue/green, zero downtime
"""

import os
import time
import threading
from pathlib import Path
from collections import Counter
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from logger_config import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Reindex job state (shared across threads) ─────────────
_reindex_status = {
    "running": False,
    "mode": None,  # "simple" | "bluegreen"
    "started_at": None,
    "finished_at": None,
    "progress": [],
    "result": None,
    "error": None,
}
_reindex_lock = threading.Lock()


class RAGEngine:

    def __init__(self):
        logger.info("Initialising RAG Engine...")

        # ── Embeddings ────────────────────────────────────
        logger.info("Loading embedding model (first run downloads ~80MB)...")
        t0 = time.time()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info(f"Embedding model loaded in {time.time() - t0:.2f}s")

        # ── LLM ───────────────────────────────────────────
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5",
            temperature=0,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        logger.info("ChatAnthropic LLM configured")

        # ── Prompt ────────────────────────────────────────
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional banking policy expert.
Answer the question using ONLY the context provided below.
If partial information exists in the context, use it to answer —
do NOT say information is unavailable if related content exists.
If truly not found, say:
'This information is not available in the current documents.'
Always mention the source document name and page number in your answer.

Context:
{context}"""),
            ("human", "{input}")
        ])

        # ── ChromaDB — load active collection ─────────────
        os.makedirs("./chroma_db", exist_ok=True)
        self.active_collection = self._load_active_collection()
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings,
            collection_name=self.active_collection
        )
        logger.info(
            f"ChromaDB ready — collection={self.active_collection} "
            f"| chunks={self.get_doc_count()}"
        )

        # ── Auto-index PDFs already in bank_docs/ ─────────
        self._index_existing_pdfs()
        logger.info("RAG Engine ready")

    # ═══════════════════════════════════════════════════════
    # ACTIVE COLLECTION PERSISTENCE
    # ═══════════════════════════════════════════════════════

    def _load_active_collection(self):
        """Load active collection name — survives restarts."""
        config = "./chroma_db/active_collection.txt"
        if os.path.exists(config):
            name = open(config).read().strip()
            logger.info(f"Loaded active collection: {name}")
            return name
        default = "bank_policies_v1"
        logger.info(f"No config found — using default: {default}")
        return default

    def _save_active_collection(self, name):
        """Persist active collection name to disk."""
        os.makedirs("./chroma_db", exist_ok=True)
        with open("./chroma_db/active_collection.txt", "w") as f:
            f.write(name)
        logger.info(f"Active collection saved: {name}")

    # ═══════════════════════════════════════════════════════
    # DYNAMIC CHUNK SIZE — auto-adapts per PDF
    # ═══════════════════════════════════════════════════════

    def _calculate_chunk_settings(self, pages, filename):
        """
        Auto-calculate optimal chunk size and overlap per PDF.
        Based on: page count, avg page length, and content type.
        No manual config — adapts automatically on every upload.

        Returns: (chunk_size, overlap, doc_class)
        """
        total_pages = len(pages)
        total_chars = sum(len(p.page_content) for p in pages)
        avg_page_len = total_chars / total_pages if total_pages > 0 else 0

        # ── Scan first 5 pages to detect document type ────
        sample_text = " ".join(
            p.page_content[:500] for p in pages[:5]
        ).lower()

        is_legal = any(kw in sample_text for kw in [
            "terms and conditions", "clause", "agreement",
            "pursuant", "hereinafter", "whereas", "shall be",
            "notwithstanding", "indemnify", "liable", "warranty"
        ])
        is_faq = any(kw in sample_text for kw in [
            "frequently asked", "q:", "question:", "faq",
            "how do i", "what is"
        ])
        is_table_heavy = any(kw in sample_text for kw in [
            "interest rate", "charges", "fee schedule",
            "tariff", "rate card", "per annum", "processing fee"
        ])

        logger.debug(
            f"PDF analysis: pages={total_pages} | avg_page_len={avg_page_len:.0f} "
            f"| legal={is_legal} | faq={is_faq} | table={is_table_heavy}"
        )

        # ── Decision matrix ────────────────────────────────
        if is_legal and total_pages > 50:
            chunk_size, overlap, doc_class = 2000, 400, "large_legal"
        elif is_legal and total_pages <= 50:
            chunk_size, overlap, doc_class = 1500, 300, "small_legal"
        elif is_faq:
            chunk_size, overlap, doc_class = 500, 100, "faq"
        elif is_table_heavy:
            chunk_size, overlap, doc_class = 1000, 200, "table_heavy"
        elif total_pages <= 5:
            chunk_size, overlap, doc_class = 500, 100, "small_doc"
        elif total_pages <= 20:
            chunk_size, overlap, doc_class = 1000, 200, "medium_doc"
        elif total_pages <= 50:
            chunk_size, overlap, doc_class = 1500, 300, "large_doc"
        else:
            chunk_size, overlap, doc_class = 2000, 400, "very_large_doc"

        # ── Fine-tune by avg page density ─────────────────
        if avg_page_len < 300:
            chunk_size = min(chunk_size, 500)
            overlap = 100
        elif avg_page_len > 3000:
            chunk_size = max(chunk_size, 1500)
            overlap = max(overlap, 300)

        logger.info(
            f"Chunk settings [{filename}]: doc_class={doc_class} "
            f"| chunk_size={chunk_size} | overlap={overlap} "
            f"| pages={total_pages} | avg_chars/page={avg_page_len:.0f}"
        )
        return chunk_size, overlap, doc_class

    # ═══════════════════════════════════════════════════════
    # PDF INDEXING
    # ═══════════════════════════════════════════════════════

    def _index_existing_pdfs(self):
        """Auto-index any PDFs in bank_docs/ not yet in ChromaDB."""
        folder = "bank_docs"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            logger.warning("bank_docs/ created — no PDFs yet.")
            return

        pdfs = list(Path(folder).glob("*.pdf"))
        if not pdfs:
            logger.warning("No PDFs in bank_docs/ — upload via /upload-pdf")
            return

        already_indexed = self._get_indexed_filenames()
        logger.info(f"PDFs: {len(pdfs)} | Indexed: {len(already_indexed)}")

        for pdf in pdfs:
            if pdf.name not in already_indexed:
                logger.info(f"New PDF — indexing: {pdf.name}")
                self.index_pdf(str(pdf))
            else:
                logger.debug(f"Already indexed, skipping: {pdf.name}")

    def index_pdf(self, pdf_path, target_vectorstore=None):
        """
        Index a PDF with auto-detected chunk settings.
        target_vectorstore: optional — used during blue/green reindex
                            to write into the NEW collection.
        Returns: stats dict
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        filename = os.path.basename(pdf_path)
        doc_type = Path(pdf_path).stem.lower().replace(" ", "_")
        store = target_vectorstore or self.vectorstore

        logger.info(f"Indexing: {filename} | doc_type={doc_type}")
        t0 = time.time()

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Auto chunk size
        chunk_size, overlap, doc_class = self._calculate_chunk_settings(
            pages, filename
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

        for page in pages:
            page.metadata["filename"] = filename
            page.metadata["doc_type"] = doc_type
            page.metadata["doc_class"] = doc_class
            page.metadata["chunk_size"] = chunk_size

        chunks = splitter.split_documents(pages)
        store.add_documents(chunks)

        elapsed = time.time() - t0
        stats = {
            "filename": filename,
            "doc_type": doc_type,
            "doc_class": doc_class,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "pages": len(pages),
            "chunks": len(chunks),
            "time_sec": round(elapsed, 2)
        }
        logger.info(
            f"Indexed  {filename} | class={doc_class} | chunk_size={chunk_size} "
            f"| pages={len(pages)} | chunks={len(chunks)} | {elapsed:.2f}s"
        )
        return stats

    # ═══════════════════════════════════════════════════════
    # REINDEX — SIMPLE (dev/staging) + BLUE/GREEN (production)
    # ═══════════════════════════════════════════════════════

    def reindex_simple(self):
        """
        DEV / STAGING — Delete current collection and reindex all PDFs.
        Causes brief downtime. Do NOT use in production.
        """
        global _reindex_status
        folder = "bank_docs"
        pdfs = list(Path(folder).glob("*.pdf")) if os.path.exists(folder) else []

        if not pdfs:
            return {"error": "No PDFs found in bank_docs/"}

        logger.info(f"Simple reindex started — {len(pdfs)} PDFs")
        _log_progress(f"Starting simple reindex — {len(pdfs)} PDFs")

        # Delete current collection
        try:
            self.vectorstore._client.delete_collection(self.active_collection)
            logger.info(f"Deleted collection: {self.active_collection}")
            _log_progress(f"Deleted old collection: {self.active_collection}")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")

        # Recreate
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings,
            collection_name=self.active_collection
        )
        _log_progress("Recreated empty collection")

        # Reindex all
        stats = []
        for pdf in pdfs:
            _log_progress(f"Indexing: {pdf.name}")
            s = self.index_pdf(str(pdf))
            stats.append(s)
            _log_progress(
                f" Done: {pdf.name} | chunks={s['chunks']} | class={s['doc_class']}"
            )

        total = self.get_doc_count()
        result = {
            "mode": "simple",
            "collection": self.active_collection,
            "total_chunks": total,
            "files": stats
        }
        logger.info(f"Simple reindex complete  | total_chunks={total}")
        return result

    def reindex_bluegreen(self):
        """
        PRODUCTION — Build new versioned collection in background,
        switch traffic when ready. Zero downtime.
        Old collection kept as backup for 30 mins.
        """
        global _reindex_status
        folder = "bank_docs"
        pdfs = list(Path(folder).glob("*.pdf")) if os.path.exists(folder) else []

        if not pdfs:
            return {"error": "No PDFs found in bank_docs/"}

        old_collection = self.active_collection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_collection = f"bank_policies_{timestamp}"

        logger.info(f"Blue/Green reindex started")
        logger.info(f"Old: {old_collection} → New: {new_collection}")
        _log_progress(f"Blue/Green reindex started")
        _log_progress(f"Old collection: {old_collection}")
        _log_progress(f"New collection: {new_collection}")

        # ── Step 1: Build new collection ──────────────────
        new_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings,
            collection_name=new_collection
        )
        _log_progress(f"Created new empty collection: {new_collection}")

        # ── Step 2: Reindex all PDFs into NEW collection ──
        stats = []
        for pdf in pdfs:
            _log_progress(f"Indexing into new collection: {pdf.name}")
            s = self.index_pdf(str(pdf), target_vectorstore=new_store)
            stats.append(s)
            _log_progress(
                f" {pdf.name} | chunks={s['chunks']} | class={s['doc_class']}"
            )

        # ── Step 3: Verify new collection ─────────────────
        new_count = new_store._collection.count()
        _log_progress(f"New collection has {new_count} chunks")

        if new_count == 0:
            error = "New collection is empty — aborting switch!"
            logger.error(error)
            _log_progress(f"❌ {error}")
            try:
                new_store._client.delete_collection(new_collection)
            except Exception:
                pass
            raise Exception(error)

        # ── Step 4: Switch traffic to new collection ──────
        logger.info(f"Switching traffic: {old_collection} → {new_collection}")
        self.vectorstore = new_store
        self.active_collection = new_collection
        self._save_active_collection(new_collection)
        _log_progress(f" Traffic switched to: {new_collection}")

        # ── Step 5: Schedule old collection cleanup ────────
        self._schedule_cleanup(old_collection, delay_mins=30)
        _log_progress(
            f"Old collection '{old_collection}' scheduled for cleanup in 30 mins"
        )

        result = {
            "mode": "bluegreen",
            "old_collection": old_collection,
            "new_collection": new_collection,
            "total_chunks": new_count,
            "files": stats
        }
        logger.info(f"Blue/Green reindex complete  | new={new_collection} | chunks={new_count}")
        return result

    def _schedule_cleanup(self, collection_name, delay_mins=30):
        """Delete old collection after delay — non-blocking."""

        def _delete():
            logger.info(
                f"Waiting {delay_mins} min before deleting old collection: {collection_name}"
            )
            time.sleep(delay_mins * 60)
            try:
                self.vectorstore._client.delete_collection(collection_name)
                logger.info(f"Old collection deleted: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete old collection {collection_name}: {e}")

        t = threading.Thread(target=_delete, daemon=True)
        t.start()

    # ═══════════════════════════════════════════════════════
    # SMART RETRIEVAL — no keywords, no hardcoding
    # ═══════════════════════════════════════════════════════

    def _get_retriever(self, question, doc_filter=None):
        """
        3-step smart retrieval — no hardcoded keywords:
        Step 1: Explicit doc_filter from API caller
        Step 2: Semantic routing via embedding similarity
        Step 3: Fallback — search ALL documents
        """
        # ── Step 1: Explicit filter ───────────────────────
        if doc_filter:
            known = self._get_indexed_doc_types()
            if doc_filter in known:
                logger.info(f"Step 1: Explicit filter → {doc_filter}")
                return self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10, "filter": {"doc_type": doc_filter}}
                )
            logger.warning(
                f"doc_filter '{doc_filter}' not in DB — known: {known}"
            )

        # ── Step 2: Semantic routing ──────────────────────
        try:
            probe = self.vectorstore.similarity_search_with_score(question, k=5)
            confident = [
                doc.metadata.get("doc_type", "")
                for doc, score in probe
                if score < 0.6
            ]
            if confident:
                best = Counter(confident).most_common(1)[0][0]
                logger.info(f"Step 2: Semantic route → {best}")
                return self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10, "filter": {"doc_type": best}}
                )
        except Exception as e:
            logger.warning(f"Step 2: Semantic routing error: {e}")

        # ── Step 3: Search all ────────────────────────────
        logger.info("Step 3: Searching all documents")
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

    # ═══════════════════════════════════════════════════════
    # LCEL CHAIN
    # ═══════════════════════════════════════════════════════

    def _format_docs(self, docs):
        return "\n\n".join(
            f"[{doc.metadata.get('filename', '?')} | Page {doc.metadata.get('page', '?')}]\n"
            f"{doc.page_content}"
            for doc in docs
        )

    def _build_chain(self, retriever):
        fmt = self._format_docs
        return (
            RunnableParallel(
                input=RunnablePassthrough(),
                context=retriever,
            ).assign(
                answer=(
                        {
                            "context": lambda x: fmt(x["context"]),
                            "input": lambda x: x["input"]
                        }
                        | self.prompt
                        | self.llm
                        | StrOutputParser()
                )
            )
        )

    # ═══════════════════════════════════════════════════════
    # MAIN ANSWER FUNCTION
    # ═══════════════════════════════════════════════════════

    def answer(self, question, doc_filter=None):
        t0 = time.time()
        logger.info(f"Question: '{question[:80]}'")

        retriever = self._get_retriever(question, doc_filter)

        try:
            chain = self._build_chain(retriever)
            result = chain.invoke(question)
        except Exception as e:
            logger.error(f"Chain failed: {e}")
            raise

        sources = list({
            f"{d.metadata.get('filename', '?')} pg.{d.metadata.get('page', '?')}"
            for d in result["context"]
        })
        routed_to = (
            result["context"][0].metadata.get("doc_type", "general")
            if result["context"] else "general"
        )
        num_docs = len(result["context"])
        confidence = "high" if num_docs >= 3 else "medium" if num_docs >= 1 else "low"
        elapsed = time.time() - t0

        logger.info(
            f"Done | routed_to={routed_to} | sources={len(sources)} "
            f"| confidence={confidence} | {elapsed:.2f}s"
        )
        return {
            "question": question,
            "answer": result["answer"],
            "sources": sources or ["No sources found"],
            "routed_to": routed_to,
            "confidence": confidence
        }

    # ═══════════════════════════════════════════════════════
    # UTILITY FUNCTIONS
    # ═══════════════════════════════════════════════════════

    def _get_indexed_filenames(self):
        try:
            results = self.vectorstore._collection.get(include=["metadatas"])
            return {m.get("filename", "") for m in results["metadatas"]}
        except Exception as e:
            logger.error(f"Failed to get filenames: {e}")
            return set()

    def _get_indexed_doc_types(self):
        try:
            results = self.vectorstore._collection.get(include=["metadatas"])
            return {m.get("doc_type", "") for m in results["metadatas"]}
        except Exception as e:
            logger.error(f"Failed to get doc types: {e}")
            return set()

    def get_doc_count(self):
        try:
            return self.vectorstore._collection.count()
        except Exception as e:
            logger.error(f"Failed to get doc count: {e}")
            return 0

    def get_collection_info(self):
        try:
            results = self.vectorstore._collection.get(include=["metadatas"])
            file_info = {}
            for m in results["metadatas"]:
                fname = m.get("filename", "unknown")
                if fname not in file_info:
                    file_info[fname] = {
                        "filename": fname,
                        "doc_type": m.get("doc_type", "?"),
                        "doc_class": m.get("doc_class", "?"),
                        "chunk_size": m.get("chunk_size", "?"),
                        "chunk_count": 0
                    }
                file_info[fname]["chunk_count"] += 1

            return {
                "active_collection": self.active_collection,
                "total_chunks": self.get_doc_count(),
                "total_files": len(file_info),
                "files": list(file_info.values())
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# REINDEX STATUS HELPERS (used by FastAPI endpoints)
# ═══════════════════════════════════════════════════════════

def _log_progress(msg):
    global _reindex_status
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _reindex_status["progress"].append(line)
    logger.info(msg)


def get_reindex_status():
    return dict(_reindex_status)


def run_reindex_background(engine, mode):
    """
    Run reindex in background thread.
    Called by FastAPI endpoint — non-blocking.
    """
    global _reindex_status

    with _reindex_lock:
        if _reindex_status["running"]:
            return False  # already running

        _reindex_status = {
            "running": True,
            "mode": mode,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "progress": [],
            "result": None,
            "error": None,
        }

    def _run():
        global _reindex_status
        try:
            if mode == "simple":
                result = engine.reindex_simple()
            else:
                result = engine.reindex_bluegreen()

            with _reindex_lock:
                _reindex_status["running"] = False
                _reindex_status["finished_at"] = datetime.now().isoformat()
                _reindex_status["result"] = result

        except Exception as e:
            logger.error(f"Reindex failed: {e}")
            with _reindex_lock:
                _reindex_status["running"] = False
                _reindex_status["finished_at"] = datetime.now().isoformat()
                _reindex_status["error"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return True
