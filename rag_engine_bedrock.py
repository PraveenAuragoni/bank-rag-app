# rag_engine_bedrock.py
# LangChain 1.3.2 + AWS Bedrock

import os
import time
import boto3
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock, BedrockEmbeddings  # NEW — replaces Anthropic + HuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from logger_config import get_logger

logger = get_logger(__name__)


class RAGEngineBedrock:
    KEYWORDS = {
        "savings": "savings_account",
        "minimum balance": "savings_account",
        "fd": "fixed_deposit",
        "fixed deposit": "fixed_deposit",
        "home loan": "home_loan",
        "processing fee": "home_loan",
        "kyc": "kyc",
        "aadhaar": "kyc",
        "password": "digital_banking",
        "otp": "digital_banking",
        "neft": "transfers",
        "rtgs": "transfers",
    }

    def __init__(self):
        logger.info("Initialising RAG Engine with AWS Bedrock...")

        # AWS Bedrock client — uses ~/.aws/credentials automatically
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "ap-southeast-1")
        )

        # 1: Cohere Embed English v3 (Titan Embeddings not available) ────────
        # BedrockEmbeddings (free on AWS, no download needed)
        logger.info("Loading Cohere Embed English v3 via Bedrock...")
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="cohere.embed-english-v3"
        )
        logger.info("Cohere Embed English v3 Embeddings ready")

        # ──  2: ChatBedrock ───────────
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_kwargs={
                "temperature": 0,
                "max_tokens": 1024
            }
        )
        logger.info("ChatBedrock (Claude 3 Haiku) ready")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional banking policy expert.
Answer using ONLY the context provided.
If not found say: 'Not available in current documents.'
Always mention document name and page number.

Context:
{context}"""),
            ("human", "{input}")
        ])

        # ChromaDB — same as before, just different embeddings
        os.makedirs("./chroma_db_bedrock", exist_ok=True)
        self.vectorstore = Chroma(
            persist_directory="./chroma_db_bedrock",
            embedding_function=self.embeddings,
            collection_name="bank_policies_bedrock"
        )
        logger.info(
            f"ChromaDB loaded | chunks={self.vectorstore._collection.count()}"
        )

        self._index_existing_pdfs()
        logger.info("RAG Engine with Bedrock ready")

    def _index_existing_pdfs(self):
        folder = "bank_docs"
        if not os.path.exists(folder):
            return
        pdfs = list(Path(folder).glob("*.pdf"))
        existing = self._get_indexed_filenames()
        for pdf in pdfs:
            if pdf.name not in existing:
                self.index_pdf(str(pdf))

    def _get_indexed_filenames(self):
        try:
            results = self.vectorstore._collection.get(include=["metadatas"])
            return {m.get("filename", "") for m in results["metadatas"]}
        except Exception:
            return set()

    def index_pdf(self, pdf_path):
        filename = os.path.basename(pdf_path)
        doc_type = Path(pdf_path).stem.lower().replace(" ", "_")
        logger.info(f"Indexing: {filename}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for page in pages:
            page.metadata["filename"] = filename
            page.metadata["doc_type"] = doc_type

        chunks = self.splitter.split_documents(pages)
        for i in range(0, len(chunks), 96):
            self.vectorstore.add_documents(chunks[i:i + 96])
        logger.info(f"Indexed {len(chunks)} chunks from {filename}")
        return len(chunks)

    def _format_docs(self, docs):
        return "\n\n".join(
            f"[{doc.metadata.get('filename', '?')} | Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )

    def _route(self, question):
        q = question.lower()
        for keyword, doc_type in self.KEYWORDS.items():
            if keyword in q:
                logger.debug(f"Keyword route: {doc_type}")
                return doc_type
        return "general"

    def _build_chain(self, retriever):
        format_fn = self._format_docs
        return RunnableParallel(
            input=RunnablePassthrough(),
            context=retriever,
        ).assign(
            answer=(
                    {
                        "context": lambda x: format_fn(x["context"]),
                        "input": lambda x: x["input"]
                    }
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
            )
        )

    def answer(self, question, doc_filter=None):
        t0 = time.time()
        doc_type = doc_filter or self._route(question)
        logger.info(f"Answering | routed_to={doc_type} | q='{question[:60]}'")

        search_kwargs = {"k": 3}
        if doc_type != "general":
            search_kwargs["filter"] = {"doc_type": doc_type}

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        chain = self._build_chain(retriever)
        result = chain.invoke(question)

        sources = list({
            f"{d.metadata.get('filename', '?')} pg.{d.metadata.get('page', '?')}"
            for d in result["context"]
        })
        confidence = "high" if len(result["context"]) >= 3 else \
            "medium" if len(result["context"]) >= 1 else "low"
        elapsed = time.time() - t0

        logger.info(
            f"Answer ready | confidence={confidence} | sources={len(sources)} | time={elapsed:.2f}s"
        )
        return {
            "question": question,
            "answer": result["answer"],
            "sources": sources or ["No sources found"],
            "routed_to": doc_type,
            "confidence": confidence
        }

    def get_doc_count(self):
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
