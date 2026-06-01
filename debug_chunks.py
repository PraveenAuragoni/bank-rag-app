# reload_pdf.py
# pip install langchain-huggingface langchain-chroma pypdf sentence-transformers

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Step 1: Load PDF ──────────────────────────────────────
PDF_PATH = "./bank_docs/gbtc.pdf"   # ← your PDF path here

print(f"Loading PDF: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
pages  = loader.load()
print(f"✅ Loaded {len(pages)} pages")

# ── Step 2: Split into chunks ─────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # bigger chunks = more context per chunk
    chunk_overlap=200,    # overlap so nothing is cut off at edges
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(pages)
print(f"✅ Split into {len(chunks)} chunks")

# Preview first chunk
print(f"\nFirst chunk preview:")
print(chunks[0].page_content[:300])
print(f"Metadata: {chunks[0].metadata}")

# ── Step 3: Check if EDP is in raw pages BEFORE embedding ─
print("\n" + "=" * 60)
print("CHECKING RAW PDF FOR 'EDP'")
print("=" * 60)
edp_found_in_pdf = False
for i, page in enumerate(pages):
    if "EDP" in page.page_content:
        edp_found_in_pdf = True
        print(f"✅ Found 'EDP' on page {i+1}")
        # Show context around EDP
        idx = page.page_content.index("EDP")
        print(page.page_content[max(0, idx-100):idx+300])
        print("─" * 40)

if not edp_found_in_pdf:
    print("❌ 'EDP' not found in PDF at all!")
    print("   Check if this is the correct PDF file")

# ── Step 4: Delete old empty DB and recreate ─────────────
import shutil
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("\n✅ Deleted old empty chroma_db")

# ── Step 5: Embed and store ───────────────────────────────
print("\nCreating embeddings — this may take a minute...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="bank_policies"   # ← explicit name, not default "langchain"
)

total = vectorstore._collection.count()
print(f"✅ Stored {total} chunks in ChromaDB")

# ── Step 6: Verify EDP is now in DB ──────────────────────
print("\n" + "=" * 60)
print("VERIFYING EDP IN CHROMADB")
print("=" * 60)

all_data = vectorstore._collection.get(include=["documents", "metadatas"])
found = False
for i in range(len(all_data["ids"])):
    chunk = all_data["documents"][i]
    if "EDP" in chunk:
        found = True
        print(f"✅ Found EDP in chunk {i} | Page: {all_data['metadatas'][i].get('page','?')}")
        print(chunk[:400])
        print("─" * 40)

if not found:
    print("❌ EDP still not found — wrong PDF file!")

# ── Step 7: Test search ───────────────────────────────────
print("\n" + "=" * 60)
print("TEST SEARCH")
print("=" * 60)

results = vectorstore.similarity_search_with_score(
    "General terms relating to EDP Instructions of Customer Payee",
    k=5
)

for i, (doc, score) in enumerate(results):
    print(f"\nRank #{i+1} | Score: {score:.4f} | Page: {doc.metadata.get('page','?')}")
    print(doc.page_content[:300])
    print("─" * 40)