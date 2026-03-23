from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as PGVectorStore
import os

# Configuration
os.environ["GOOGLE_API_KEY"] = "<API-KEY>"

PDF_PATH = "/Users/pritamdhara/Documents/Python/smallproj/AI_practitioner_removed.pdf"
COLLECTION_NAME = "ai_practitioner_docs"
CONNECTION_STRING = "postgresql+psycopg://pritamdhara:password@localhost:5432/vectordb"

# Step 1: Load PDF
print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Step 2: Split into chunks
print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3: Initialize embeddings
print("Initializing embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Step 4: Create vector store and add documents
print(f"Storing chunks in PostgreSQL table: {COLLECTION_NAME}")

try:
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
    
    # Add all chunks to the database
    # vector_store.add_documents(chunks)
    
    print(f"✅ Successfully stored {len(chunks)} chunks!")
    
    # Verify storage
    print("\nTesting similarity search...")
    results = vector_store.similarity_search("what is primary source for training a dataset", k=2)
    
    print(f"\nFound {len(results)} similar chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:200] + "...")
        
except Exception as e:
    print(f"❌ Error: {e}")


from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_scores(query: str, docs: list, top_n: int = 3) -> list:
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [(round(float(score), 4), doc) for score, doc in scored[:top_n]]

# Usage
candidates = vector_store.similarity_search(
    "what is primary source for training a dataset", k=20
)
results = rerank_with_scores(
    "what is primary source for training a dataset", candidates, top_n=3
)

for rank, (score, doc) in enumerate(results, 1):
    print(f"\n--- Result {rank} | Score: {score} ---")
    print(doc.page_content[:200] + "...")
    print(f"Source: page {doc.metadata.get('page', 'N/A')}")
