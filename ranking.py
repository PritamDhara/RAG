from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as PGVectorStore
from sentence_transformers import CrossEncoder
import os

# Configuration
os.environ["GOOGLE_API_KEY"] = "<API-KEY>"

PDF_PATH = "/Users/pritamdhara/Documents/Python/smallproj/AI_practitioner_removed.pdf"
COLLECTION_NAME = "ai_practitioner_docs"
CONNECTION_STRING = "postgresql+psycopg://<Username>:<Password>@localhost:5432/<Database name>"

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

# Step 4: Initialize reranker
print("Initializing reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_scores(query: str, docs: list, top_n: int = 3) -> list:
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [(round(float(score), 4), doc) for score, doc in scored[:top_n]]

# Step 5: Create vector store and run searches
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

    QUERY = "Explain the concept of timeseries data?"

    # --- Original similarity search (unchanged) ---
    # print("\nTesting similarity search...")
    # similarity_results = vector_store.similarity_search(QUERY, k=2)

    # print(f"\nFound {len(similarity_results)} similar chunks:")
    # for i, doc in enumerate(similarity_results, 1):
    #     print(f"\n--- Result {i} ---")
    #     print(doc.page_content[:200] + "...")

    # --- Reranking ---
    print("\nRunning reranker...")
    candidates = vector_store.similarity_search(QUERY, k=20)
    reranked_results = rerank_with_scores(QUERY, candidates, top_n=1)

    print(f"\nTop {len(reranked_results)} reranked results:")
    for rank, (score, doc) in enumerate(reranked_results, 1):
        print(f"\n--- Result {rank} | Score: {score} ---")
        print(doc.page_content)
        print(f"Source: page {doc.metadata.get('page', 'N/A')}")

except Exception as e:
    print(f"❌ Error: {e}")
