from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, Session
from pgvector.sqlalchemy import Vector
from datetime import datetime
import os

# Configuration
os.environ["GOOGLE_API_KEY"] = "<YOUR-API-KEY>"

PDF_PATH = "/Users/pritamdhara/Documents/Python/smallproj/AI_practitioner_removed.pdf"
CONNECTION_STRING = "postgresql+psycopg://pritamdhara:password@localhost:5432/vectordb"
EMBEDDING_DIM = 768  # gemini-embedding-2-preview dimension

# ─── Step 1: Define your own schema ───────────────────────────────────────────

Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(String(500), nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content     = Column(Text, nullable=False)
    embedding   = Column(Vector(EMBEDDING_DIM), nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)

# ─── Step 2: Create table if not exists ───────────────────────────────────────

engine = create_engine(CONNECTION_STRING)

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

Base.metadata.create_all(engine)  # creates document_chunks table
print("✅ Table 'document_chunks' ready")

# ─── Step 3: Load PDF ─────────────────────────────────────────────────────────

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# ─── Step 4: Split into chunks ────────────────────────────────────────────────

print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ─── Step 5: Initialize models ────────────────────────────────────────────────

print("Initializing embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

print("Initializing reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Initializing LLM...")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:
""")

# ─── Step 6: Insert chunks into your custom table ─────────────────────────────

# Uncomment this block to insert chunks (run once)
# print("Generating embeddings and inserting chunks...")
# with Session(engine) as session:
#     for i, chunk in enumerate(chunks):
#         vector = embeddings.embed_query(chunk.page_content)
#         row = DocumentChunk(
#             source_file = PDF_PATH,
#             page_number = chunk.metadata.get("page", 0),
#             chunk_index = i,
#             content     = chunk.page_content,
#             embedding   = vector,
#         )
#         session.add(row)
#     session.commit()
# print(f"✅ Inserted {len(chunks)} chunks")

# ─── Step 7: Similarity search against your custom table ──────────────────────

def similarity_search(query: str, k: int = 20) -> list[Document]:
    query_vector = embeddings.embed_query(query)
    
    with Session(engine) as session:
        # pgvector <=> = cosine distance operator
        results = session.execute(
            text("""
                SELECT content, source_file, page_number, chunk_index,
                       1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM document_chunks
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :k
            """),
            {"vec": str(query_vector), "k": k}
        ).fetchall()

    return [
        Document(
            page_content=row.content,
            metadata={
                "source_file": row.source_file,
                "page":        row.page_number,
                "chunk_index": row.chunk_index,
                "similarity":  round(row.similarity, 4),
            }
        )
        for row in results
    ]

# ─── Step 8: Reranker ─────────────────────────────────────────────────────────

def rerank_with_scores(query: str, docs: list, top_n: int = 3) -> list:
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [(round(float(score), 4), doc) for score, doc in scored[:top_n]]

# ─── Step 9: Answer generation ────────────────────────────────────────────────

def generate_answer(query: str, reranked_results: list) -> str:
    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for _, doc in reranked_results
    )
    print(f"\n📄 Sending {len(reranked_results)} chunks to LLM...")
    print(f"📝 Total context length: {len(context)} characters\n")

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})

    if isinstance(response.content, list):
        return " ".join(b['text'] for b in response.content if b.get('type') == 'text')
    return response.content

# ─── Step 10: Run the RAG pipeline ────────────────────────────────────────────

try:
    QUERY = "Explain the concept of timeseries data?"

    print("\nRunning similarity search...")
    candidates = similarity_search(QUERY, k=20)

    print("Running reranker...")
    reranked_results = rerank_with_scores(QUERY, candidates, top_n=3)

    print(f"\nTop {len(reranked_results)} reranked chunks:")
    for rank, (score, doc) in enumerate(reranked_results, 1):
        print(f"\n--- Chunk {rank} | Rerank: {score} | Vector sim: {doc.metadata['similarity']} | Page {doc.metadata['page']} ---")
        print(doc.page_content[:150] + "...")

    print("\n" + "=" * 60)
    print("GENERATED ANSWER:")
    print("=" * 60)
    answer = generate_answer(QUERY, reranked_results)
    print(answer)
    print("=" * 60)

except Exception as e:
    print(f"❌ Error: {e}")
    raise
