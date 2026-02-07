import os
import time
import glob
import redis
import numpy as np
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from langchain_groq import ChatGroq

from qdrant_client import QdrantClient, models

from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from langchain_community.embeddings import FastEmbedEmbeddings

from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION_NAME = "groq_hybrid_rag"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

qdrant_client = QdrantClient(url="http://localhost:6333")
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.2)

redis_client = redis.from_url(REDIS_URL)

vectorizer = HFTextVectorizer(
    model="redis/langcache-embed-v1",
    cache=EmbeddingsCache(redis_client=redis_client, ttl=3600),
)

semantic_cache = SemanticCache(
    name="rag-semantic-cache",
    vectorizer=vectorizer,
    redis_client=redis_client,
    distance_threshold=0.4,
)

semantic_cache.set_ttl(86400)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

def init_db():
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
                "colbert": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

SUPPORTED_LOADERS = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": UnstructuredHTMLLoader,
}

def load_dataset(folder: str):
    docs = []

    for file in glob.glob(f"{folder}/**/*", recursive=True):
        if not Path(file).is_file():
            continue

        ext = Path(file).suffix.lower()
        if ext not in SUPPORTED_LOADERS:
            continue

        try:
            loader = SUPPORTED_LOADERS[ext](file)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed loading {file}: {e}")

    print(f"Loaded {len(docs)} raw documents")
    return docs

def chunk_documents(documents: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def ingest_chunks(chunks: List[Document]):
    if not chunks:
        return

    texts = [c.page_content for c in chunks]

    d_vecs = [embeddings.embed_query(t) for t in texts]

    s_vecs = [
        models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
        for s in sparse_model.embed(texts)
    ]

    c_vecs = [v.tolist() for v in colbert_model.embed(texts)]

    points = []

    for i, chunk in enumerate(chunks):
        points.append(
            models.PointStruct(
                id=i,
                vector={"dense": d_vecs[i], "sparse": s_vecs[i], "colbert": c_vecs[i]},
                payload={
                    "id": i,
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "unknown"),
                },
            )
        )

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} chunks")

class HybridColBERTRetriever(BaseRetriever):
    client: QdrantClient
    top_k: int = 5
    last_latency_ms: float = 0.0

    def _get_relevant_documents(self, query: str) -> List[Document]:
        start_time = time.time()

        dq = embeddings.embed_query(query)

        sq_raw = list(sparse_model.embed([query]))[0]
        sq = models.SparseVector(indices=sq_raw.indices.tolist(), values=sq_raw.values.tolist())

        cq = next(colbert_model.embed([query])).tolist()

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dq, using="dense", limit=25),
                models.Prefetch(query=sq, using="sparse", limit=25),
            ],
            query=cq,
            using="colbert",
            limit=self.top_k,
            with_payload=True,
        ).points

        self.last_latency_ms = (time.time() - start_time) * 1000

        seen = set()
        docs = []

        for r in results:
            text = r.payload["text"]
            if text in seen:
                continue
            seen.add(text)

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "id": r.payload.get("id"),
                        "source": r.payload.get("source"),
                    },
                )
            )

        return docs

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a precise technical assistant.

RULES:
- Use ONLY the provided context.
- Remove duplicate information.
- Write a concise, well-structured explanation.
- If the answer is missing, say "I don't know".

Context:
{context}

Question:
{question}

Final concise answer:
"""
)

qa_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
)

def build_context(docs: List[Document], max_tokens: int = 1500):
    context, token_count = [], 0

    for i, d in enumerate(docs):
        tokens = len(d.page_content.split())
        if token_count + tokens > max_tokens:
            break

        context.append(f"[Doc {i+1}] {d.page_content}")
        token_count += tokens

    return "\n".join(context)

def rag_pipeline(query: str, retriever: HybridColBERTRetriever):
    total_start = time.time()
    
    cache_start = time.time()
    cached = semantic_cache.check(query)
    cache_latency = (time.time() - cache_start) * 1000

    if cached:
        hit = cached[0]
        total_latency = (time.time() - total_start) * 1000
        print(f"✅ CACHE HIT | Cache Latency: {cache_latency:.2f}ms | Total: {total_latency:.2f}ms")
        return hit["response"]

    print("❌ CACHE MISS — running RAG")

    docs = retriever.invoke(query)
    retrieval_latency = retriever.last_latency_ms

    gen_start = time.time()
    context = build_context(docs)
    answer = qa_chain.invoke({"context": context, "question": query})
    gen_latency = (time.time() - gen_start) * 1000

    semantic_cache.store(prompt=query, response=answer)

    total_latency = (time.time() - total_start) * 1000

    print(f"\n--- Performance Metrics ---")
    print(f"🔹 Cache Lookup:  {cache_latency:7.2f} ms")
    print(f"🔹 Retrieval:     {retrieval_latency:7.2f} ms")
    print(f"🔹 LLM Generation: {gen_latency:7.2f} ms")
    print(f"---------------------------")
    print(f"TOTAL LATENCY:     {total_latency:7.2f} ms\n")

    return answer

def main():
    init_db()

    retriever = HybridColBERTRetriever(client=qdrant_client)

    # Uncomment for ingestion
    raw_docs = load_dataset("data")
    chunks = chunk_documents(raw_docs)
    ingest_chunks(chunks)

    question = "What are Positional Encodings?"
    start= time.time()
    answer = rag_pipeline(question, retriever)
    end = time.time()
    print(f"RAG Pipeline completed in {(end - start)*1000:.2f} ms")

    print("\nRAG ANSWER:\n", answer)


if __name__ == "__main__":
    main()
