import os
import time
import numpy as np
from typing import List, Dict

from dotenv import load_dotenv

# LangChain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from langchain_groq import ChatGroq

from qdrant_client import QdrantClient, models

# FastEmbed sparse + ColBERT
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from langchain_community.embeddings import FastEmbedEmbeddings


# =========================
# Environment Variables
# =========================
COLLECTION_NAME = "groq_hybrid_rag"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================
# Clients
# =========================
qdrant_client = QdrantClient(url="http://localhost:6333")

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.2)


# =========================
# Embeddings
# =========================
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")


# =========================
# DB Initialization (Hybrid)
# =========================
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


# =========================
# Hybrid Retriever with ColBERT rerank
# =========================
class HybridColBERTRetriever(BaseRetriever):
    client: QdrantClient
    top_k: int = 5
    last_latency_ms: float = 0.0

    def __init__(self, client: QdrantClient, top_k: int = 5):
        super().__init__(client=client, top_k=top_k)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        start_time = time.time()

        # Dense
        dq = embeddings.embed_query(query)

        # Sparse
        sq_raw = list(sparse_model.embed([query]))[0]
        sq = models.SparseVector(indices=sq_raw.indices.tolist(), values=sq_raw.values.tolist())

        # ColBERT
        cq = next(colbert_model.embed([query])).tolist()

        # Qdrant hybrid prefetch + ColBERT rerank
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

        return [
            Document(page_content=r.payload["text"], metadata={"id": r.payload["id"]})
            for r in results
        ]


# =========================
# Query Rewriting
# =========================
def rewrite_query(query: str) -> str:
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [("user", "Rewrite this query for better document retrieval:\n{query}")]
    )

    chain = rewrite_prompt | llm | StrOutputParser()
    return chain.invoke({"query": query}).strip()


# =========================
# Context Builder
# =========================
def build_context(docs: List[Document], max_tokens: int = 1500):
    context = []
    token_count = 0

    for i, d in enumerate(docs):
        text = d.page_content
        tokens = len(text.split())

        if token_count + tokens > max_tokens:
            break

        context.append(f"[Doc {i+1}] {text}")
        token_count += tokens

    return "\n".join(context)


# =========================
# Answer Generation Chain
# =========================
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a technical assistant.
Answer using ONLY the context below.
If the answer is not present, say \"I don't know\".

Context:
{context}

Question:
{question}
"""
)

qa_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
)


# =========================
# RAG Pipeline
# =========================
def rag_pipeline(query: str, retriever: HybridColBERTRetriever):
    rewritten_query = rewrite_query(query)

    docs = retriever.invoke(rewritten_query)
    latency = retriever.last_latency_ms

    context = build_context(docs)

    answer = qa_chain.invoke({"context": context, "question": query})

    return {
        "query": query,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources": [d.metadata.get("id") for d in docs],
        "latency_ms": latency,
    }


# =========================
# Evaluation
# =========================
def run_evaluation(test_data: List[Dict], retriever: HybridColBERTRetriever):
    latencies = []
    reciprocal_ranks = []
    recall_at_10 = []

    print(f"🚀 Running Evaluation on {len(test_data)} queries...")

    for item in test_data:
        docs = retriever.invoke(item["query"])
        latencies.append(retriever.last_latency_ms)

        hit_rank = -1
        for i, d in enumerate(docs):
            if d.metadata.get("id") == item["ground_truth_id"]:
                hit_rank = i + 1
                break

        if hit_rank != -1:
            reciprocal_ranks.append(1.0 / hit_rank)
            recall_at_10.append(1)
        else:
            reciprocal_ranks.append(0.0)
            recall_at_10.append(0)

    print("\n" + "=" * 30)
    print("📊 RETRIEVAL PERFORMANCE")
    print(f"Recall@10: {np.mean(recall_at_10):.4f}")
    print(f"MRR:       {np.mean(reciprocal_ranks):.4f}")
    print("\n⏱️ LATENCY PROFILING")
    print(f"P50: {np.percentile(latencies, 50):.2f} ms")
    print(f"P95: {np.percentile(latencies, 95):.2f} ms")
    print("=" * 30)


# =========================
# Main Execution
# =========================
def main():
    init_db()

    retriever = HybridColBERTRetriever(client=qdrant_client)

    # Dummy KB
    docs = [
        {"id": 1, "text": "Groq LPUs use a deterministic architecture to eliminate jitter in LLM inference."},
        {"id": 2, "text": "Qdrant server-side reranking reduces data transfer and improves precision."},
        {"id": 3, "text": "ColBERT late interaction enables token-level matching without cross-encoder latency."},
        {"id": 4, "text": "Hybrid search combines dense semantic vectors with sparse keyword vectors."},
    ]

    texts = [d["text"] for d in docs]

    d_vecs = [embeddings.embed_query(t) for t in texts]

    s_vecs = []
    for s in sparse_model.embed(texts):
        s_vecs.append(models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()))

    c_vecs = [v.tolist() for v in colbert_model.embed(texts)]

    points = [
        models.PointStruct(
            id=d["id"],
            vector={"dense": d_vecs[i], "sparse": s_vecs[i], "colbert": c_vecs[i]},
            payload=d,
        )
        for i, d in enumerate(docs)
    ]

    qdrant_client.upsert(COLLECTION_NAME, points)

    eval_set = [
        {"query": "How do LPUs handle jitter?", "ground_truth_id": 1},
        {"query": "Tell me about Qdrant reranking benefits.", "ground_truth_id": 2},
        {"query": "token level matching model", "ground_truth_id": 3},
    ]

    run_evaluation(eval_set, retriever)

    response = rag_pipeline("How does ColBERT improve retrieval?", retriever)

    print("\n🧠 RAG ANSWER")
    print(response["answer"])
    print("Sources:", response["sources"])


if __name__ == "__main__":
    main()