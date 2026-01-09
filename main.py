# import os
# import time
# import numpy as np
# from typing import List, Dict
# from qdrant_client import QdrantClient, models
# from groq import Groq
# from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
# from dotenv import load_dotenv


# COLLECTION_NAME = "groq_hybrid_rag"
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# QDRANT_URL = os.getenv("QDRANT_URL")        
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# groq_client = Groq(api_key=GROQ_API_KEY)

# # Embedding Models (unchanged)
# dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
# sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
# colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

# client = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY
# )

# def init_db():
#     if not client.collection_exists(COLLECTION_NAME):
#         client.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config={
#                 "dense": models.VectorParams(
#                     size=384, distance=models.Distance.COSINE
#                 ),
#                 "colbert": models.VectorParams(
#                     size=128,
#                     distance=models.Distance.COSINE,
#                     multivector_config=models.MultiVectorConfig(
#                         comparator=models.MultiVectorComparator.MAX_SIM
#                     ),
#                     hnsw_config=models.HnswConfigDiff(m=0)
#                 )
#             },
#             sparse_vectors_config={
#                 "sparse": models.SparseVectorParams(
#                     modifier=models.Modifier.IDF
#                 )
#             }
#         )
        
# def rewrite_query(query: str) -> str:
#     response = groq_client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[
#             {
#                 "role": "user",
#                 "content": f"Rewrite this query for better document retrieval:\n{query}"
#             }
#         ],
#         temperature=0.0
#     )
#     return response.choices[0].message.content.strip()

# def advanced_retrieval(query: str, top_k=5):
#     start_time = time.time()

#     dq = list(dense_model.embed([query]))[0]

#     sq_raw = list(sparse_model.embed([query]))[0]
#     sq = models.SparseVector(
#         indices=sq_raw.indices.tolist(),
#         values=sq_raw.values.tolist()
#     )

#     cq = next(colbert_model.embed([query])).tolist()

#     results = client.query_points(
#         collection_name=COLLECTION_NAME,
#         prefetch=[
#             models.Prefetch(query=dq, using="dense", limit=25),
#             models.Prefetch(query=sq, using="sparse", limit=25),
#         ],
#         query=cq,
#         using="colbert",
#         limit=top_k,
#         with_payload=True
#     ).points

#     latency = (time.time() - start_time) * 1000
#     return results, latency

# def build_context(results, max_tokens=1500):
#     context = []
#     token_count = 0

#     for r in results:
#         text = r.payload["text"]
#         tokens = len(text.split())
#         if token_count + tokens > max_tokens:
#             break
#         context.append(f"[Doc {r.payload['id']}] {text}")
#         token_count += tokens

#     return "\n".join(context)

# def generate_answer(query: str, context: str) -> str:
#     prompt = f"""
# You are a technical assistant.
# Answer using the context below.
# If the answer is not present, say "I don't know".

# Context:
# {context}

# Question:
# {query}
# """
#     response = groq_client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2
#     )
#     print(response.choices[0].message.content.strip())
#     return response.choices[0].message.content.strip()

# def rag_pipeline(query: str, top_k=5):
#     rewritten_query = rewrite_query(query)

#     results, latency = advanced_retrieval(rewritten_query, top_k=top_k)

#     context = build_context(results)

#     answer = generate_answer(query, context)

#     return {
#         "query": query,
#         "rewritten_query": rewritten_query,
#         "answer": answer,
#         "sources": [r.payload["id"] for r in results],
#         "latency_ms": latency
#     }
    
# def run_evaluation(test_data: List[Dict]):
#     latencies = []
#     reciprocal_ranks = []
#     recall_at_10 = []

#     print(f"🚀 Running Evaluation on {len(test_data)} queries...")

#     for item in test_data:
#         results, lat = advanced_retrieval(item['query'], top_k=10)
#         latencies.append(lat)

#         hit_rank = -1
#         for i, res in enumerate(results):
#             if res.payload['id'] == item['ground_truth_id']:
#                 hit_rank = i + 1
#                 break

#         if hit_rank != -1:
#             reciprocal_ranks.append(1.0 / hit_rank)
#             recall_at_10.append(1)
#         else:
#             reciprocal_ranks.append(0.0)
#             recall_at_10.append(0)

#     print("\n" + "=" * 30)
#     print("📊 RETRIEVAL PERFORMANCE")
#     print(f"Recall@10: {np.mean(recall_at_10):.4f}")
#     print(f"MRR:       {np.mean(reciprocal_ranks):.4f}")
#     print("\n⏱️ LATENCY PROFILING")
#     print(f"P50: {np.percentile(latencies, 50):.2f} ms")
#     print(f"P95: {np.percentile(latencies, 95):.2f} ms")
#     print("=" * 30)

# init_db()

# # Dummy Knowledge Base (UNCHANGED)
# docs = [
#     {"id": 1, "text": "Groq LPUs use a deterministic architecture to eliminate jitter in LLM inference."},
#     {"id": 2, "text": "Qdrant server-side reranking reduces data transfer and improves precision."},
#     {"id": 3, "text": "ColBERT late interaction enables token-level matching without cross-encoder latency."},
#     {"id": 4, "text": "Hybrid search combines the semantic depth of dense vectors with keyword-exact sparse vectors."}
# ]

# texts = [d["text"] for d in docs]

# d_vecs = [v.tolist() for v in dense_model.embed(texts)]
# s_vecs = [
#     models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
#     for s in sparse_model.embed(texts)
# ]
# c_vecs = [v.tolist() for v in colbert_model.embed(texts)]

# points = [
#     models.PointStruct(
#         id=d["id"],
#         vector={
#             "dense": d_vecs[i],
#             "sparse": s_vecs[i],
#             "colbert": c_vecs[i]
#         },
#         payload=d
#     )
#     for i, d in enumerate(docs)
# ]

# client.upsert(COLLECTION_NAME, points)


# eval_set = [
#     {"query": "How do LPUs handle jitter?", "ground_truth_id": 1},
#     {"query": "Tell me about Qdrant's reranking benefits.", "ground_truth_id": 2},
#     {"query": "token-level matching models", "ground_truth_id": 3}
# ]

# run_evaluation(eval_set)

# # Example RAG call
# response = rag_pipeline("How does ColBERT improve retrieval?")
# print("\n🧠 RAG ANSWER")
# print(response["answer"])
# print("Sources:", response["sources"])


# =========================
# Imports & Environment
# =========================
import os
import time
import numpy as np
from typing import List, Dict

from qdrant_client import QdrantClient, models
from groq import Groq
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from dotenv import load_dotenv


# =========================
# Environment Variables
# =========================
COLLECTION_NAME = "groq_hybrid_rag"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# =========================
# Clients
# =========================
groq_client = Groq(api_key=GROQ_API_KEY)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# =========================
# Embedding Models
# =========================
dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")


# =========================
# Database Initialization
# =========================
def init_db():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                ),
                "colbert": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )


# =========================
# Query Rewriting
# =========================
def rewrite_query(query: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": f"Rewrite this query for better document retrieval:\n{query}"
            }
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


# =========================
# Retrieval
# =========================
def advanced_retrieval(query: str, top_k=5):
    start_time = time.time()

    dq = list(dense_model.embed([query]))[0]

    sq_raw = list(sparse_model.embed([query]))[0]
    sq = models.SparseVector(
        indices=sq_raw.indices.tolist(),
        values=sq_raw.values.tolist()
    )

    cq = next(colbert_model.embed([query])).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dq, using="dense", limit=25),
            models.Prefetch(query=sq, using="sparse", limit=25),
        ],
        query=cq,
        using="colbert",
        limit=top_k,
        with_payload=True
    ).points

    latency = (time.time() - start_time) * 1000
    return results, latency


# =========================
# Context Builder
# =========================
def build_context(results, max_tokens=1500):
    context = []
    token_count = 0

    for r in results:
        text = r.payload["text"]
        tokens = len(text.split())
        if token_count + tokens > max_tokens:
            break
        context.append(f"[Doc {r.payload['id']}] {text}")
        token_count += tokens

    return "\n".join(context)


# =========================
# Answer Generation
# =========================
def generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are a technical assistant.
Answer using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()


# =========================
# RAG Pipeline
# =========================
def rag_pipeline(query: str, top_k=5):
    rewritten_query = rewrite_query(query)
    results, latency = advanced_retrieval(rewritten_query, top_k=top_k)
    context = build_context(results)
    answer = generate_answer(query, context)

    return {
        "query": query,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources": [r.payload["id"] for r in results],
        "latency_ms": latency
    }


# =========================
# Evaluation
# =========================
def run_evaluation(test_data: List[Dict]):
    latencies = []
    reciprocal_ranks = []
    recall_at_10 = []

    print(f"🚀 Running Evaluation on {len(test_data)} queries...")

    for item in test_data:
        results, lat = advanced_retrieval(item['query'], top_k=10)
        latencies.append(lat)

        hit_rank = -1
        for i, res in enumerate(results):
            if res.payload['id'] == item['ground_truth_id']:
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

    # Dummy Knowledge Base
    docs = [
        {"id": 1, "text": "Groq LPUs use a deterministic architecture to eliminate jitter in LLM inference."},
        {"id": 2, "text": "Qdrant server-side reranking reduces data transfer and improves precision."},
        {"id": 3, "text": "ColBERT late interaction enables token-level matching without cross-encoder latency."},
        {"id": 4, "text": "Hybrid search combines the semantic depth of dense vectors with keyword-exact sparse vectors."}
    ]

    texts = [d["text"] for d in docs]

    d_vecs = [v.tolist() for v in dense_model.embed(texts)]
    s_vecs = [
        models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
        for s in sparse_model.embed(texts)
    ]
    c_vecs = [v.tolist() for v in colbert_model.embed(texts)]

    points = [
        models.PointStruct(
            id=d["id"],
            vector={
                "dense": d_vecs[i],
                "sparse": s_vecs[i],
                "colbert": c_vecs[i]
            },
            payload=d
        )
        for i, d in enumerate(docs)
    ]

    client.upsert(COLLECTION_NAME, points)

    eval_set = [
        {"query": "How do LPUs handle jitter?", "ground_truth_id": 1},
        {"query": "Tell me about Qdrant's reranking benefits.", "ground_truth_id": 2},
        {"query": "token-level matching models", "ground_truth_id": 3}
    ]

    run_evaluation(eval_set)

    response = rag_pipeline("How does ColBERT improve retrieval?")
    print("\n🧠 RAG ANSWER")
    print(response["answer"])
    print("Sources:", response["sources"])


if __name__ == "__main__":
    main()
