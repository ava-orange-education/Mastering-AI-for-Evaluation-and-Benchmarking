# ============================================================
# Retrieval and Embedding Evaluation Metrics
# RAG Evaluation Framework
# ============================================================

import time
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ------------------------------------------------------------
# Sample Evaluation Dataset
# ------------------------------------------------------------

documents = [
    "To reset your password, go to the account settings and select Reset Password.",
    "Enterprise customers receive 24x7 technical support.",
    "The product warranty is valid for two years from the purchase date.",
    "Users can update their email address from the profile settings.",
    "The standard subscription includes 10 GB of storage.",
    "Password reset requests expire after 30 minutes.",
    "Premium customers receive unlimited cloud storage.",
    "Technical support requests can be submitted through the support portal."
]

queries = [
    "How can I reset my password?",
    "How long is the product warranty?",
    "Where can I change my email address?",
    "How long does a password reset request remain valid?"
]

# Ground-truth relevant document indexes for each query
ground_truth = {
    0: [0, 5],   # Password reset
    1: [2],      # Warranty
    2: [3],      # Email address
    3: [5]       # Password reset expiration
}


# ------------------------------------------------------------
# Example Embedding Function
# ------------------------------------------------------------

def generate_demo_embeddings(texts, dimension=384):
    """
    Demo embedding generator.

    Replace this with a real embedding model in production.
    """
    np.random.seed(42)
    embeddings = np.random.rand(len(texts), dimension)
    return normalize(embeddings)


# ------------------------------------------------------------
# Generate Document and Query Embeddings
# ------------------------------------------------------------

document_embeddings = generate_demo_embeddings(documents)
query_embeddings = generate_demo_embeddings(queries)


# ------------------------------------------------------------
# Similarity Search
# ------------------------------------------------------------

def retrieve_documents(query_embedding, document_embeddings, top_k=5):
    """Retrieve the top-k most similar documents."""
    scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        document_embeddings
    )[0]

    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[:top_k]
    top_scores = scores[top_indices]

    return top_indices, top_scores


# ------------------------------------------------------------
# Precision@K
# ------------------------------------------------------------

def precision_at_k(retrieved, relevant, k):
    """Relevant retrieved documents / K."""
    retrieved = retrieved[:k]
    relevant_retrieved = sum(
        1 for doc in retrieved if doc in relevant
    )
    return relevant_retrieved / k


# ------------------------------------------------------------
# Recall@K
# ------------------------------------------------------------

def recall_at_k(retrieved, relevant, k):
    """Relevant retrieved documents / Total relevant documents."""
    retrieved = retrieved[:k]
    relevant_retrieved = sum(
        1 for doc in retrieved if doc in relevant
    )

    if len(relevant) == 0:
        return 0.0

    return relevant_retrieved / len(relevant)


# ------------------------------------------------------------
# Hit Rate@K
# ------------------------------------------------------------

def hit_rate_at_k(retrieved, relevant, k):
    """Return 1 if at least one relevant document is in top-k."""
    retrieved = retrieved[:k]
    return int(any(doc in relevant for doc in retrieved))


# ------------------------------------------------------------
# Mean Reciprocal Rank
# ------------------------------------------------------------

def reciprocal_rank(retrieved, relevant):
    """1 / position of the first relevant document."""
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1 / position

    return 0.0


def mean_reciprocal_rank(all_retrieved, ground_truth):
    """Mean reciprocal rank across all queries."""
    scores = []

    for query_id, retrieved in all_retrieved.items():
        relevant = ground_truth[query_id]
        scores.append(reciprocal_rank(retrieved, relevant))

    return np.mean(scores)


# ------------------------------------------------------------
# NDCG@K
# ------------------------------------------------------------

def ndcg_at_k(retrieved, relevant, k):
    """Calculate NDCG@K using binary relevance."""
    retrieved = retrieved[:k]

    dcg = 0.0

    for position, doc in enumerate(retrieved, start=1):
        relevance = 1 if doc in relevant else 0
        dcg += relevance / np.log2(position + 1)

    ideal_relevances = [1] * min(len(relevant), k)
    idcg = 0.0

    for position, relevance in enumerate(
        ideal_relevances,
        start=1
    ):
        idcg += relevance / np.log2(position + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ------------------------------------------------------------
# Evaluate Retrieval Performance
# ------------------------------------------------------------

def evaluate_retrieval(
    queries,
    query_embeddings,
    document_embeddings,
    ground_truth,
    top_k=5
):
    results = []
    all_retrieved = {}

    for query_id, query in enumerate(queries):

        start_time = time.perf_counter()

        retrieved_indices, scores = retrieve_documents(
            query_embeddings[query_id],
            document_embeddings,
            top_k=top_k
        )

        latency = time.perf_counter() - start_time
        relevant = ground_truth.get(query_id, [])
        retrieved = retrieved_indices.tolist()

        all_retrieved[query_id] = retrieved

        precision = precision_at_k(
            retrieved, relevant, top_k
        )

        recall = recall_at_k(
            retrieved, relevant, top_k
        )

        hit_rate = hit_rate_at_k(
            retrieved, relevant, top_k
        )

        mrr = reciprocal_rank(
            retrieved, relevant
        )

        ndcg = ndcg_at_k(
            retrieved, relevant, top_k
        )

        results.append({
            "query": query,
            "precision@k": precision,
            "recall@k": recall,
            "hit_rate@k": hit_rate,
            "reciprocal_rank": mrr,
            "ndcg@k": ndcg,
            "latency_ms": latency * 1000
        })

    results_df = pd.DataFrame(results)

    summary = {
        "Mean Precision@K":
            results_df["precision@k"].mean(),

        "Mean Recall@K":
            results_df["recall@k"].mean(),

        "Mean Hit Rate@K":
            results_df["hit_rate@k"].mean(),

        "MRR":
            mean_reciprocal_rank(
                all_retrieved,
                ground_truth
            ),

        "Mean NDCG@K":
            results_df["ndcg@k"].mean(),

        "Average Latency (ms)":
            results_df["latency_ms"].mean()
    }

    return results_df, summary


# ------------------------------------------------------------
# Run Evaluation
# ------------------------------------------------------------

results_df, summary = evaluate_retrieval(
    queries,
    query_embeddings,
    document_embeddings,
    ground_truth,
    top_k=5
)


# ------------------------------------------------------------
# Display Query-Level Results
# ------------------------------------------------------------

print("\nQuery-Level Evaluation")
print("=" * 70)
print(results_df.to_string(index=False))


# ------------------------------------------------------------
# Display Overall Metrics
# ------------------------------------------------------------

print("\nOverall Retrieval Metrics")
print("=" * 70)

for metric, value in summary.items():
    print(f"{metric}: {value:.4f}")


# ------------------------------------------------------------
# Optional: Real Embedding Model Example
# ------------------------------------------------------------
#
# Install:
#     pip install sentence-transformers
#
# Then use:
#
# from sentence_transformers import SentenceTransformer
#
# model = SentenceTransformer("all-MiniLM-L6-v2")
#
# document_embeddings = model.encode(
#     documents,
#     normalize_embeddings=True
# )
#
# query_embeddings = model.encode(
#     queries,
#     normalize_embeddings=True
# )


# ------------------------------------------------------------
# Optional: Compare Multiple Embedding Models
# ------------------------------------------------------------
#
# models = {
#     "MiniLM": "all-MiniLM-L6-v2",
#     "MPNet": "all-mpnet-base-v2"
# }
#
# comparison = []
#
# for model_name, model_path in models.items():
#
#     model = SentenceTransformer(model_path)
#
#     doc_embeddings = model.encode(
#         documents,
#         normalize_embeddings=True
#     )
#
#     query_embeddings = model.encode(
#         queries,
#         normalize_embeddings=True
#     )
#
#     _, metrics = evaluate_retrieval(
#         queries,
#         query_embeddings,
#         doc_embeddings,
#         ground_truth,
#         top_k=5
#     )
#
#     metrics["Embedding Model"] = model_name
#     comparison.append(metrics)
#
# comparison_df = pd.DataFrame(comparison)
#
# print("\nEmbedding Model Comparison")
# print("=" * 70)
# print(comparison_df.to_string(index=False))
