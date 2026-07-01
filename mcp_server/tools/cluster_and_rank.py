"""
cluster_and_rank.py
───────────────────
MCP Tool: cluster_and_rank
Groups a list of text summaries into semantic clusters using TF-IDF + K-Means,
then ranks each cluster by relevance to the original research query using
cosine similarity. Returns clusters sorted by relevance score.
"""

import logging
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def cluster_and_rank(
    summaries: list[str],
    query: str,
    num_clusters: int = 3,
) -> dict[str, Any]:
    """
    Cluster summaries by semantic similarity and rank by relevance to the query.

    Args:
        summaries:    List of text summaries to cluster.
        query:        Original research query for relevance ranking.
        num_clusters: Number of clusters to form (default 3).

    Returns:
        A dict with keys:
          - clusters: list of {cluster_id, relevance_score, summaries, theme}
          - total_summaries (int)
          - num_clusters (int)
          - error (str | None)
    """
    if not summaries:
        return {
            "clusters": [],
            "total_summaries": 0,
            "num_clusters": 0,
            "error": "No summaries provided.",
        }

    # Clamp clusters to number of summaries
    actual_clusters = min(num_clusters, len(summaries))

    if len(summaries) == 1:
        return {
            "clusters": [
                {
                    "cluster_id": 0,
                    "relevance_score": 1.0,
                    "summaries": summaries,
                    "theme": "Single result",
                }
            ],
            "total_summaries": 1,
            "num_clusters": 1,
            "error": None,
        }

    try:
        logger.info(f"[cluster_and_rank] Clustering {len(summaries)} summaries into {actual_clusters} clusters.")

        # ── TF-IDF vectorization ───────────────────────────────────────────────
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
        )
        # Include query in corpus for better representation
        corpus = summaries + [query]
        tfidf_matrix = vectorizer.fit_transform(corpus)

        summary_vectors = tfidf_matrix[:-1]  # all except the query
        query_vector = tfidf_matrix[-1]       # last row = query

        # ── K-Means clustering ────────────────────────────────────────────────
        km = KMeans(n_clusters=actual_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(summary_vectors)

        # ── Rank clusters by centroid-query similarity ────────────────────────
        cluster_info = []
        for cluster_id in range(actual_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_summaries = [summaries[i] for i in cluster_indices]
            centroid = km.cluster_centers_[cluster_id].reshape(1, -1)

            # Cosine similarity between centroid and query vector
            similarity = cosine_similarity(centroid, query_vector.toarray())[0][0]

            # Simple theme extraction: top TF-IDF terms in this cluster
            cluster_text = " ".join(cluster_summaries)
            theme_terms = _extract_top_terms(cluster_text, vectorizer, n=5)

            cluster_info.append(
                {
                    "cluster_id": int(cluster_id),
                    "relevance_score": round(float(similarity), 4),
                    "summaries": cluster_summaries,
                    "theme": ", ".join(theme_terms),
                    "summary_count": len(cluster_summaries),
                }
            )

        # Sort clusters by relevance score (highest first)
        cluster_info.sort(key=lambda x: x["relevance_score"], reverse=True)

        logger.info(f"[cluster_and_rank] Done. Top cluster relevance: {cluster_info[0]['relevance_score'] if cluster_info else 'N/A'}")
        return {
            "clusters": cluster_info,
            "total_summaries": len(summaries),
            "num_clusters": len(cluster_info),
            "error": None,
        }

    except Exception as e:
        logger.exception(f"[cluster_and_rank] Error: {e}")
        return {
            "clusters": [],
            "total_summaries": len(summaries),
            "num_clusters": 0,
            "error": str(e),
        }


def _extract_top_terms(text: str, vectorizer: TfidfVectorizer, n: int = 5) -> list[str]:
    """Extract the top N TF-IDF terms from a text block."""
    try:
        vec = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = vec.toarray()[0]
        top_indices = scores.argsort()[::-1][:n]
        return [feature_names[i] for i in top_indices if scores[i] > 0]
    except Exception:
        return []
