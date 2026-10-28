"""Group complaint sentences by meaning and describe each group."""

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# Words that show up in complaints about any app and say nothing about the problem.
DOMAIN_STOP_WORDS = frozenset(
    """
    app apps application use using used just like really please dont don doesnt doesn cant
    can won im ive time times make makes get gets got want even still now every thing
    things way fix update updated
    """.split()
)
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)


def cluster(embeddings: NDArray[np.float32], distance_threshold: float) -> NDArray[np.intp]:
    """Agglomerative clustering on cosine distance; the number of clusters falls out of the data."""
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=np.intp)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return np.asarray(model.fit_predict(embeddings), dtype=np.intp)


def representatives(embeddings: NDArray[np.float32], members: Sequence[int], n: int) -> list[int]:
    """Indices of the n members closest to the cluster centroid."""
    vectors = embeddings[list(members)]
    centroid = vectors.mean(axis=0)
    similarity = vectors @ centroid
    order = np.argsort(-similarity)
    return [members[i] for i in order[:n]]


def label_clusters(
    documents: Sequence[str], top_n: int = 3, extra_stop_words: Iterable[str] = ()
) -> list[str]:
    """Label each document (all sentences of one cluster) with its most distinctive terms.

    Uses class-based TF-IDF: a term scores high when it is frequent in this
    cluster but rare across the others. Pass the app's own name as extra stop
    words, since it appears in complaints about everything.
    """
    if not documents:
        return []
    vectorizer = CountVectorizer(
        stop_words=sorted({*STOP_WORDS, *extra_stop_words}),
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        counts = vectorizer.fit_transform(documents).toarray().astype(float)
    except ValueError:  # every word was a stop word
        return ["" for _ in documents]

    terms = vectorizer.get_feature_names_out()
    term_frequency = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    average_words = counts.sum() / len(documents)
    idf = np.log1p(average_words / np.maximum(counts.sum(axis=0), 1))
    scores = term_frequency * idf

    labels = []
    for row in scores:
        chosen: list[str] = []
        for index in np.argsort(-row):
            if row[index] <= 0 or len(chosen) == top_n:
                break
            term = str(terms[index])
            # Skip terms that repeat a word already chosen ("ads" after "many ads").
            if any(set(term.split()) & set(c.split()) for c in chosen):
                continue
            chosen.append(term)
        labels.append(", ".join(chosen))
    return labels
