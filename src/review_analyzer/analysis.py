"""Turn a batch of stored reviews into an Analysis.

- Sentiment from the star rating and, when a model has scored them, from the text.
- Complaints (1-2 stars or negative text) are split into sentences and grouped
  by meaning with sentence embeddings, or by keyword category without a model.
- 4-5 star reviews whose text is clearly negative are surfaced as mismatches.
- Per-version stats show whether a release made things better or worse.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from review_analyzer.categories import SUGGESTIONS, categorize
from review_analyzer.clustering import cluster, label_clusters, representatives
from review_analyzer.ml import Embedder, SentimentModel
from review_analyzer.schemas import (
    Analysis,
    Issue,
    IssueReview,
    Review,
    Sentiment,
    SentimentCounts,
    VersionStats,
)

MIN_SENTENCE_LENGTH = 10
MAX_ISSUES = 8
EXAMPLES_PER_ISSUE = 3
# Clusters raised by a single review are noise, not a recurring issue.
MIN_CLUSTER_REVIEWS = 2
# Cosine distance under which sentences join the same cluster. Tuned on live reviews.
CLUSTER_DISTANCE = 0.6
# Share of a cluster's sentences that must match a category for it to inherit the label.
CATEGORY_MAJORITY = 0.4
MISMATCH_MIN_CONFIDENCE = 0.8
MAX_MISMATCHES = 5
MIN_VERSION_REVIEWS = 5
MAX_VERSIONS = 8

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class ComplaintSentence:
    text: str
    review_id: str
    version: str | None


def sentiment_from_rating(rating: int) -> Sentiment:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def split_sentences(text: str) -> list[str]:
    sentences = (" ".join(s.split()) for s in _SENTENCE_SPLIT.split(text))
    return [s for s in sentences if len(s) > MIN_SENTENCE_LENGTH]


def is_complaint(review: Review) -> bool:
    return review.rating <= 2 or review.text_sentiment == "negative"


def is_negative(review: Review) -> bool:
    """Text sentiment when the review has been scored, otherwise the rating."""
    if review.text_sentiment is not None:
        return review.text_sentiment == "negative"
    return review.rating <= 2


def count_sentiment(labels: Iterable[Sentiment]) -> SentimentCounts:
    counts = Counter(labels)
    return SentimentCounts(
        positive=counts["positive"], neutral=counts["neutral"], negative=counts["negative"]
    )


def find_mismatches(reviews: Sequence[Review]) -> list[Review]:
    """High ratings with clearly negative text: complaints the star average hides."""
    candidates = [
        r
        for r in reviews
        if r.rating >= 4
        and r.text_sentiment == "negative"
        and (r.sentiment_score or 0) >= MISMATCH_MIN_CONFIDENCE
    ]
    candidates.sort(key=lambda r: (r.thumbs_up, r.sentiment_score or 0), reverse=True)
    return candidates[:MAX_MISMATCHES]


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def version_stats(reviews: Sequence[Review]) -> list[VersionStats]:
    by_version: dict[str, list[Review]] = defaultdict(list)
    for review in reviews:
        if review.app_version:
            by_version[review.app_version].append(review)

    stats = [
        VersionStats(
            version=version,
            reviews=len(group),
            average_rating=round(mean(r.rating for r in group), 2),
            negative_share=round(sum(map(is_negative, group)) / len(group), 3),
        )
        for version, group in by_version.items()
        if len(group) >= MIN_VERSION_REVIEWS
    ]
    stats.sort(key=lambda s: version_key(s.version))
    return stats[-MAX_VERSIONS:]


def complaint_sentences(
    complaints: Sequence[Review], sentiment_model: SentimentModel | None = None
) -> list[ComplaintSentence]:
    sentences = [
        ComplaintSentence(text, r.review_id, r.app_version)
        for r in complaints
        for text in split_sentences(r.content)
    ]
    if sentiment_model is None or not sentences:
        return sentences
    # Complaints often open with praise ("Love the music, but..."); keep only the complaint.
    predictions = sentiment_model.predict([s.text for s in sentences])
    return [s for s, (label, _) in zip(sentences, predictions, strict=True) if label != "positive"]


def helpfulness_rank(reviews: Sequence[Review]) -> dict[str, int]:
    """Position of each review when sorted most helpful first, then newest first."""
    ordered = sorted(reviews, key=lambda r: (r.thumbs_up, r.created_at), reverse=True)
    return {r.review_id: i for i, r in enumerate(ordered)}


def build_issue(
    label: str,
    sentences: Sequence[ComplaintSentence],
    examples: list[str],
    complaint_count: int,
    rank: Mapping[str, int],
) -> Issue:
    version_by_review = {s.review_id: s.version for s in sentences}
    sentences_by_review: dict[str, list[str]] = defaultdict(list)
    for sentence in sentences:
        sentences_by_review[sentence.review_id].append(sentence.text)
    members = [
        IssueReview(review_id=review_id, sentences=texts)
        for review_id, texts in sorted(
            sentences_by_review.items(), key=lambda item: rank.get(item[0], len(rank))
        )
    ]
    categories = Counter(c for s in sentences if (c := categorize(s.text)))
    category = None
    if categories:
        name, hits = categories.most_common(1)[0]
        if hits / len(sentences) >= CATEGORY_MAJORITY:
            category = name
    return Issue(
        label=label or category or examples[0],
        category=category,
        suggestion=SUGGESTIONS[category] if category else None,
        reviews=len(version_by_review),
        share=round(len(version_by_review) / max(complaint_count, 1), 3),
        examples=examples,
        by_version=dict(Counter(v for v in version_by_review.values() if v)),
        members=members,
    )


def app_name_words(app_id: str) -> set[str]:
    """Words from the package name ("com.spotify.music" -> spotify, music)."""
    return set(re.findall(r"[a-z]{3,}", app_id.lower())) - {"com", "android", "app"}


def clustered_issues(
    app_id: str,
    sentences: Sequence[ComplaintSentence],
    embedder: Embedder,
    complaint_count: int,
    rank: Mapping[str, int],
) -> list[Issue]:
    embeddings = embedder.encode([s.text for s in sentences])
    groups: dict[int, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(cluster(embeddings, CLUSTER_DISTANCE)):
        groups[int(cluster_id)].append(index)

    def review_count(members: list[int]) -> int:
        return len({sentences[i].review_id for i in members})

    kept = [m for m in groups.values() if review_count(m) >= MIN_CLUSTER_REVIEWS]
    kept.sort(key=review_count, reverse=True)
    kept = kept[:MAX_ISSUES]
    labels = label_clusters(
        [" ".join(sentences[i].text for i in m) for m in kept],
        extra_stop_words=app_name_words(app_id),
    )

    return [
        build_issue(
            label,
            [sentences[i] for i in members],
            [sentences[i].text for i in representatives(embeddings, members, EXAMPLES_PER_ISSUE)],
            complaint_count,
            rank,
        )
        for members, label in zip(kept, labels, strict=True)
    ]


def keyword_issues(
    sentences: Sequence[ComplaintSentence], complaint_count: int, rank: Mapping[str, int]
) -> list[Issue]:
    by_category: dict[str, list[ComplaintSentence]] = defaultdict(list)
    for sentence in sentences:
        if category := categorize(sentence.text):
            by_category[category].append(sentence)
    issues = [
        build_issue(
            name, group, [s.text for s in group[:EXAMPLES_PER_ISSUE]], complaint_count, rank
        )
        for name, group in by_category.items()
    ]
    issues.sort(key=lambda i: i.reviews, reverse=True)
    return issues[:MAX_ISSUES]


def analyze(
    app_id: str,
    reviews: Sequence[Review],
    embedder: Embedder | None = None,
    sentiment_model: SentimentModel | None = None,
) -> Analysis:
    complaints = [r for r in reviews if is_complaint(r)]
    sentences = complaint_sentences(complaints, sentiment_model)
    rank = helpfulness_rank(complaints)

    method: Literal["clustering", "keywords"]
    if embedder is not None and len(sentences) >= 2:
        method = "clustering"
        issues = clustered_issues(app_id, sentences, embedder, len(complaints), rank)
    else:
        method = "keywords"
        issues = keyword_issues(sentences, len(complaints), rank)

    scored = [r.text_sentiment for r in reviews if r.text_sentiment is not None]
    return Analysis(
        app_id=app_id,
        review_count=len(reviews),
        rating_sentiment=count_sentiment(sentiment_from_rating(r.rating) for r in reviews),
        text_sentiment=count_sentiment(scored) if scored else None,
        complaint_count=len(complaints),
        issue_method=method,
        issues=issues,
        mismatches=find_mismatches(reviews),
        versions=version_stats(reviews),
    )
