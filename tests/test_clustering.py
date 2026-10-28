import numpy as np

from review_analyzer.clustering import cluster, label_clusters, representatives


def unit(*rows: list[float]) -> np.ndarray:
    vectors = np.array(rows, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_cluster_separates_distinct_directions() -> None:
    embeddings = unit([1, 0.1, 0], [1, 0, 0.1], [0, 1, 0.1], [0.1, 1, 0])
    labels = cluster(embeddings, distance_threshold=0.3)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_threshold_controls_granularity() -> None:
    embeddings = unit([1, 0.1, 0], [1, 0, 0.1], [0, 1, 0.1], [0.1, 1, 0])
    assert len(set(cluster(embeddings, distance_threshold=2.0))) == 1


def test_cluster_handles_single_sentence() -> None:
    assert list(cluster(unit([1, 0]), distance_threshold=0.5)) == [0]


def test_representatives_are_closest_to_centroid() -> None:
    embeddings = unit([1, 0], [0.9, 0.1], [0.5, 0.5], [0, 1])
    assert representatives(embeddings, [0, 1, 2], n=2) == [1, 0]


def test_label_clusters_picks_distinctive_terms() -> None:
    labels = label_clusters(
        [
            "Too many ads. Ads after every song. The ads are loud.",
            "The app crashes on launch. Crashes when I open settings.",
        ]
    )
    assert labels[0].startswith("ads")
    assert labels[1].startswith("crashes")
    # Stop words ("the", "app") never make it into a label.
    assert "app" not in labels[1].split(", ")


def test_label_clusters_does_not_repeat_words() -> None:
    [label] = label_clusters(["too many ads, many ads, many ads everywhere"])
    words = [w for term in label.split(", ") for w in term.split()]
    assert len(words) == len(set(words))


def test_label_clusters_drops_extra_stop_words() -> None:
    [label] = label_clusters(
        ["spotify keeps crashing, spotify crashing"], extra_stop_words={"spotify"}
    )
    assert "spotify" not in label


def test_label_clusters_with_only_stop_words() -> None:
    assert label_clusters(["it is what it is"]) == [""]
