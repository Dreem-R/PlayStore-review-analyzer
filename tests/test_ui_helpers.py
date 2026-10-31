from review_analyzer.ui.app import escape_markdown, highlight, review_header


def test_escape_markdown_neutralizes_formatting() -> None:
    assert escape_markdown("Costs $5 *now* #1 [x](y) :(") == (
        r"Costs \$5 \*now\* \#1 \[x\]\(y\) \:\("
    )


def test_highlight_marks_matching_sentences() -> None:
    text = highlight("Love it.  But the ads   are constant! Fix pls", ["But the ads are constant!"])
    assert text == r"Love it\. :orange-background[But the ads are constant\!] Fix pls"


def test_highlight_without_a_match_returns_the_escaped_review() -> None:
    assert highlight("Nothing to see.", ["missing"]) == r"Nothing to see\."


def test_highlight_handles_several_sentences_in_order() -> None:
    text = highlight("Slow. Fine otherwise. Crashes daily.", ["Crashes daily.", "Slow."])
    assert text == (
        r":orange-background[Slow\.] Fine otherwise\. :orange-background[Crashes daily\.]"
    )


def test_review_header() -> None:
    review = {
        "rating": 2,
        "app_version": "3.2.0",
        "created_at": "2026-10-02T10:30:00",
        "thumbs_up": 12,
    }
    assert review_header(review) == "★★☆☆☆ · v3.2.0 · 2026-10-02 · 12 found this helpful"
    review |= {"app_version": None, "thumbs_up": 0}
    assert review_header(review) == "★★☆☆☆ · 2026-10-02"
