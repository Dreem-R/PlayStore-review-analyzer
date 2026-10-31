"""Check that the Play Store client still works against the live store.

google-play-scraper is unofficial and breaks when Google changes its pages, so
the scraper-health workflow runs this daily and fails (emailing the repo owner)
before a demo visitor finds out.

Usage: python scripts/check_scraper.py
"""

import sys

from review_analyzer.scraper import GooglePlayClient

SEARCH = ("spotify", "com.spotify.music")
APPS = ("com.spotify.music", "com.whatsapp")
REVIEWS = 20


def main() -> int:
    client = GooglePlayClient()
    failures = []

    query, expected = SEARCH
    found = [app.app_id for app in client.search(query, limit=5)]
    print(f"search {query!r}: {found}")
    if expected not in found:
        failures.append(f"search {query!r} didn't return {expected}")

    for app_id in APPS:
        print(f"lookup {app_id}: {client.lookup(app_id).title}")
        reviews = client.fetch_reviews(app_id, REVIEWS)
        print(f"reviews {app_id}: {len(reviews)}")
        if len(reviews) < REVIEWS // 2:
            failures.append(f"{app_id}: only {len(reviews)} of {REVIEWS} reviews")

    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
