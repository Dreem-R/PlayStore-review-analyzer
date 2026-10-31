"""Streamlit Community Cloud entry point for the public demo.

Community Cloud runs a single Streamlit process, so the UI runs the analysis
in-process instead of calling the API. It installs this folder's requirements.txt
(exported from uv.lock) but not the project itself, so `src` goes on the path:
pushing new code updates the app without reinstalling dependencies.
"""

import json
import logging
import os
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[2] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Public-demo settings. The scraper is unofficial, so the demo re-uses fetches
# for 6 hours, caps how much it fetches, and preloads apps visitors are likely to try.
DEMO_SETTINGS = {
    # Community Cloud's disk is wiped on restart; /tmp is always writable.
    "REVIEW_ANALYZER_DATABASE_URL": "sqlite:////tmp/review_analyzer.db",
    "REVIEW_ANALYZER_SYNC_TTL_MINUTES": "360",
    "REVIEW_ANALYZER_MAX_SYNC_COUNT": "500",
    "REVIEW_ANALYZER_MAX_SYNCS_PER_HOUR": "30",
    "REVIEW_ANALYZER_PRELOAD_COUNT": "200",
    "REVIEW_ANALYZER_PRELOAD_APPS": json.dumps(
        [
            "com.spotify.music",
            "com.whatsapp",
            "com.instagram.android",
            "com.duolingo",
            "com.netflix.mediaclient",
            "com.ubercab",
            "com.discord",
            "com.king.candycrushsaga",
        ]
    ),
}
for key, value in DEMO_SETTINGS.items():
    os.environ.setdefault(key, value)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from review_analyzer.ui.app import main  # noqa: E402

main()
