"""Known complaint categories, each with a suggested fix.

Clusters are labeled with the category most of their sentences match. When no
embedding model is available, these keywords are the whole issue detector.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    name: str
    keywords: tuple[str, ...]
    suggestion: str


CATEGORIES = (
    Category(
        "Interface",
        ("interface", "design", "navigation", "ui", "ux", "layout", "redesign"),
        "Improve interface consistency and streamline navigation.",
    ),
    Category(
        "Performance",
        (
            "slow",
            "lag",
            "laggy",
            "lagging",
            "crash",
            "freeze",
            "freezing",
            "bug",
            "buggy",
            "glitch",
            "glitchy",
            "performance",
            "battery",
        ),
        "Optimize loading times and fix reported bugs/crashes.",
    ),
    Category(
        "Security",
        ("security", "privacy", "hack", "leak", "scam"),
        "Enhance user data protection and address privacy concerns.",
    ),
    Category(
        "Size",
        ("size", "storage", "space"),
        "Reduce app size by optimizing assets and libraries.",
    ),
    Category(
        "Ads",
        ("ad", "ads", "advert", "advertisement", "commercial", "pop-up", "popup"),
        "Reduce intrusive ads or offer an ad-free option.",
    ),
    Category(
        "Login",
        (
            "login",
            "log in",
            "logged out",
            "signup",
            "sign up",
            "account",
            "verification",
            "otp",
            "password",
        ),
        "Simplify authentication and fix signup/login flows.",
    ),
    Category(
        "Pricing",
        ("price", "subscription", "premium", "paywall", "refund", "charged", "expensive"),
        "Review pricing and make the free tier's limits clear.",
    ),
)
SUGGESTIONS = {c.name: c.suggestion for c in CATEGORIES}

# Whole-word match with simple inflections, so "ui" doesn't match "build"
# and "ads" doesn't match "loads".
_PATTERNS = {
    c.name: re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in c.keywords) + r")(?:s|es|ed|ing)?\b",
        re.IGNORECASE,
    )
    for c in CATEGORIES
}


def categorize(sentence: str) -> str | None:
    """Return the first category whose keywords appear in the sentence."""
    for name, pattern in _PATTERNS.items():
        if pattern.search(sentence):
            return name
    return None
