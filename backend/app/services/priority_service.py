"""Deterministic prioritisation. The AI supplies the signals; product rules
decide the priority. These thresholds are starting points to tune against
the labelled benchmark, not universal standards.
"""
import math
from datetime import datetime, timezone

from .. import config

TEAM_BY_AREA = {
    "payments": "Payments Engineering",
    "auth": "Identity & Access",
    "onboarding": "Growth",
    "search": "Search & Discovery",
    "notifications": "Engagement Platform",
    "content": "Content Platform",
    "profile": "Accounts & Settings",
    "core_app": "Core App",
}


def feedback_priority(severity: float, churn: float) -> str:
    """Per-feedback rule: makes the Jev decision dependency explicit."""
    if severity >= 4 and churn >= 0.75:
        return "P0"
    if severity >= 4:
        return "P1"
    if severity >= 3:
        return "P2"
    return "P3"


def issue_score(avg_severity: float, count: int, avg_churn: float, last_seen: datetime) -> float:
    """0-100. Each term is normalised to 0..1 and weighted (config.py).
    Volume uses log scaling so 200 reports isn't 200x more urgent than 1."""
    sev = (avg_severity - 1) / 4
    vol = min(1.0, math.log1p(count) / math.log1p(50))
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - last_seen).total_seconds() / 86400
    rec = math.exp(-days / 7)  # half-life about 5 days
    s = config.W_SEVERITY * sev + config.W_VOLUME * vol + config.W_CHURN * avg_churn + config.W_RECENCY * rec
    return round(100 * s, 1)


def issue_priority(score: float, avg_severity: float, count: int = 1) -> str:
    # P0 = "drop everything", so it needs corroboration. The first run on real Jev
    # output put 44% of issues at P0 under a severity-only rule (alarm fatigue).
    # Severity floor kept: a critical bug hitting even one user is never below P1.
    if score >= 80 or (avg_severity >= 4.5 and count >= 3):
        return "P0"
    if score >= 55 or avg_severity >= 4:
        return "P1"
    if score >= 40:
        return "P2"
    return "P3"


def explain(avg_severity, count, avg_churn, score) -> str:
    return (f"score {score} = {config.W_SEVERITY}×severity({avg_severity:.1f}/5) + "
            f"{config.W_VOLUME}×volume({count}) + {config.W_CHURN}×churn({avg_churn:.0%}) + "
            f"{config.W_RECENCY}×recency")
