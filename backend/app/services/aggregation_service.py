"""Groups individual feedback into recurring product issues.

MVP approach (deterministic, explainable):
1. Hard partition: an entry can only join an open issue with the SAME
   category and product area (both decided by Jev).
2. Within that partition, compare normalised token sets (light stemming +
   a small domain synonym map, e.g. checkout/upi/card -> "pay") against each
   of the issue's member reports. Join the best issue if the max Ochiai
   similarity >= CLUSTER_THRESHOLD, else start a new issue.

Tuned on data/sample_feedback.csv: at 0.15 all 8 payment-failure reports form
one issue. Upgrade path: swap `similarity()` for sentence-embedding cosine.
"""
import re

from .. import config

STOP = set("""a an the and or but if then so to of in on at for with from by is are was
were be been it its this that these those i me my we our you your they them their
he she app just very really still also can not no yes do does did have has had
when while again get got im i'm ive i've dont don't cant can't wont won't would
could should please even only any all some more most much many one every time
tried try use using after before since now like way""".split())

SYNONYMS = {
    # payments
    "payment": "pay", "payments": "pay", "paid": "pay", "checkout": "pay", "upi": "pay",
    "card": "pay", "cards": "pay", "purchase": "pay", "charged": "pay", "debited": "pay",
    "transaction": "pay", "refund": "pay", "billing": "pay",
    # auth
    "otp": "login", "log": "login", "logging": "login", "logged": "login", "password": "login",
    "signin": "login", "locked": "login",
    # stability / errors
    "crashes": "crash", "crashing": "crash", "crashed": "crash",
    "fails": "fail", "failed": "fail", "failure": "fail", "error": "fail", "errors": "fail",
    # content / perf
    "buffering": "buffer", "videos": "video", "playback": "video",
    "loading": "load", "loads": "load", "slow": "slow", "lag": "slow", "lags": "slow",
    "notifications": "notif", "notification": "notif", "subscription": "subscr",
}
MAX_MEMBERS = 30  # token sets kept per issue for comparison


def _stem(w: str) -> str:
    w = SYNONYMS.get(w, w)
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 5 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokens(text: str) -> list[str]:
    return sorted({_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOP})


def similarity(a: set, b: set) -> float:
    return len(a & b) / (len(a) * len(b)) ** 0.5 if a and b else 0.0  # Ochiai


def find_issue(open_issues, category: str, area: str, text: str):
    t = set(tokens(text))
    best, best_sim = None, 0.0
    for issue in open_issues:
        if issue.category != category or issue.product_area != area:
            continue
        members = (issue.keywords or {}).get("members", [])
        s = max((similarity(t, set(m)) for m in members), default=0.0)
        if s > best_sim:
            best, best_sim = issue, s
    return (best, best_sim) if best_sim >= config.CLUSTER_THRESHOLD else (None, best_sim)


def find_issue_semantic(session, open_issues, category: str, area: str, vec):
    """Embedding version: join the issue whose member reports are closest in
    meaning (max cosine), within the same (category, area) partition."""
    import numpy as np

    from ..db import Feedback, FeedbackEmbedding
    cands = {i.id: i for i in open_issues if i.category == category and i.product_area == area}
    if not cands:
        return None, 0.0
    rows = (session.query(Feedback.issue_id, FeedbackEmbedding.vector)
            .join(FeedbackEmbedding, FeedbackEmbedding.feedback_id == Feedback.id)
            .filter(Feedback.issue_id.in_(list(cands))).all())
    best, best_sim = None, 0.0
    for issue_id, v in rows:
        s = float(np.dot(vec, np.asarray(v, dtype=np.float32)))
        if s > best_sim:
            best, best_sim = cands[issue_id], s
    return (best, best_sim) if best_sim >= config.SEMANTIC_THRESHOLD else (None, best_sim)


def merge_keywords(keywords: dict | None, text: str) -> dict:
    members = list((keywords or {}).get("members", []))
    members.append(tokens(text))
    return {"members": members[-MAX_MEMBERS:]}
