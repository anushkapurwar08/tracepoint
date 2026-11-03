"""Jev (TypeSafe AI) client.

Jev takes a `state` plus typed questions and returns typed answers with
probabilities. Three primitives:
  choice -> one label from a fixed set   (+ per-option probabilities, confidence)
  score  -> a level on an ordered scale  (+ per-level probabilities, confidence)
  noul   -> P(yes) for a statement, 0..1 (no separate confidence)

Questions in one request are evaluated INDEPENDENTLY, so any dependency
between decisions (e.g. "priority depends on severity AND churn") lives in our
backend (priority_service.py), not in the model.

If no TYPESAFE_API_KEY is set, a transparent keyword heuristic stands in so the
whole pipeline is demoable. Every decision records `model_used`, so mock
results are never confused with real ones.
"""
import time

import httpx

from .. import config

CATEGORIES = {
    "bug": "A broken or malfunctioning feature: crashes, errors, things that do not work",
    "ux": "A usability or interface problem: confusing, hard to find, poorly designed",
    "feature": "A request for new functionality that does not exist yet",
    "performance": "A speed, loading, battery or responsiveness issue",
    "other": "Praise, general comments, or anything that does not fit above",
}

SEVERITY_LEVELS = [
    "Minor inconvenience, cosmetic",
    "Low impact, easy workaround",
    "Moderate impact, a workflow is degraded",
    "Major functionality affected for this user",
    "Critical: core functionality (payments, login, data) unavailable",
]

PRODUCT_AREAS = {
    "payments": "Checkout, payments, billing, refunds, subscriptions",
    "auth": "Login, signup, password, OTP, account access",
    "onboarding": "First-time setup, tutorials, permissions",
    "search": "Search, filters, discovery, recommendations",
    "notifications": "Push notifications, emails, alerts",
    "content": "Feed, videos, articles, media playback",
    "profile": "Profile, settings, preferences, privacy",
    "core_app": "General app stability, navigation, anything else",
}


def build_request(text: str) -> dict:
    return {
        "model": config.JEV_MODEL,
        "state": {"feedback": text},
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Which product issue category best fits this customer feedback?",
                "criteria": CATEGORIES,
            },
            "product_area": {
                "type": "choice",
                "instructions": "Which part of the product is this feedback about?",
                "criteria": PRODUCT_AREAS,
            },
            "severity": {
                "type": "score",
                "instructions": "Rate the severity of the product issue described, from the customer's point of view.",
                "criteria": SEVERITY_LEVELS,
            },
            "churn_risk": {
                "type": "noul",
                "instructions": "Does this feedback indicate a credible risk that the customer will stop using or uninstall the product?",
            },
        },
    }


# ---------- response parsing (tolerant to small schema differences) ----------

def _first(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _parse_choice(a: dict):
    label = _first(a, "choice", "answer", "value", "label")
    probs = _first(a, "probabilities", "distribution", default={}) or {}
    if isinstance(probs, list):  # [{"option":..,"probability":..}]
        probs = {p.get("option") or p.get("label"): p.get("probability") for p in probs}
    conf = _first(a, "confidence", default=None)
    if label is None and probs:
        label = max(probs, key=probs.get)
    if conf is None and probs and label in probs:
        conf = probs[label]
    return label, conf, probs


def _parse_score(a: dict, n_levels: int):
    raw = _first(a, "score", "expected_value", "value", "answer")
    probs = _first(a, "probabilities", "distribution", default={}) or {}
    conf = _first(a, "confidence", default=None)
    # Normalise to 1..n. Jev may return 0-indexed levels, 1-indexed, or 0..1.
    val = None
    if isinstance(raw, (int, float)):
        val = float(raw)
        if "0" in (a.get("legend") or {}):  # Jev: levels are 0-indexed, legend "0".."n-1"
            val = val + 1
        elif 0 <= val <= 1 and n_levels > 2 and not float(val).is_integer():
            val = 1 + val * (n_levels - 1)
    elif isinstance(raw, str) and raw in SEVERITY_LEVELS:
        val = float(SEVERITY_LEVELS.index(raw) + 1)
    if val is None and isinstance(probs, dict) and probs:
        val = sum((SEVERITY_LEVELS.index(k) + 1 if k in SEVERITY_LEVELS else int(k) + 1) * p
                  for k, p in probs.items())
    return max(1.0, min(float(n_levels), val or 1.0)), conf, probs


def parse_response(body: dict) -> dict:
    answers = body.get("answers", body)
    cat, cat_conf, cat_probs = _parse_choice(answers.get("category", {}))
    area, _, _ = _parse_choice(answers.get("product_area", {}))
    sev, sev_conf, sev_probs = _parse_score(answers.get("severity", {}), len(SEVERITY_LEVELS))
    churn_ans = answers.get("churn_risk", {})
    churn = _first(churn_ans, "noul", "probability", "value", default=0.0)
    return {
        "category": cat or "other",
        "category_confidence": cat_conf,
        "category_probs": cat_probs,
        "product_area": area or "core_app",
        "severity": round(sev, 2),
        "severity_confidence": sev_conf,
        "severity_probs": sev_probs,
        "churn_probability": float(churn),
        "model_used": body.get("model", config.JEV_MODEL),
    }


# ---------- mock heuristic (used only when no API key) ----------

_KW = {
    "bug": ["crash", "error", "broken", "fail", "not working", "doesn't work", "won't", "stuck", "bug", "can't", "cannot", "charged twice", "lost"],
    "performance": ["slow", "lag", "loading", "battery", "freez", "takes forever", "heavy", "hang"],
    "ux": ["confusing", "hard to find", "can't find", "unclear", "cluttered", "too many steps", "hidden", "design", "font", "dark mode is hard"],
    "feature": ["please add", "would love", "wish", "add a", "feature", "option to", "support for", "should have", "it would be great"],
}
_AREA_KW = {
    "payments": ["pay", "checkout", "card", "upi", "refund", "charged", "subscription", "billing", "purchase"],
    "auth": ["login", "log in", "otp", "password", "sign up", "signup", "logged out", "account"],
    "onboarding": ["onboarding", "tutorial", "first time", "setup", "permission"],
    "search": ["search", "filter", "recommend", "discover"],
    "notifications": ["notification", "alert", "email", "spam"],
    "content": ["video", "feed", "play", "article", "story", "buffer"],
    "profile": ["profile", "setting", "privacy", "picture", "dark mode"],
}
_CHURN_KW = ["uninstall", "delete", "switch", "cancel", "never again", "leaving", "moving to", "last time", "done with", "worst", "refund"]


def mock_decide(text: str) -> dict:
    t = text.lower()
    scores = {c: sum(k in t for k in kws) for c, kws in _KW.items()}
    cat = max(scores, key=scores.get) if max(scores.values()) > 0 else "other"
    tot = sum(scores.values()) or 1
    probs = {c: round(scores.get(c, 0) / tot, 2) for c in CATEGORIES} if tot else {}
    area_scores = {a: sum(k in t for k in kws) for a, kws in _AREA_KW.items()}
    area = max(area_scores, key=area_scores.get) if max(area_scores.values()) > 0 else "core_app"
    sev = {"bug": 3.5, "performance": 3.0, "ux": 2.5, "feature": 2.0, "other": 1.5}[cat]
    if area in ("payments", "auth") and cat == "bug":
        sev += 1.2
    if any(k in t for k in ["every time", "always", "again", "three times", "multiple times"]):
        sev += 0.4
    churn_hits = sum(k in t for k in _CHURN_KW)
    churn = min(0.95, 0.1 + 0.35 * churn_hits + (0.15 if sev >= 4 else 0))
    return {
        # A keyword heuristic has no calibrated confidence, so report none rather than a fake one.
        "category": cat, "category_confidence": None, "category_probs": probs,
        "product_area": area, "severity": round(min(5.0, sev), 2),
        "severity_confidence": None, "severity_probs": {},
        "churn_probability": round(churn, 2), "model_used": "mock-heuristic",
    }


def decide(text: str) -> dict:
    """Return typed decisions for one feedback entry. Falls back to the
    heuristic on any API failure and labels it so (never silently)."""
    start = time.perf_counter()
    if not config.jev_live():
        out = mock_decide(text)
    else:
        try:
            r = httpx.post(
                config.JEV_URL,
                headers={"Authorization": f"Bearer {config.TYPESAFE_API_KEY}"},
                json=build_request(text),
                timeout=20,
            )
            r.raise_for_status()
            out = parse_response(r.json())
        except Exception as e:  # network, 4xx/5xx, schema drift
            out = mock_decide(text)
            out["model_used"] = f"mock-fallback ({type(e).__name__})"
    out["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return out
