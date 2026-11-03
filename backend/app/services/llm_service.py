"""Gemini: the generative half of the system.

Jev decides (category, severity, churn). Gemini writes (issue titles,
summaries, ticket descriptions, the daily digest). Gemini is also used as the
baseline classifier in eval/benchmark.py, so we can measure when a specialised
decision model beats a general LLM, rather than assuming it.
"""
import json
import time

import httpx

from .. import config

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def post_with_retry(body: dict, timeout: int = 40, retries: int = 4) -> dict:
    """POST to Gemini, retrying rate limits (429) and overloads (500/503) with
    exponential backoff. Free-tier keys allow only a few requests per minute,
    so bulk uploads and multi-step agent runs hit this routinely."""
    for attempt in range(retries + 1):
        r = httpx.post(GEMINI_URL.format(model=config.GEMINI_MODEL),
                       headers={"x-goog-api-key": config.GEMINI_API_KEY}, json=body, timeout=timeout)
        if r.status_code in (429, 500, 503) and attempt < retries:
            wait = float(r.headers.get("retry-after") or 0) or min(30, 2 * 2 ** attempt)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()


def _call(prompt: str, json_mode: bool = False, temperature: float = 0.2) -> str:
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    data = post_with_retry(body)
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def generate(prompt: str, fallback: str, json_mode: bool = False) -> tuple[str, str]:
    """Returns (text, model_used). Never raises: the product keeps working
    with a templated fallback if Gemini is down or not configured."""
    if not config.gemini_live():
        return fallback, "template"
    try:
        return _call(prompt, json_mode), config.GEMINI_MODEL
    except Exception as e:
        return fallback, f"template-fallback ({type(e).__name__})"


def issue_title_and_summary(category: str, area: str, samples: list[str]) -> dict:
    samples = samples or ["(no text)"]
    joined = "\n- ".join(s[:300] for s in samples[:8])
    fallback = {
        "title": f"{category.title()} in {area.replace('_', ' ')}: {samples[0][:60]}",
        "summary": f"{len(samples)} user report(s). Representative: \"{samples[0][:160]}\"",
        "suggested_action": f"Triage {area.replace('_', ' ')} {category} reports and reproduce the most common path.",
    }
    prompt = (
        "You are a product analyst. These customer feedback entries were grouped as one product issue "
        f"(category: {category}, area: {area}).\n- {joined}\n\n"
        'Return JSON: {"title": "<= 10 word issue title", "summary": "2 sentences on the pattern", '
        '"suggested_action": "one concrete next step for engineering"}'
    )
    text, model = generate(prompt, json.dumps(fallback), json_mode=True)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = fallback
    out["model_used"] = model
    return out


def daily_digest(stats: dict, top_issues: list[dict]) -> tuple[str, str]:
    lines = [f"- [{i['priority']}] {i['title']} — {i['feedback_count']} reports, "
             f"avg severity {i['avg_severity']:.1f}, churn {i['avg_churn']:.0%}" for i in top_issues]
    fallback = (
        f"SignalOS digest: {stats['total_feedback']} feedback entries, {stats['open_issues']} open issues, "
        f"{stats['critical_issues']} critical (P0/P1).\nTop issues:\n" + "\n".join(lines)
    )
    prompt = (
        "Write a concise end-of-day product intelligence digest for a product team (max 150 words). "
        "Lead with what needs action today, then trends. Do not invent numbers.\n\n"
        f"Stats: {json.dumps(stats)}\nTop issues:\n" + "\n".join(lines)
    )
    return generate(prompt, fallback)


def gemini_classify(text: str) -> dict:
    """Baseline classifier for the benchmark. Same answer space as Jev."""
    prompt = (
        "Classify this customer feedback. Return JSON only: "
        '{"category": one of ["bug","ux","feature","performance","other"], '
        '"severity": integer 1-5 (1 minor, 5 critical functionality unavailable), '
        '"churn_risk": probability 0-1 that the customer will stop using the product}\n\n'
        f"Feedback: {text}"
    )
    start = time.perf_counter()
    raw = _call(prompt, json_mode=True, temperature=0)
    out = json.loads(raw)
    out["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return out
