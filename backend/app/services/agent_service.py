"""Investigator agent: root-cause analysis for one product issue.

Why an agent HERE (and a fixed workflow everywhere else):
triage is high-volume and must be predictable, so it is a deterministic
pipeline. Investigating a P0 is open-ended: the next step depends on what the
last one found ("spike after the update? -> check versions -> mostly Android 14
-> search Android 14 payment reports"). That is the job agents are for.

Loop (Gemini function calling):
    goal + tools ─► Gemini picks a tool ─► we run it (read-only) ─► result back
                 ◄────────────────────────── repeat, max MAX_STEPS ───────────┘
    ... ─► verify_hypothesis (Jev noul: does the evidence support the claim?)
        ─► submit_report   (only accepted after a passing verification)

Guardrails:
- Tools are READ-ONLY. The agent can recommend a ticket; a human creates it.
- Hard step cap; every step (reasoning, tool, args, result) is stored as a trace.
- Generator/verifier split: Gemini proposes the root cause, Jev scores whether
  the cited feedback supports it. The report is rejected below SUPPORT_MIN.
- Without a Gemini key, a fixed scripted investigation runs instead and is
  labelled "scripted (not agentic)", never presented as the agent.
"""
import json
import re
import time
from collections import Counter

import httpx
import numpy as np

from .. import config
from ..db import AgentRun, Feedback, FeedbackEmbedding, ProductIssue
from . import embedding_service, jev_service, llm_service, rag_service

MAX_STEPS = 8
SUPPORT_MIN = 0.6

SYSTEM = """You are SignalOS Investigator, a senior product analyst agent.
Goal: find the most likely ROOT CAUSE of one product issue, using only evidence from tools.
Work like an investigator: look at the issue, form a hypothesis, test it with targeted tools
(mentions of devices/versions/payment methods, timeline, related issues, targeted searches),
revise if the evidence disagrees. Prefer specific, testable causes ("UPI payments fail on
Android 14 after the latest update") over vague ones ("payments are buggy").
Before submitting you MUST call verify_hypothesis with your hypothesis and the feedback ids
that support it. Then call submit_report. Cite feedback ids exactly as tools return them
(e.g. F-1a2b3c4d). Never invent numbers, devices or ids. Be efficient: you have at most
{max_steps} tool calls. Write one short sentence of reasoning before each tool call."""

# ----------------------------- tool schemas ---------------------------------

def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "OBJECT", "properties": props, "required": required}


S = {"type": "STRING"}
TOOLS = [
    {"name": "get_issue_details",
     "description": "Stats, summary and up to 12 member reports (with ids) for the issue under investigation.",
     "parameters": _obj({"issue_id": S}, ["issue_id"])},
    {"name": "extract_mentions",
     "description": "Counts concrete mentions inside the issue's reports: platforms/OS versions, devices, "
                    "app updates, payment methods, error messages, repeat attempts.",
     "parameters": _obj({"issue_id": S}, ["issue_id"])},
    {"name": "feedback_timeline",
     "description": "Report counts for the issue by day and by source channel. Use to detect spikes.",
     "parameters": _obj({"issue_id": S}, ["issue_id"])},
    {"name": "search_feedback",
     "description": "Semantic search over ALL feedback (not only this issue). Use to test a hypothesis, "
                    "e.g. 'UPI payment failing after update'. Returns ids, quotes and their issue.",
     "parameters": _obj({"query": S, "k": {"type": "INTEGER"}, "product_area": S}, ["query"])},
    {"name": "find_related_issues",
     "description": "Other open issues most similar in meaning to this one, across all product areas. "
                    "Use to check whether the cause spans areas (e.g. a bad release).",
     "parameters": _obj({"issue_id": S}, ["issue_id"])},
    {"name": "verify_hypothesis",
     "description": "Independent check by the Jev decision model: probability (0-1) that the given "
                    "feedback directly supports the hypothesis. Required before submit_report.",
     "parameters": _obj({"hypothesis": S, "feedback_ids": {"type": "ARRAY", "items": S}},
                        ["hypothesis", "feedback_ids"])},
    {"name": "submit_report",
     "description": "Final root-cause brief. Only accepted after verify_hypothesis.",
     "parameters": _obj({
         "root_cause": S,
         "confidence": {"type": "STRING", "enum": ["low", "medium", "high"]},
         "affected_segment": S,
         "evidence_ids": {"type": "ARRAY", "items": S},
         "recommended_actions": {"type": "ARRAY", "items": S},
         "open_questions": {"type": "ARRAY", "items": S},
     }, ["root_cause", "confidence", "evidence_ids", "recommended_actions"])},
]

# ----------------------------- tool implementations -------------------------

def _fid(f: Feedback) -> str:
    return "F-" + f.id[:8]


def _resolve(session, short_ids: list[str]) -> list[Feedback]:
    out = []
    for s in short_ids or []:
        p = s.replace("F-", "").strip()
        if len(p) >= 6:
            f = session.query(Feedback).filter(Feedback.id.like(p + "%")).first()
            if f:
                out.append(f)
    return out


MENTION_PATTERNS = {
    "platform/os": r"\b(android\s?\d{1,2}|ios\s?\d{1,2}|android|iphone|ipad|ios)\b",
    "device": r"\b(samsung|pixel|oneplus|xiaomi|redmi|vivo|oppo|realme|motorola|iphone\s?\d+)\b",
    "app update": r"\b(after (the )?(latest |last |new )?update|new version|latest version|since (the )?update|v?\d+\.\d+(\.\d+)?)\b",
    "payment method": r"\b(upi|credit card|debit card|card|net ?banking|wallet|apple pay|google pay|gpay|paytm|phonepe)\b",
    "error text": r"(error[^.]{0,40}|unknown error|token expired|something went wrong)",
    "repeat attempts": r"\b(every time|again|three times|twice|multiple times|always|keeps)\b",
    "money impact": r"\b(debited|charged twice|refund|money)\b",
}


def _members(session, issue_id: str, limit: int = 200) -> list[Feedback]:
    return session.query(Feedback).filter(Feedback.issue_id == issue_id).limit(limit).all()


def t_get_issue_details(session, issue_id, **_):
    i = session.get(ProductIssue, issue_id)
    if not i:
        return {"error": "unknown issue_id"}
    m = _members(session, issue_id, 12)
    return {"title": i.title, "category": i.category, "product_area": i.product_area,
            "reports": i.feedback_count, "avg_severity": round(i.avg_severity, 2),
            "avg_churn_probability": round(i.avg_churn, 2), "priority": i.priority,
            "priority_score": i.priority_score, "summary": i.summary,
            "sample_reports": [{"id": _fid(f), "text": f.content[:240], "source": f.source} for f in m]}


def t_extract_mentions(session, issue_id, **_):
    m = _members(session, issue_id)
    out = {}
    for label, pat in MENTION_PATTERNS.items():
        c = Counter()
        for f in m:
            for hit in re.findall(pat, f.content.lower()):
                hit = hit[0] if isinstance(hit, tuple) else hit
                c[hit.strip()] += 1
        if c:
            out[label] = dict(c.most_common(6))
    return {"reports_scanned": len(m), "mentions": out or "no concrete mentions found"}


def t_feedback_timeline(session, issue_id, **_):
    m = _members(session, issue_id)
    by_day = Counter(f.created_at.strftime("%Y-%m-%d") for f in m if f.created_at)
    return {"by_day": dict(sorted(by_day.items())), "by_source": dict(Counter(f.source for f in m)),
            "note": "single day only: timeline cannot show a trend" if len(by_day) <= 1 else ""}


def t_search_feedback(session, query, k=6, product_area=None, **_):
    hits = rag_service.retrieve(session, query, min(int(k or 6), 10), None, product_area or None)
    out = []
    for h in hits:
        f = session.get(Feedback, h["feedback_id"])
        out.append({"id": _fid(f), "text": h["content"][:200], "similarity": h["score"],
                    "area": h["product_area"], "in_issue": f.issue_id[:8] if f.issue_id else None})
    return {"results": out}


def t_find_related_issues(session, issue_id, **_):
    def centroid(iid):
        vs = (session.query(FeedbackEmbedding.vector).join(Feedback, Feedback.id == FeedbackEmbedding.feedback_id)
              .filter(Feedback.issue_id == iid).all())
        if not vs:
            return None
        c = np.mean(np.asarray([v[0] for v in vs], dtype=np.float32), axis=0)
        n = np.linalg.norm(c)
        return c / n if n else c
    me = centroid(issue_id)
    if me is None:
        return {"error": "issue has no embeddings"}
    res = []
    for other in session.query(ProductIssue).filter(ProductIssue.status == "open", ProductIssue.id != issue_id):
        c = centroid(other.id)
        if c is not None:
            res.append((float(me @ c), other))
    res.sort(key=lambda x: -x[0])
    return {"related": [{"issue_id": o.id, "title": o.title, "area": o.product_area, "reports": o.feedback_count,
                         "priority": o.priority, "similarity": round(s, 3)} for s, o in res[:5]]}


def t_verify_hypothesis(session, hypothesis, feedback_ids, **_):
    fbs = _resolve(session, feedback_ids)
    unknown = _unknown_ids(session, feedback_ids)
    if not fbs:
        return {"error": "none of the feedback_ids exist; cite ids returned by tools", "unknown_ids": unknown}
    evidence = [f.content[:300] for f in fbs[:10]]
    start = time.perf_counter()
    if config.jev_live():
        try:
            r = httpx.post(config.JEV_URL, headers={"Authorization": f"Bearer {config.TYPESAFE_API_KEY}"},
                           json={"model": config.JEV_MODEL,
                                 "state": {"hypothesis": hypothesis, "customer_feedback": evidence},
                                 "questions": {"supported": {
                                     "type": "noul",
                                     "instructions": "Does the customer feedback directly support the hypothesis, "
                                                     "including its specific details (platform, version, method), "
                                                     "rather than merely being on the same topic?"}}},
                           timeout=30)
            r.raise_for_status()
            p, verifier = float(r.json()["answers"]["supported"]["noul"]), r.json().get("model", "jev")
        except Exception as e:
            p, verifier = _lexical_support(hypothesis, evidence), f"lexical-fallback ({type(e).__name__})"
    else:
        p, verifier = _lexical_support(hypothesis, evidence), "lexical-fallback (no Jev key)"
    return {"support_probability": round(p, 3), "verifier": verifier, "evidence_used": len(evidence),
            "unknown_ids": unknown,
            "passed": p >= SUPPORT_MIN, "latency_ms": round((time.perf_counter() - start) * 1000),
            "guidance": "" if p >= SUPPORT_MIN else
            "Evidence does not support this hypothesis well enough. Narrow or revise it, or find better evidence."}


def _unknown_ids(session, short_ids) -> list[str]:
    known = {_fid(f) for f in _resolve(session, short_ids)}
    return [x for x in (short_ids or []) if ("F-" + x.replace("F-", "").strip()[:8]) not in known]


def _lexical_support(hypothesis: str, evidence: list[str]) -> float:
    h = set(re.findall(r"[a-z0-9]+", hypothesis.lower())) - {"the", "a", "on", "in", "of", "and", "to", "is", "users", "when"}
    if not h:
        return 0.0
    covered = set()
    for e in evidence:
        covered |= h & set(re.findall(r"[a-z0-9]+", e.lower()))
    return len(covered) / len(h)


IMPL = {"get_issue_details": t_get_issue_details, "extract_mentions": t_extract_mentions,
        "feedback_timeline": t_feedback_timeline, "search_feedback": t_search_feedback,
        "find_related_issues": t_find_related_issues, "verify_hypothesis": t_verify_hypothesis}

# ----------------------------- the loop -------------------------------------


def _gemini(contents: list) -> dict:
    body = {"systemInstruction": {"parts": [{"text": SYSTEM.format(max_steps=MAX_STEPS)}]},
            "contents": contents,
            "tools": [{"functionDeclarations": TOOLS}],
            "generationConfig": {"temperature": 0.2}}
    return llm_service.post_with_retry(body, timeout=60)["candidates"][0]["content"]


def _summarise(result: dict) -> str:
    s = json.dumps(result, default=str)
    return s if len(s) < 600 else s[:600] + "…"


def investigate(session, issue_id: str, llm=None) -> AgentRun:
    """Run the agent on one issue. `llm` lets tests inject a fake model."""
    issue = session.get(ProductIssue, issue_id)
    if not issue:
        raise ValueError("issue not found")
    llm = llm or (_gemini if config.gemini_live() else None)
    run = AgentRun(issue_id=issue_id, status="running", steps=[],
                   mode="agent" if llm else "scripted (not agentic: no Gemini key)",
                   model=config.GEMINI_MODEL if llm is _gemini else ("injected" if llm else "none"))
    session.add(run)
    session.commit()
    steps, report, verified, fabricated = [], None, None, []
    t0 = time.perf_counter()
    try:
        if llm is None:
            report, verified = _scripted(session, issue, steps)
        else:
            contents = [{"role": "user", "parts": [{"text":
                        f"Investigate issue {issue_id}: \"{issue.title}\" ({issue.feedback_count} reports, "
                        f"{issue.priority}). Find the root cause."}]}]
            for _ in range(MAX_STEPS + 2):  # +2 lets it recover from a rejected submit
                content = llm(contents)
                contents.append(content)
                parts = content.get("parts", [])
                thought = " ".join(p["text"] for p in parts if p.get("text")).strip()
                calls = [p["functionCall"] for p in parts if p.get("functionCall")]
                if not calls:
                    steps.append({"thought": thought, "tool": None, "note": "model stopped without submitting"})
                    break
                responses = []
                for call in calls:
                    name, args = call["name"], call.get("args", {}) or {}
                    tool_calls = sum(1 for s in steps if s.get("tool"))
                    if tool_calls >= MAX_STEPS and name != "submit_report":
                        result = {"error": f"step limit {MAX_STEPS} reached: call submit_report now"}
                    elif name == "submit_report":
                        bad = _unknown_ids(session, args.get("evidence_ids"))
                        if bad:
                            fabricated.extend(bad)
                            result = {"error": f"rejected: evidence_ids not found {bad}. Cite only ids returned by tools."}
                        elif not verified or not verified["passed"]:
                            result = {"error": "rejected: run verify_hypothesis and get passed=true first"}
                        else:
                            report = args
                            result = {"accepted": True}
                    elif name in IMPL:
                        if name == "get_issue_details" or "issue_id" in args:
                            args["issue_id"] = issue_id  # agent may only investigate its own issue
                        result = IMPL[name](session, **args)
                        if name == "verify_hypothesis":
                            verified = {**result, "hypothesis": args.get("hypothesis")}
                            fabricated.extend(result.get("unknown_ids", []))
                    else:
                        result = {"error": f"unknown tool {name}"}
                    steps.append({"thought": thought, "tool": name, "args": args, "result": _summarise(result)})
                    thought = ""
                    responses.append({"functionResponse": {"name": name, "response": {"result": result}}})
                contents.append({"role": "user", "parts": responses})
                run.steps = list(steps)
                session.commit()  # trace is visible while the run is in progress
                if report:
                    break
        run.status = "completed" if report else "incomplete"
    except Exception as e:
        run.status = f"failed ({type(e).__name__}: {str(e)[:120]})"
    run.steps, run.report = steps, report
    run.verification = {**(verified or {}), "fabricated_ids_caught": fabricated,
                        "verifier_rejections": sum(1 for st in steps if st.get("tool") == "verify_hypothesis"
                                                   and '"passed": false' in st["result"])}
    run.duration_ms = round((time.perf_counter() - t0) * 1000)
    session.commit()
    return run


def _scripted(session, issue, steps):
    """Fixed-order fallback. Same tools, no decisions: shows the difference."""
    def call(name, **args):
        res = IMPL[name](session, **args)
        steps.append({"thought": "(scripted step)", "tool": name, "args": args, "result": _summarise(res)})
        return res
    call("get_issue_details", issue_id=issue.id)
    m = call("extract_mentions", issue_id=issue.id)
    call("feedback_timeline", issue_id=issue.id)
    call("find_related_issues", issue_id=issue.id)
    top = []
    if isinstance(m.get("mentions"), dict):
        for label in ("payment method", "platform/os", "app update", "device"):
            if label in m["mentions"]:
                top.append(next(iter(m["mentions"][label])))
    hyp = f"{issue.category} in {issue.product_area.replace('_', ' ')}" + (f" linked to {', '.join(top[:2])}" if top else "")
    ids = [_fid(f) for f in _members(session, issue.id, 8)]
    v = call("verify_hypothesis", hypothesis=hyp, feedback_ids=ids)
    verified = {**v, "hypothesis": hyp}
    report = {"root_cause": hyp, "confidence": "low", "affected_segment": ", ".join(top) or "unknown",
              "evidence_ids": ids, "recommended_actions": ["Run the agent with a Gemini key for a real investigation."],
              "open_questions": ["Scripted run: no hypothesis was formed or revised."]} if v["passed"] else None
    return report, verified
