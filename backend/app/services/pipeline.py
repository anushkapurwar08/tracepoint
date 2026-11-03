"""Orchestration: feedback -> Jev decision -> aggregation -> priority -> ticket.

    ingest ─► jev_service.decide ─► priority rule (per feedback)
                                   └► aggregation.find_issue ─► update issue stats
                                                              ─► priority_service.issue_score
                                                              ─► llm_service title/summary (new issues)
                                                              ─► ticket_service.upsert_ticket
"""
from datetime import datetime, timezone

from ..db import AIDecision, Feedback, FeedbackEmbedding, ProductIssue
from . import aggregation_service as agg
from . import embedding_service, jev_service, llm_service, priority_service, ticket_service

REVIEW_CONFIDENCE = 0.55  # below this, flag for human review instead of trusting it


def process(session, content: str, source: str = "api", user_id: str | None = None) -> dict:
    content = (content or "").strip()
    if len(content) < 3:
        raise ValueError("feedback too short")
    fb = Feedback(content=content[:5000], source=source, user_id=user_id)
    session.add(fb)
    session.flush()

    vec = embedding_service.embed([content])[0]
    session.add(FeedbackEmbedding(feedback_id=fb.id, vector=vec.round(5).tolist(),
                                  model=embedding_service.backend_name()))
    semantic = embedding_service.backend_name() != "hashed-bow-fallback"

    d = jev_service.decide(content)
    prio = priority_service.feedback_priority(d["severity"], d["churn_probability"])
    conf = d.get("category_confidence")
    dec = AIDecision(feedback_id=fb.id, priority=prio,
                     needs_review=int(conf is not None and conf < REVIEW_CONFIDENCE),
                     **{k: d.get(k) for k in ["category", "category_confidence", "category_probs",
                                              "severity", "severity_confidence", "severity_probs",
                                              "churn_probability", "product_area", "model_used", "latency_ms"]})
    session.add(dec)

    # Aggregate. Pure praise ("other" with low severity) doesn't become an issue.
    issue = None
    if not (d["category"] == "other" and d["severity"] < 2.5):
        open_issues = session.query(ProductIssue).filter(ProductIssue.status == "open").all()
        if semantic:
            issue, _sim = agg.find_issue_semantic(session, open_issues, d["category"], d["product_area"], vec)
        else:
            issue, _sim = agg.find_issue(open_issues, d["category"], d["product_area"], content)
        if issue is None:
            issue = ProductIssue(category=d["category"], product_area=d["product_area"],
                                 feedback_count=0, avg_severity=0, avg_churn=0, keywords={})
            session.add(issue)
        n = issue.feedback_count
        issue.avg_severity = (issue.avg_severity * n + d["severity"]) / (n + 1)
        issue.avg_churn = (issue.avg_churn * n + d["churn_probability"]) / (n + 1)
        issue.feedback_count = n + 1
        issue.keywords = agg.merge_keywords(issue.keywords, content)
        issue.last_seen = datetime.now(timezone.utc)
        issue.priority_score = priority_service.issue_score(issue.avg_severity, issue.feedback_count,
                                                           issue.avg_churn, issue.last_seen)
        issue.priority = priority_service.issue_priority(issue.priority_score, issue.avg_severity, issue.feedback_count)
        session.flush()
        fb.issue_id = issue.id
        session.flush()  # autoflush is off: make the link visible to the query below

        # Regenerate the title/summary when the issue is born and at a few milestones
        # (keeps Gemini calls bounded: O(log n) per issue, not O(n)).
        if issue.feedback_count in (1, 3, 10, 25, 50):
            samples = [f.content for f in session.query(Feedback).filter_by(issue_id=issue.id).limit(8)]
            meta = llm_service.issue_title_and_summary(issue.category, issue.product_area, samples)
            issue.title = meta.get("title", issue.title)
            issue.summary = f"{meta.get('summary', '')}\nSuggested action: {meta.get('suggested_action', '')}"
        ticket_service.upsert_ticket(session, issue)

    session.commit()
    return {
        "feedback_id": fb.id,
        "decision": {**d, "priority": prio, "needs_review": bool(dec.needs_review)},
        "issue": None if issue is None else {
            "id": issue.id, "title": issue.title, "priority": issue.priority,
            "priority_score": issue.priority_score, "feedback_count": issue.feedback_count,
            "explanation": priority_service.explain(issue.avg_severity, issue.feedback_count,
                                                    issue.avg_churn, issue.priority_score),
        },
    }
