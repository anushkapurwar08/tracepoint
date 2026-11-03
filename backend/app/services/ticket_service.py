"""Turns a high-priority issue into an engineering ticket. One ticket per issue
(idempotent), so 20 payment complaints produce 1 ticket, not 20.
If TICKET_WEBHOOK_URL is set (an n8n webhook, or a Jira proxy) the ticket is
POSTed there; otherwise it is stored locally with status 'created-local'."""
import httpx

from .. import config
from ..db import EngineeringTicket, ProductIssue
from .priority_service import TEAM_BY_AREA

AUTO_TICKET_PRIORITIES = {"P0", "P1"}


def upsert_ticket(session, issue: ProductIssue) -> EngineeringTicket | None:
    if issue.priority not in AUTO_TICKET_PRIORITIES:
        return None
    t = session.query(EngineeringTicket).filter_by(issue_id=issue.id).first()
    desc = (f"{issue.summary or ''}\n\nReports: {issue.feedback_count} | Avg severity: "
            f"{issue.avg_severity:.1f}/5 | Avg churn risk: {issue.avg_churn:.0%} | "
            f"Priority score: {issue.priority_score}")
    if t:  # keep ticket in sync as more reports arrive
        t.description, t.priority = desc, issue.priority
        return t
    t = EngineeringTicket(issue_id=issue.id, title=f"[{issue.priority}] {issue.title}",
                          description=desc, assigned_team=TEAM_BY_AREA.get(issue.product_area, "Core App"),
                          priority=issue.priority, ticket_status="created-local")
    session.add(t)
    if config.TICKET_WEBHOOK_URL:
        try:
            r = httpx.post(config.TICKET_WEBHOOK_URL, json={
                "issue_id": issue.id, "title": t.title, "description": desc,
                "team": t.assigned_team, "priority": t.priority}, timeout=10)
            r.raise_for_status()
            t.ticket_status = "sent"
            t.external_ref = (r.json() or {}).get("id") if r.headers.get("content-type", "").startswith("application/json") else None
        except Exception as e:
            t.ticket_status = f"send-failed ({type(e).__name__})"
    return t
