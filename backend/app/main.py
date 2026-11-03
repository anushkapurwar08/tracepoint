"""SignalOS API. Run: uvicorn app.main:app --reload  (from backend/)"""
import csv
import io
from collections import Counter
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .db import (AgentRun, AIDecision, Digest, EngineeringTicket, Feedback, FeedbackEmbedding,
                 ProductIssue, SessionLocal, init_db)
from .services import agent_service, embedding_service, llm_service, pipeline, rag_service

app = FastAPI(title="SignalOS", version="0.2.0",
              description="AI product intelligence: feedback → typed decisions → prioritised issues → tickets")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
init_db()
STATIC = Path(__file__).resolve().parents[1] / "static"


def get_db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int | None = Field(None, ge=1, le=25)
    category: str | None = None
    product_area: str | None = None


class FeedbackIn(BaseModel):
    content: str = Field(..., min_length=3, max_length=5000)
    source: str = "api"
    user_id: str | None = None


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok",
            "jev": "live" if config.jev_live() else "mock",
            "gemini": "live" if config.gemini_live() else "template",
            "embeddings": embedding_service.backend_name(),
            "database": config.DATABASE_URL.split(":")[0]}


@app.post("/feedback")
def submit(fb: FeedbackIn, db=Depends(get_db)):
    try:
        return pipeline.process(db, fb.content, fb.source, fb.user_id)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/feedback/bulk")
async def bulk(file: UploadFile = File(...), db=Depends(get_db)):
    """CSV with a `content` column (optional: source, user_id)."""
    text = (await file.read()).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "content" not in rows[0]:
        raise HTTPException(422, "CSV needs a 'content' column")
    ok, errors = 0, []
    for i, r in enumerate(rows):
        try:
            pipeline.process(db, r["content"], r.get("source") or "csv", r.get("user_id"))
            ok += 1
        except Exception as e:
            db.rollback()
            errors.append({"row": i + 2, "error": str(e)})
    return {"processed": ok, "failed": len(errors), "errors": errors[:20]}


@app.get("/feedback")
def list_feedback(limit: int = 100, issue_id: str | None = None, db=Depends(get_db)):
    q = db.query(Feedback).order_by(Feedback.created_at.desc())
    if issue_id:
        q = q.filter(Feedback.issue_id == issue_id)
    out = []
    for f in q.limit(limit):
        d = f.decision
        out.append({"id": f.id, "content": f.content, "source": f.source, "created_at": f.created_at,
                    "issue_id": f.issue_id, "decision": None if not d else {
                        "category": d.category, "category_confidence": d.category_confidence,
                        "product_area": d.product_area, "severity": d.severity,
                        "churn_probability": d.churn_probability, "priority": d.priority,
                        "needs_review": bool(d.needs_review), "model_used": d.model_used,
                        "latency_ms": d.latency_ms}})
    return out


@app.get("/issues")
def list_issues(status: str = "open", priority: str | None = None, db=Depends(get_db)):
    q = db.query(ProductIssue).filter(ProductIssue.status == status)
    if priority:
        q = q.filter(ProductIssue.priority == priority)
    tickets = {t.issue_id: t for t in db.query(EngineeringTicket)}
    res = []
    for i in q.order_by(ProductIssue.priority_score.desc()):
        t = tickets.get(i.id)
        res.append({"id": i.id, "title": i.title, "category": i.category, "product_area": i.product_area,
                    "feedback_count": i.feedback_count, "avg_severity": round(i.avg_severity, 2),
                    "avg_churn": round(i.avg_churn, 2), "priority_score": i.priority_score,
                    "priority": i.priority, "summary": i.summary, "status": i.status,
                    "ticket": None if not t else {"id": t.id, "team": t.assigned_team, "status": t.ticket_status}})
    return res


@app.patch("/issues/{issue_id}")
def update_issue(issue_id: str, status: str, db=Depends(get_db)):
    i = db.get(ProductIssue, issue_id)
    if not i:
        raise HTTPException(404, "issue not found")
    i.status = status
    db.commit()
    return {"id": i.id, "status": i.status}


@app.post("/ask")
def ask(q: AskIn, db=Depends(get_db)):
    """Grounded Q&A over feedback (RAG). Answers cite feedback tags; citations are verified."""
    return rag_service.answer(db, q.question, q.k, q.category, q.product_area)


def _run_json(r: AgentRun) -> dict:
    return {"id": r.id, "issue_id": r.issue_id, "status": r.status, "mode": r.mode, "model": r.model,
            "steps": r.steps, "verification": r.verification, "report": r.report,
            "duration_ms": r.duration_ms, "created_at": r.created_at}


@app.post("/issues/{issue_id}/investigate")
def investigate(issue_id: str, db=Depends(get_db)):
    """Run the Investigator agent on one issue (read-only tools, Jev-verified, step-capped)."""
    try:
        return _run_json(agent_service.investigate(db, issue_id))
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/investigations")
def list_investigations(issue_id: str | None = None, db=Depends(get_db)):
    q = db.query(AgentRun).order_by(AgentRun.created_at.desc())
    if issue_id:
        q = q.filter(AgentRun.issue_id == issue_id)
    return [_run_json(r) for r in q.limit(20)]


@app.get("/tickets")
def list_tickets(db=Depends(get_db)):
    return [{"id": t.id, "issue_id": t.issue_id, "title": t.title, "team": t.assigned_team,
             "priority": t.priority, "status": t.ticket_status, "description": t.description}
            for t in db.query(EngineeringTicket).order_by(EngineeringTicket.created_at.desc())]


def _stats(db) -> dict:
    decs = db.query(AIDecision).all()
    issues = db.query(ProductIssue).filter(ProductIssue.status == "open").all()
    return {
        "total_feedback": db.query(Feedback).count(),
        "open_issues": len(issues),
        "critical_issues": sum(i.priority in ("P0", "P1") for i in issues),
        "needs_review": sum(d.needs_review for d in decs),
        "by_category": dict(Counter(d.category for d in decs)),
        "by_area": dict(Counter(d.product_area for d in decs)),
        "models_used": dict(Counter((d.model_used or "").split(" ")[0] for d in decs)),
        "avg_latency_ms": round(sum(d.latency_ms or 0 for d in decs) / len(decs), 1) if decs else 0,
    }


@app.get("/stats")
def stats(db=Depends(get_db)):
    return _stats(db)


@app.post("/digest")
def make_digest(db=Depends(get_db)):
    """Called daily by the n8n digest workflow."""
    s = _stats(db)
    top = list_issues(db=db)[:5]
    text, model = llm_service.daily_digest(s, top)
    d = Digest(content=text, model_used=model)
    db.add(d)
    db.commit()
    return {"id": d.id, "content": text, "model_used": model, "created_at": d.created_at}


@app.get("/digest/latest")
def latest_digest(db=Depends(get_db)):
    d = db.query(Digest).order_by(Digest.created_at.desc()).first()
    return d and {"content": d.content, "model_used": d.model_used, "created_at": d.created_at}


@app.post("/admin/reset")
def reset(db=Depends(get_db)):
    """Demo helper: wipe all data."""
    for m in (AgentRun, EngineeringTicket, AIDecision, FeedbackEmbedding, Feedback, ProductIssue, Digest):
        db.query(m).delete()
    db.commit()
    return {"reset": True}
