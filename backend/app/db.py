"""SQLAlchemy models. Mirrors sql/schema.sql (the Supabase version)."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def _id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(String, primary_key=True, default=_id)
    content = Column(Text, nullable=False)
    source = Column(String, default="api")
    user_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    issue_id = Column(String, ForeignKey("product_issues.id"), nullable=True)
    decision = relationship("AIDecision", uselist=False, back_populates="feedback")


class AIDecision(Base):
    __tablename__ = "ai_decisions"
    id = Column(String, primary_key=True, default=_id)
    feedback_id = Column(String, ForeignKey("feedback.id"), nullable=False)
    category = Column(String)
    category_confidence = Column(Float)
    category_probs = Column(JSON)
    severity = Column(Float)            # expected value on 1..5
    severity_confidence = Column(Float)
    severity_probs = Column(JSON)
    churn_probability = Column(Float)   # Noul P(yes)
    product_area = Column(String)
    priority = Column(String)           # P0..P3 (per-feedback rule)
    needs_review = Column(Integer, default=0)  # low-confidence flag for humans
    model_used = Column(String)         # "jev-latest" | "mock-heuristic" | ...
    latency_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), default=_now)
    feedback = relationship("Feedback", back_populates="decision")


class ProductIssue(Base):
    __tablename__ = "product_issues"
    id = Column(String, primary_key=True, default=_id)
    title = Column(String)
    category = Column(String)
    product_area = Column(String)
    feedback_count = Column(Integer, default=0)
    avg_severity = Column(Float, default=0)
    avg_churn = Column(Float, default=0)
    priority_score = Column(Float, default=0)
    priority = Column(String, default="P3")
    keywords = Column(JSON)             # centroid tokens used for clustering
    summary = Column(Text)
    status = Column(String, default="open")
    last_seen = Column(DateTime(timezone=True), default=_now)
    created_at = Column(DateTime(timezone=True), default=_now)


class EngineeringTicket(Base):
    __tablename__ = "engineering_tickets"
    id = Column(String, primary_key=True, default=_id)
    issue_id = Column(String, ForeignKey("product_issues.id"), unique=True)
    title = Column(String)
    description = Column(Text)
    assigned_team = Column(String)
    priority = Column(String)
    ticket_status = Column(String, default="created")
    external_ref = Column(String)       # id returned by Jira/n8n, if any
    created_at = Column(DateTime(timezone=True), default=_now)


class FeedbackEmbedding(Base):
    """One vector per feedback entry. In Supabase this becomes a pgvector column."""
    __tablename__ = "feedback_embeddings"
    feedback_id = Column(String, ForeignKey("feedback.id"), primary_key=True)
    vector = Column(JSON)  # list[float]
    model = Column(String)


class AgentRun(Base):
    """One Investigator agent run: full step trace + final report, for audit."""
    __tablename__ = "agent_runs"
    id = Column(String, primary_key=True, default=_id)
    issue_id = Column(String, ForeignKey("product_issues.id"))
    status = Column(String)
    mode = Column(String)           # "agent" | "scripted (not agentic...)"
    model = Column(String)
    steps = Column(JSON)            # [{thought, tool, args, result}]
    verification = Column(JSON)     # last verify_hypothesis result
    report = Column(JSON)           # submitted root-cause brief
    duration_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), default=_now)


class Digest(Base):
    __tablename__ = "digests"
    id = Column(String, primary_key=True, default=_id)
    content = Column(Text)
    model_used = Column(String)
    created_at = Column(DateTime(timezone=True), default=_now)


def init_db():
    Base.metadata.create_all(engine)
