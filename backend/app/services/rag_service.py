"""Ask-your-feedback: grounded Q&A over customer feedback (RAG).

  question ─► embed ─► top-k feedback (optional category/area filter)
           ─► Gemini answers using ONLY those quotes, citing [F1], [F2]...
           ─► citation check: every cited tag must be one we retrieved

Grounding is measured, not assumed: the response reports how many citations
resolved to retrieved feedback and flags answers with no citations at all.
Without a Gemini key, returns the retrieved evidence with an extractive summary
(no generated claims), so the feature degrades honestly.
"""
import re
from collections import Counter

import numpy as np

from .. import config
from ..db import Feedback, FeedbackEmbedding
from . import embedding_service, llm_service


def retrieve(session, question: str, k: int, category=None, area=None) -> list[dict]:
    q = (session.query(Feedback, FeedbackEmbedding.vector)
         .join(FeedbackEmbedding, FeedbackEmbedding.feedback_id == Feedback.id))
    rows = q.all()
    rows = [(f, v) for f, v in rows if f.decision and
            (category is None or f.decision.category == category) and
            (area is None or f.decision.product_area == area)]
    if not rows:
        return []
    mat = np.asarray([v for _, v in rows], dtype=np.float32)
    qv = embedding_service.embed([question])[0]
    hits = embedding_service.top_k(qv, mat, k)
    out = []
    for n, (i, score) in enumerate(hits, 1):
        f = rows[i][0]
        out.append({"tag": f"F{n}", "feedback_id": f.id, "content": f.content, "score": round(score, 3),
                    "category": f.decision.category, "product_area": f.decision.product_area,
                    "severity": f.decision.severity, "churn_probability": f.decision.churn_probability})
    return out


def answer(session, question: str, k: int | None = None, category=None, area=None) -> dict:
    k = k or config.RAG_TOP_K
    ev = retrieve(session, question, k, category, area)
    if not ev:
        return {"answer": "No feedback matches this question yet.", "evidence": [], "grounding": None}

    ctx = "\n".join(f"[{e['tag']}] (category={e['category']}, area={e['product_area']}, "
                    f"severity={e['severity']}, churn={e['churn_probability']:.0%}) {e['content']}" for e in ev)
    cats = Counter(e["category"] for e in ev)
    fallback = (f"Retrieved {len(ev)} relevant reports (mostly {cats.most_common(1)[0][0]}). "
                f"Avg severity {np.mean([e['severity'] for e in ev]):.1f}/5. "
                f"Top match [F1]: \"{ev[0]['content'][:160]}\"")
    prompt = (
        "You are a product analyst answering a PM's question using ONLY the customer feedback below.\n"
        "Rules: cite the supporting feedback tag after every claim, like [F2] or [F1][F4]. "
        "If the feedback does not answer the question, say so. Do not invent numbers or users. "
        "Max 150 words.\n\n"
        f"Feedback:\n{ctx}\n\nQuestion: {question}"
    )
    text, model = llm_service.generate(prompt, fallback)

    cited = re.findall(r"\[(F\d+)\]", text)
    valid = {e["tag"] for e in ev}
    grounding = {
        "citations": len(cited),
        "valid_citations": sum(c in valid for c in cited),
        "invalid_citations": sorted({c for c in cited if c not in valid}),
        "uncited_answer": len(cited) == 0,
    }
    return {"answer": text, "model_used": model, "embedding_model": embedding_service.backend_name(),
            "evidence": ev, "grounding": grounding}
