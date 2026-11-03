# SignalOS — handoff brief (paste this into a new chat instead of the long history)

**Goal:** a portfolio project for a Google APM application. One deep project; SkillRoute was dropped on purpose.

**What it is:** feedback → Jev typed decisions → semantic grouping into issues → priority formula → tickets, plus `/ask` RAG over feedback with verified citations. Dashboard at `/`.

**Done (runs end to end with no keys, in labelled fallback mode):**
- Investigator agent (`app/services/agent_service.py`): Gemini function calling over read-only tools, Jev-verified hypotheses, fabricated-ID rejection, 8-step cap, trace stored in `agent_runs`, Investigate button in the dashboard. Hidden-cause eval: `eval/agent_eval.py` + `eval/agent_cases.json`.
- Jev verified live: real response format confirmed; score levels are 0-indexed (fixed in parser). Gemini key works after project switch.
- FastAPI backend in `backend/app/`. Services: jev, llm (Gemini), embedding, aggregation, priority, ticket, rag, pipeline.
- Jev client: `POST https://api.typesafe.ai/v1/systemone`, Bearer `TYPESAFE_API_KEY`, model `jev-latest`. Real response shape: `answers.<key>` → choice `{choice, confidence, probabilities}`, score `{score (0-indexed), legend, probabilities}`, noul `{noul}`.
- Gemini via REST `generateContent`, model alias `gemini-flash-latest`.
- Embeddings: local `all-MiniLM-L6-v2`, stored as JSON, numpy cosine search (pgvector upgrade path in `sql/schema.sql`).
- 50 labelled synthetic entries in `data/sample_feedback.csv`. `eval/benchmark.py` compares keyword baseline vs Jev vs Gemini.
- n8n workflow exports in `n8n/` (intake + P0/P1 alert; daily digest). README written.

**Key design decisions (be ready to defend in interviews):**
- Jev decides; code prioritises (weighted formula, explainable); Gemini only writes.
- Grouping is partitioned by category + area *before* semantic matching, to stop unrelated merges.
- At most one ticket per issue. Gemini is called only at milestones, to bound cost.
- Mock and fallback results are always labelled via `model_used`.

**To do next (in order):**
1. Run `python ../eval/benchmark.py --repeats 2` and `python ../eval/agent_eval.py --runs 3`; put the real numbers in both README tables.
2. Replace the synthetic data with ~100–200 real public Play Store reviews of one consumer app; hand-label a held-out set of 50.
3. Tune `SEMANTIC_THRESHOLD` on the real data.
4. Redesign the UI (agreed direction: "signal field" of embedding dots + a command bar for /ask and the agent).
5. Push to GitHub, screenshot the n8n workflows, record a 2-minute demo.
6. Write a 1–2 page product doc and the resume bullet with real numbers.

**Token tips:** run things locally and paste only errors; batch requests.
