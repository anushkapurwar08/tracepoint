# SignalOS — AI product intelligence engine

Turns raw customer feedback (app reviews, support tickets, surveys) into **prioritised product issues and engineering tickets**, lets a PM **ask questions of the feedback** with answers grounded in real user quotes, and runs an **Investigator agent** that finds the root cause of a critical issue.

**Design principle:** a *workflow* for routine, high-volume decisions (predictable, cheap, auditable), and an *agent* only for open-ended investigation, where the next step depends on what was just found.

> 50 complaints about failed payments should become **one P0 ticket for Payments Engineering**, not 50 tickets or a vague summary.

## The problem

A product team receiving thousands of feedback entries a month can't read them all, spot recurring problems, judge severity and decide what to fix first. Generic "summarise my reviews" tools produce prose. PMs need **decisions**: what is broken, how badly, for how many people, and who should fix it.

## How it works

```
feedback (API / CSV / n8n webhook)
   │
   ├─► Embed (all-MiniLM-L6-v2, local) ──────────────► vector index ◄── /ask (RAG)
   │                                                        │
   ├─► Jev typed decisions                                  │
   │     choice: category (bug/ux/feature/performance/other)│
   │     choice: product area (payments/auth/…)             │
   │     score : severity 1–5                               │
   │     noul  : P(churn)                                   │
   │                                                        │
   ├─► Group into issue: same category+area AND nearest ────┘
   │   member by meaning (cosine ≥ threshold), else new issue
   │
   ├─► Priority = transparent weighted formula (not an LLM's opinion)
   │     0.4·severity + 0.3·log(volume) + 0.2·churn + 0.1·recency  → P0–P3
   │
   ├─► Gemini writes issue title, summary, suggested action
   │
   └─► P0/P1 issue → one engineering ticket (idempotent) → n8n → Slack/Jira
```

### Which part of the system makes which decision

| Decision | Made by | Why |
|---|---|---|
| Category, product area, severity, churn risk | **Jev** (typed decision model) | Fixed answer space; returns calibrated probabilities; cheap per entry |
| Which reports are the same issue | **Embeddings** + deterministic rules | Matching by meaning; the category/area partition stops unrelated merges |
| What to fix first | **Weighted formula** (code) | Must be explainable and tunable; every score shows its breakdown |
| Issue titles, digests, answers to PM questions | **Gemini** | Needs language generation |
| When to trust the AI | **Code** | Low-confidence decisions are flagged `needs_review` for a human |

Jev answers each question independently, so dependencies between decisions (priority depends on severity *and* churn) live in our backend, not in the model.

### Ask your feedback (RAG)

`POST /ask {"question": "Why are users churning in payments?"}`

1. Embed the question and retrieve the top-k most similar feedback entries (optionally filtered by category or area).
2. Gemini answers **only from those entries**, citing each claim like `[F2]`.
3. **Citations are verified**: the response reports how many cited tags match retrieved feedback and flags uncited answers. Grounding is measured, not assumed.

## Investigator agent

`POST /issues/{id}/investigate`, or the **Investigate** button on any issue.

Triage is a fixed pipeline on purpose. Investigating a P0 is different: the right next step depends on the last finding. For example, a spike after an update leads to checking versions, which shows mostly Android 14, which leads to searching Android 14 payment reports. So that step is an agent: Gemini with function calling picks its own tools in a loop.

| Tool (all read-only) | What it gives the agent |
|---|---|
| `get_issue_details` | Stats, summary, member reports with IDs |
| `extract_mentions` | Counts of OS versions, devices, app updates, payment methods, error text |
| `feedback_timeline` | Reports by day and channel (spike detection) |
| `search_feedback` | Semantic search across *all* feedback, to test a hypothesis |
| `find_related_issues` | Issues close in meaning across areas (e.g. one bad release breaking several areas) |
| `verify_hypothesis` | **Jev** `noul`: P(the cited feedback supports this hypothesis) |
| `submit_report` | Root cause, confidence, affected segment, evidence IDs, actions, open questions |

**Guardrails**
- **Generator/verifier split:** Gemini proposes, Jev checks. A report is only accepted after verification passes (support ≥ 0.6). A weak or vague hypothesis is sent back to be narrowed.
- **No fabricated citations:** every cited feedback ID is checked against the database. Unknown IDs are rejected and counted.
- **Read-only, bounded:** the agent can recommend a ticket but never create one; it has a hard cap of 8 tool calls and can only investigate the issue it was given.
- **Auditable:** every step (the agent's stated reason, tool, arguments, result) is stored in `agent_runs` and shown in the dashboard.
- **Honest fallback:** without a Gemini key, a fixed scripted investigation runs and is labelled *"scripted (not agentic)"*.

### Hidden-cause evaluation

`eval/agent_eval.py` plants a known root cause among same-area noise in three scenarios: UPI failing on Android 14; OTP SMS not reaching Jio numbers; video crashes on Samsung devices. It then measures whether the agent names the cause, against a scripted baseline using the same tools in a fixed order.

| | Root causes found | Avg tool calls | Fabricated IDs reaching the report |
|---|---|---|---|
| Agent | _run with keys_ | | 0 (enforced) |
| Scripted baseline | _run with keys_ | | 0 |

The Jio case is the real test: no keyword rule finds it, so the agent has to read the reports. Full traces and briefs: [`eval/agent_results.md`](eval/agent_results.md).

## Evaluation

`eval/benchmark.py` runs the same 50 human-labelled entries through a **keyword baseline**, **Jev** and **Gemini**. It reports category accuracy, severity MAE, churn precision/recall, latency and consistency across repeats.

| Model | Category acc. | Severity MAE | Churn precision | Churn recall |
|---|---|---|---|---|
| Keyword baseline | 0.82 | 0.68 | 1.00 | 0.19 |
| Jev | _run with key_ | | | |
| Gemini | _run with key_ | | | |

The keyword baseline's 19% churn recall is the gap a model has to close to justify its cost. Full results and misclassified examples: [`eval/results.md`](eval/results.md).

## Failure handling

- **No API keys / provider down:** each component falls back to a labelled alternative (keyword heuristic, templated text, hashed vectors). Every decision stores `model_used`, so mock results are never mistaken for real ones.
- **Malformed CSV rows:** skipped and reported with row numbers; the batch continues.
- **Bounded LLM spend:** Gemini regenerates an issue's title only at milestones (1, 3, 10, 25, 50 reports), not on every new report.
- **Duplicate tickets:** at most one ticket per issue; it updates as reports accumulate.

## Run it

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add TYPESAFE_API_KEY / GEMINI_API_KEY (optional)
uvicorn app.main:app --reload
```

Open http://localhost:8000, click **Upload CSV**, and choose `data/sample_feedback.csv`. API docs: http://localhost:8000/docs

- **Benchmark:** `cd backend && python ../eval/benchmark.py --repeats 3`
- **Agent eval:** `cd backend && python ../eval/agent_eval.py --runs 3` (this wipes the local DB, so re-upload the CSV afterwards)
- **Supabase:** run `sql/schema.sql`, then set `DATABASE_URL` in `.env`.
- **n8n:** import the JSON files in `n8n/`, then set the backend URL and Slack webhook in the HTTP nodes.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/feedback` | Analyse one entry → decision + issue + priority explanation |
| POST | `/feedback/bulk` | CSV upload (`content` column) |
| POST | `/ask` | Grounded Q&A over feedback, with verified citations |
| POST | `/issues/{id}/investigate` | Run the Investigator agent on an issue |
| GET | `/investigations` | Past agent runs with full traces |
| GET | `/issues` | Issues ranked by priority score (filter `?priority=P0`) |
| GET | `/feedback` | Entries with AI decisions (filter `?issue_id=`) |
| GET | `/tickets` | Engineering tickets |
| GET | `/stats` | Totals, categories, review queue, latency |
| POST | `/digest` | Daily digest (called by n8n) |

## Stack

FastAPI · Gemini function calling (agent) · SQLAlchemy (SQLite locally, Supabase Postgres + pgvector in production) · Jev (TypeSafe AI) · Gemini · sentence-transformers · n8n · vanilla JS dashboard

## Known limitations

- The labelled set is small (50 entries) and labelled by one person, so accuracy numbers have wide error bars.
- Vector search is a numpy scan over all rows: fine for thousands of rows; past ~100k, switch to the pgvector index in the schema.
- Grouping thresholds were tuned on the same small set they're shown on; they need a held-out set.
- Issues are never auto-closed or re-split if a cluster drifts.
- Citation checking confirms a cited tag exists, not that the quote actually supports the claim. Checking that needs an entailment step (a good job for a Jev `noul` question).
- The agent's hidden-cause test has only 3 scenarios, written by the author, and scoring is keyword-based. It's a sanity check, not a benchmark.
- The agent investigates one issue per request, synchronously (10–40 s). Production would queue runs and stream the trace.
- The dashboard has no authentication.

## Next

Real public reviews for a live app · held-out evaluation set · Jev `noul` faithfulness check on RAG answers · Jira integration · issue resolution tracking (did complaints drop after the fix shipped?)
