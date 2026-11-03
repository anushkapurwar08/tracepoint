-- SignalOS schema for Supabase (Postgres + pgvector).
-- Run in Supabase SQL editor, then set DATABASE_URL in backend/.env to the
-- Supabase connection string (Settings → Database → Connection string → URI).
-- Locally the app uses SQLite with the same tables (created automatically).

create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists product_issues (
  id text primary key default gen_random_uuid()::text,
  title text,
  category text,
  product_area text,
  feedback_count int default 0,
  avg_severity double precision default 0,
  avg_churn double precision default 0,
  priority_score double precision default 0,
  priority text default 'P3',
  keywords jsonb,
  summary text,
  status text default 'open',
  last_seen timestamptz default now(),
  created_at timestamptz default now()
);

create table if not exists feedback (
  id text primary key default gen_random_uuid()::text,
  content text not null,
  source text default 'api',
  user_id text,
  issue_id text references product_issues(id),
  created_at timestamptz default now()
);

create table if not exists ai_decisions (
  id text primary key default gen_random_uuid()::text,
  feedback_id text not null references feedback(id) on delete cascade,
  category text,
  category_confidence double precision,
  category_probs jsonb,
  severity double precision,
  severity_confidence double precision,
  severity_probs jsonb,
  churn_probability double precision,
  product_area text,
  priority text,
  needs_review int default 0,
  model_used text,
  latency_ms double precision,
  created_at timestamptz default now()
);

-- The app writes vectors as JSON (portable across SQLite/Postgres).
-- For production-scale search, add a pgvector column and index:
create table if not exists feedback_embeddings (
  feedback_id text primary key references feedback(id) on delete cascade,
  vector jsonb,
  model text,
  embedding vector(384)   -- optional: populated by a trigger or backfill for pgvector search
);
create index if not exists feedback_embeddings_hnsw
  on feedback_embeddings using hnsw (embedding vector_cosine_ops);

create table if not exists engineering_tickets (
  id text primary key default gen_random_uuid()::text,
  issue_id text unique references product_issues(id),
  title text,
  description text,
  assigned_team text,
  priority text,
  ticket_status text default 'created',
  external_ref text,
  created_at timestamptz default now()
);

create table if not exists digests (
  id text primary key default gen_random_uuid()::text,
  content text,
  model_used text,
  created_at timestamptz default now()
);

create index if not exists idx_feedback_issue on feedback(issue_id);
create index if not exists idx_issues_status_score on product_issues(status, priority_score desc);
