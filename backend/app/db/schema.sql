
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS corpus_hackathons (
    id        TEXT PRIMARY KEY,           -- '<source>:<event id>', for example 'alphahack:1234'
    platform  TEXT NOT NULL,
    name      TEXT,
    start_date DATE,
    end_date DATE,
    winners_announced BOOLEAN NOT NULL DEFAULT TRUE,
    last_checked TIMESTAMPTZ,
    themes    TEXT[]
);

CREATE TABLE IF NOT EXISTS corpus_projects (
    id             BIGSERIAL PRIMARY KEY,
    platform       TEXT NOT NULL,
    hackathon_id   TEXT REFERENCES corpus_hackathons(id),
    submission_url TEXT NOT NULL UNIQUE,
    title          TEXT,
    tagline        TEXT,
    overview       TEXT,
    gap_highlights TEXT,
    tech_tags      TEXT[],
    tech_text      TEXT,
    prizes         JSONB,
    hackathon_date DATE,
    repo_url       TEXT,
    source         TEXT NOT NULL,
    dead_link_flag BOOLEAN NOT NULL DEFAULT FALSE,
    embedding_model TEXT NOT NULL DEFAULT 'BAAI/bge-small-en-v1.5',
    imported_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding      vector(384),
    search_vector  tsvector GENERATED ALWAYS AS (
        to_tsvector('english',
            coalesce(title, '') || ' ' || coalesce(tagline, '') || ' ' ||
            coalesce(overview, '') || ' ' || coalesce(gap_highlights, '') || ' ' ||
            coalesce(tech_text, ''))
    ) STORED
);

CREATE INDEX IF NOT EXISTS corpus_projects_fts ON corpus_projects USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS corpus_projects_date ON corpus_projects (hackathon_date);
CREATE TABLE IF NOT EXISTS repo_cache (
 repo_url TEXT PRIMARY KEY, trimmed_readme TEXT NOT NULL DEFAULT '', folder_tree JSONB NOT NULL DEFAULT '[]', tree_complete BOOLEAN NOT NULL DEFAULT FALSE,
 readme_hash TEXT, readme_present BOOLEAN NOT NULL DEFAULT FALSE, explainer_json JSONB, license_spdx TEXT, is_copyleft BOOLEAN NOT NULL DEFAULT FALSE,
 fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(), expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + interval '7 days'
);
CREATE INDEX IF NOT EXISTS repo_cache_expires ON repo_cache(expires_at);
ALTER TABLE repo_cache ADD COLUMN IF NOT EXISTS tree_complete BOOLEAN NOT NULL DEFAULT FALSE;
CREATE TABLE IF NOT EXISTS user_accounts (
 id UUID PRIMARY KEY, auth_provider TEXT NOT NULL, auth_subject_id TEXT NOT NULL UNIQUE,
 daily_profiles_used INTEGER NOT NULL DEFAULT 0, daily_analyses_used INTEGER NOT NULL DEFAULT 0,
 daily_explains_used INTEGER NOT NULL DEFAULT 0,
 quota_reset_date DATE NOT NULL DEFAULT CURRENT_DATE, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE user_accounts ADD COLUMN IF NOT EXISTS daily_profiles_used INTEGER NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS guest_usage (
 guest_key TEXT PRIMARY KEY,
 daily_profiles_used INTEGER NOT NULL DEFAULT 0,
 daily_analyses_used INTEGER NOT NULL DEFAULT 0,
 daily_explains_used INTEGER NOT NULL DEFAULT 0,
 quota_reset_date DATE NOT NULL DEFAULT CURRENT_DATE
);
CREATE TABLE IF NOT EXISTS analysis_overviews (
 id UUID PRIMARY KEY,
 user_id UUID NOT NULL REFERENCES user_accounts(id) ON DELETE CASCADE,
 idea TEXT NOT NULL,
 mode TEXT NOT NULL,
 verdict_direction TEXT NOT NULL,
 crowdedness_summary TEXT NOT NULL,
 feature_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
 top_projects JSONB NOT NULL DEFAULT '[]'::jsonb,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS analysis_overviews_user_recent ON analysis_overviews(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS ingestion_runs (
 id BIGSERIAL PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ,
 trigger_type TEXT NOT NULL, source_name TEXT, added_count INTEGER NOT NULL DEFAULT 0,
 pruned_count INTEGER NOT NULL DEFAULT 0, db_size_mb NUMERIC(6,2), status TEXT NOT NULL
);
