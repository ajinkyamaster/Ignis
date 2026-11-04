#!/usr/bin/env python3
"""
Ignis corpus builder
====================

Builds the winners corpus for Ignis from two public Devpost datasets and loads it
into Postgres (with pgvector).

Sources
  - xenosaac/alphahack-devpost            (Hugging Face, CC BY 4.0, crawled March 2026)  [main]
  - alvanlii/devpost-hackathon-projects   (Hugging Face, snapshot Jan 2025)              [gap-filler]

What counts as "recognized" (kept in the corpus)
  Any project with at least one prize or award, including honorable mentions and
  sponsor prizes. The full prize list is stored so you can filter later.

Privacy
  Team member names and profile links are never read or stored. E-mail addresses
  found in text are removed.

Install
  pip install huggingface_hub pyarrow "psycopg[binary]"
  pip install sentence-transformers          # only for the optional `embed` step

Typical run
  python build_corpus.py download
  python build_corpus.py build                 # writes corpus.jsonl
  python build_corpus.py build --limit 1000    # small sample for the storage benchmark
  export DATABASE_URL=postgresql://user:pass@host:5432/dbname
  python build_corpus.py load
  python build_corpus.py embed                 # optional, needs sentence-transformers
  python build_corpus.py stats --budget-mb 350
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

DATA_DIR = Path("data")
OUT_FILE = Path("corpus.jsonl")

ALPHAHACK_REPO = "xenosaac/alphahack-devpost"
ALVANLII_REPO = "alvanlii/devpost-hackathon-projects"
ALPHAHACK_SOURCE = "alphahack-devpost (CC BY 4.0), crawled Mar 2026"
ALVANLII_SOURCE = "alvanlii/devpost-hackathon-projects, snapshot Jan 2025"

OVERVIEW_MAX = 500   # characters kept for the short overview
GAP_PART_MAX = 300   # characters kept for each gap section (challenges, what's next)

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
JUNK_MARKERS = ("window.fbAsyncInit", "FB.init(", "function(", "<script")
GITHUB_RE = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def clean(text: Any) -> str:
    """Strip HTML tags and e-mail addresses, collapse whitespace."""
    if text is None:
        return ""
    s = str(text)
    s = TAG_RE.sub(" ", s)
    s = EMAIL_RE.sub("", s)
    return WS_RE.sub(" ", s).strip()


def looks_like_junk(s: str) -> bool:
    return any(m in s for m in JUNK_MARKERS)


def trunc(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    cut = s[:n - 1].rsplit(" ", 1)[0]
    return cut.rstrip(".,;:- ") + "…"


def as_list(v: Any) -> list:
    """Turn lists, numpy arrays, JSON strings or Python-repr strings into a list."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if hasattr(v, "tolist"):
        out = v.tolist()
        return out if isinstance(out, list) else [out]
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(v)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            return [parsed]
        return [v]
    return [v]


def parse_date(v: Any) -> date | None:
    """Best-effort date. Falls back to Jan 1 of a year found in the text."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if hasattr(v, "to_pydatetime"):
        return v.to_pydatetime().date()
    if isinstance(v, (int, float)) and 1990 <= int(v) <= 2100:
        return date(int(v), 1, 1)
    s = str(v)
    m = ISO_RE.search(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = YEAR_RE.search(s)
    if m:
        return date(int(m.group(1)), 1, 1)
    return None


def norm_url(u: Any) -> str | None:
    if not u:
        return None
    p = urlparse(str(u).strip())
    host = p.netloc.lower()
    if not host:
        return None
    return f"https://{host}{p.path.rstrip('/')}"


def norm_repo(u: Any) -> str | None:
    n = norm_url(u)
    return n.lower() if n and n.startswith("https://github.com/") else None


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------- #
# Section parsing (Devpost template headings)
# --------------------------------------------------------------------------- #

# In the order they appear on a Devpost project page. Some datasets glue the heading
# straight to the following text ("What it doesThe website..."), so we match the
# heading words anywhere and keep only matches that respect this order.
HEADINGS = [
    ("problem", r"The problem it solves"),
    ("inspiration", r"Inspiration"),
    ("what_it_does", r"What it does"),
    ("how_built", r"How (?:we|I) built it"),
    ("challenges", r"Challenges (?:we|I) ran into"),
    ("accomplishments", r"Accomplishments that (?:we|I)(?:[’']re|[’']m|\s+are|\s+am)?\s+proud of"),
    ("learned", r"What (?:we|I) learned"),
    ("whats_next", r"What[’']s next for"),
    ("built_with", r"Built With"),
    ("try_it", r"Try it out"),
]
HEADING_RES = [(key, re.compile(pat)) for key, pat in HEADINGS]


def split_sections(text: str) -> tuple[str, dict[str, str]]:
    found = []
    for rank, (key, rx) in enumerate(HEADING_RES):
        for m in rx.finditer(text):
            found.append((m.start(), m.end(), rank, key))
    found.sort()
    accepted = []
    last_rank = -1
    for start, end, rank, key in found:
        if rank > last_rank and (not accepted or start >= accepted[-1][1]):
            accepted.append((start, end, rank, key))
            last_rank = rank
    if not accepted:
        return text.strip(), {}
    preamble = text[: accepted[0][0]].strip()
    sections: dict[str, str] = {}
    for i, (_, end, _, key) in enumerate(accepted):
        nxt = accepted[i + 1][0] if i + 1 < len(accepted) else len(text)
        sections[key] = text[end:nxt].strip()
    return preamble, sections


def strip_title_prefix(s: str, title: str) -> str:
    """'What's next for <Title>' is followed by the title; remove it if present."""
    words = title.split()
    if not words:
        return s
    rx = re.compile(r"^\s*" + r"\s*".join(re.escape(w) for w in words) + r"\s*[:?!.-]*\s*", re.I)
    return rx.sub("", s, count=1)


def make_fields(title: str, tagline: str, description: Any) -> tuple[str, str]:
    """Return (overview, gap_highlights) built from a raw description."""
    text = clean(description)
    preamble, sections = split_sections(text) if text else ("", {})

    overview = ""
    for key in ("what_it_does", "problem", "inspiration"):
        s = sections.get(key, "")
        if len(s) >= 40 and not looks_like_junk(s):
            overview = trunc(s, OVERVIEW_MAX)
            break
    if not overview and len(preamble) >= 80 and not looks_like_junk(preamble):
        overview = trunc(preamble, OVERVIEW_MAX)
    if not overview and tagline:
        overview = trunc(tagline, OVERVIEW_MAX)

    parts = []
    ch = sections.get("challenges", "")
    if len(ch) >= 20 and not looks_like_junk(ch):
        parts.append("Challenges: " + trunc(ch, GAP_PART_MAX))
    nx = strip_title_prefix(sections.get("whats_next", ""), title)
    if len(nx) >= 20 and not looks_like_junk(nx):
        parts.append("What's next: " + trunc(nx, GAP_PART_MAX))
    return overview, trunc("\n".join(parts), 600)


# --------------------------------------------------------------------------- #
# Record store (deduplicated by project URL)
# --------------------------------------------------------------------------- #

def norm_prizes(raw: Any) -> list[tuple[str, Any]]:
    out = []
    for p in as_list(raw):
        if isinstance(p, dict):
            name = p.get("prize_name") or p.get("name") or p.get("prize")
            amount = p.get("prize_amount") or p.get("amount")
        else:
            name, amount = p, None
        name = clean(name)
        if name:
            out.append((name, amount))
    return out


def add_record(store: dict, *, url: str, source: str, title: str, tagline: str,
               description: Any, tech: list, repo: str | None, event_key: str,
               event_name: str | None, event_date: date | None, themes: list,
               prizes: list[tuple[str, Any]], platform: str = "devpost",
               project_level_awards: bool = False) -> None:
    rec = store.get(url)
    if rec is None:
        overview, gap = make_fields(title, tagline, description)
        rec = store[url] = {
            "submission_url": url, "platform": platform, "title": title, "tagline": tagline,
            "overview": overview, "gap_highlights": gap,
            "tech_tags": [clean(t) for t in tech if clean(t)][:30],
            "repo_url": repo, "recognitions": [], "project_awards": [], "source": source,
        }
    else:
        if not rec["overview"] or not rec["gap_highlights"]:
            overview, gap = make_fields(rec["title"] or title, rec["tagline"] or tagline, description)
            rec["overview"] = rec["overview"] or overview
            rec["gap_highlights"] = rec["gap_highlights"] or gap
        if not rec["repo_url"] and repo:
            rec["repo_url"] = repo
        if not rec["tech_tags"] and tech:
            rec["tech_tags"] = [clean(t) for t in tech if clean(t)][:30]
    if project_level_awards:
        # AlphaHack repeats a project's global award list on every submitted-to
        # event row. The list cannot establish which event granted each prize.
        existing = {item["prize"] for item in rec["project_awards"]}
        for name, amount in prizes or [("Award reported by source (event unverified)", None)]:
            if name not in existing:
                rec["project_awards"].append({"prize": name, "amount": amount,
                                               "event": None, "event_date": None})
                existing.add(name)
        prizes = [("Award reported by source (event unverified)", None)]
    seen = {(r["event_key"], r["prize"]) for r in rec["recognitions"]}
    for name, amount in (prizes or [("Recognized (prize not listed)", None)]):
        if (event_key, name) not in seen:
            rec["recognitions"].append({
                "event_key": event_key, "event_name": event_name,
                "event_date": event_date.isoformat() if event_date else None,
                "themes": themes, "prize": name, "amount": amount,
            })
            seen.add((event_key, name))


def finalize(store: dict, max_age_years: int, stats: dict) -> list[dict]:
    cutoff = None
    if max_age_years:
        today = date.today()
        cutoff = date(today.year - max_age_years, today.month, min(today.day, 28))
    out = []
    for rec in store.values():
        recs = rec.pop("recognitions")
        project_awards = rec.pop("project_awards", [])
        dated = [r for r in recs if r["event_date"]]
        primary = max(dated, key=lambda r: r["event_date"]) if dated else recs[0]
        h_date = parse_date(primary["event_date"])
        if cutoff and (not h_date or h_date < cutoff):
            stats["dropped_too_old"] += 1
            continue
        rec["hackathon_id"] = primary["event_key"]
        rec["hackathon_name"] = primary["event_name"]
        rec["hackathon_date"] = primary["event_date"]
        rec["themes"] = primary["themes"]
        rec["prizes"] = project_awards or [
            {"prize": r["prize"], "amount": r["amount"], "event": r["event_name"],
             "event_date": r["event_date"]} for r in recs]
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Source readers
# --------------------------------------------------------------------------- #

def load_alphahack_events(root: Path) -> dict[str, dict]:
    path = root / "events" / "events.jsonl"
    events: dict[str, dict] = {}
    if not path.exists():
        print(f"  ! {path} not found; event names and dates will be missing")
        return events
    keys = None
    for ev in read_jsonl(path):
        keys = keys or sorted(ev.keys())
        eid = ev.get("event_id", ev.get("id"))
        if eid is None:
            continue
        events[str(eid)] = {
            "name": ev.get("event_name") or ev.get("name") or ev.get("title"),
            "date": parse_date(ev.get("start_date") or ev.get("start") or ev.get("date")
                               or ev.get("submission_period_dates") or ev.get("_year") or ev.get("year")),
            "themes": [clean(t) for t in as_list(ev.get("themes") or ev.get("tags")) if clean(t)],
        }
    print(f"  events: {len(events)}  (fields seen: {keys})")
    return events


def read_alphahack(root: Path, store: dict, stats: dict, limit: int | None) -> None:
    shards = sorted((root / "projects").glob("projects_*.jsonl"))
    if not shards:
        print(f"  ! no projects_*.jsonl under {root}/projects; run `download` first")
        return
    events = load_alphahack_events(root)
    for shard in shards:
        print(f"  reading {shard.name}")
        for row in read_jsonl(shard):
            stats["alphahack_rows"] += 1
            prizes = norm_prizes(row.get("prizes_won"))
            try:
                n_prizes = int(row.get("num_prizes_won") or 0)
            except (TypeError, ValueError):
                n_prizes = 0
            if not (row.get("is_winner") or prizes or n_prizes > 0):
                continue
            url = norm_url(row.get("source_url"))
            if not url:
                continue
            eid = str(row.get("event_id"))
            ev = events.get(eid, {})
            add_record(
                store, url=url, source=ALPHAHACK_SOURCE,
                title=clean(row.get("title")), tagline=clean(row.get("tagline")),
                description=row.get("full_description"),
                tech=as_list(row.get("tech_tags")), repo=norm_repo(row.get("github_url")),
                event_key=f"alphahack:{eid}",
                event_name=clean(row.get("event_name") or ev.get("name")) or None,
                event_date=parse_date(row.get("start_date")) or ev.get("date") or parse_date(row.get("_year")),
                themes=ev.get("themes", []), prizes=prizes,
                project_level_awards=True,
            )
            stats["alphahack_kept_rows"] += 1
            if limit and len(store) >= limit:
                return


def load_alvanlii_hackathons(root: Path) -> dict[str, dict]:
    path = root / "hackathons.json"
    meta: dict[str, dict] = {}
    if not path.exists():
        return meta
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not read {path.name}: {e}")
        return meta
    items = data.items() if isinstance(data, dict) else enumerate(data)
    sample = None
    for k, v in items:
        if not isinstance(v, dict):
            continue
        sample = sample or sorted(v.keys())
        hid = v.get("id", v.get("hackathon_id", k if isinstance(data, dict) else None))
        if hid is None:
            continue
        meta[str(hid)] = {
            "name": v.get("title") or v.get("name"),
            "date": parse_date(v.get("start_date") or v.get("submission_period_dates")
                               or v.get("date") or v.get("end_date")),
            "themes": [clean(t) for t in as_list(v.get("themes")) if clean(t)],
        }
    print(f"  hackathons: {len(meta)}  (fields seen: {sample})")
    return meta


def read_alvanlii(root: Path, store: dict, stats: dict, limit: int | None) -> None:
    path = root / "combined_hackathons.parquet"
    if not path.exists():
        print(f"  ! {path} not found; skipping alvanlii")
        return
    import pyarrow.parquet as pq

    meta = load_alvanlii_hackathons(root)
    pf = pq.ParquetFile(path)
    wanted = ("hackathon_id", "project_link", "full_desc", "title", "brief_desc", "prize", "tags")
    cols = [c for c in wanted if c in pf.schema_arrow.names]  # team_members is never read
    for batch in pf.iter_batches(batch_size=2000, columns=cols):
        for row in batch.to_pylist():
            stats["alvanlii_rows"] += 1
            prizes = norm_prizes(row.get("prize"))
            if not prizes:
                continue
            url = norm_url(row.get("project_link"))
            if not url:
                continue
            hid = str(row.get("hackathon_id"))
            hm = meta.get(hid, {})
            repo_match = GITHUB_RE.search(str(row.get("full_desc") or ""))
            add_record(
                store, url=url, source=ALVANLII_SOURCE,
                title=clean(row.get("title")), tagline=clean(row.get("brief_desc")),
                description=row.get("full_desc"), tech=as_list(row.get("tags")),
                repo=norm_repo(repo_match.group(0) if repo_match else None),
                event_key=f"alvanlii:{hid}", event_name=clean(hm.get("name")) or None,
                event_date=hm.get("date"), themes=hm.get("themes", []), prizes=prizes,
            )
            stats["alvanlii_kept_rows"] += 1
            if limit and len(store) >= limit:
                return


# --------------------------------------------------------------------------- #
# Commands: download / build
# --------------------------------------------------------------------------- #

def cmd_download(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    files = [(ALPHAHACK_REPO, f"projects/projects_{i:02d}.jsonl") for i in range(10)]
    files.append((ALPHAHACK_REPO, "events/events.jsonl"))
    if not args.skip_alvanlii:
        files += [(ALVANLII_REPO, "combined_hackathons.parquet"), (ALVANLII_REPO, "hackathons.json")]
    for repo, name in files:
        local = Path(args.data_dir) / repo.replace("/", "__")
        path = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", local_dir=local)
        print(f"downloaded {repo}:{name} -> {path}")
    print("\nNext: python build_corpus.py build")


def cmd_build(args: argparse.Namespace) -> None:
    data = Path(args.data_dir)
    stats = {"alphahack_rows": 0, "alphahack_kept_rows": 0, "alvanlii_rows": 0,
             "alvanlii_kept_rows": 0, "dropped_too_old": 0}
    store: dict[str, dict] = {}
    print("AlphaHack")
    read_alphahack(data / ALPHAHACK_REPO.replace("/", "__"), store, stats, args.limit)
    if not args.skip_alvanlii and not (args.limit and len(store) >= args.limit):
        print("alvanlii")
        read_alvanlii(data / ALVANLII_REPO.replace("/", "__"), store, stats, args.limit)
    records = finalize(store, args.max_age_years, stats)
    if args.limit:
        records = records[: args.limit]

    with Path(args.out).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records) or 1
    print("\n--- build summary ---")
    print(f"rows read:   alphahack {stats['alphahack_rows']:,}   alvanlii {stats['alvanlii_rows']:,}")
    print(f"rows kept:   alphahack {stats['alphahack_kept_rows']:,}   alvanlii {stats['alvanlii_kept_rows']:,}"
          "  (recognized = any prize, honorable mentions included)")
    print(f"unique projects after dedupe: {len(records):,}  (dropped as too old: {stats['dropped_too_old']:,})")
    print(f"with gap highlights: {sum(1 for r in records if r['gap_highlights']):,} "
          f"({100 * sum(1 for r in records if r['gap_highlights']) // n}%)")
    print(f"with repo link:      {sum(1 for r in records if r['repo_url']):,}")
    print(f"with hackathon date: {sum(1 for r in records if r['hackathon_date']):,}")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")
    print("\nNext: python build_corpus.py load")


def cmd_import_manual(args: argparse.Namespace) -> None:
    """Import curated recognized submissions, with no live platform scraping."""
    path = Path(args.input)
    rows = read_jsonl(path) if path.suffix == ".jsonl" else csv.DictReader(path.open(newline="", encoding="utf-8"))
    store = {}
    for row in rows:
        prizes = norm_prizes(row.get("prizes") or row.get("prize"))
        url = norm_url(row.get("submission_url") or row.get("project_link"))
        if not prizes or not url:
            continue
        description = row.get("description") or row.get("full_desc") or ""
        match = GITHUB_RE.search(str(description))
        add_record(store, url=url, source=clean(row.get("source") or f"manual:{path.name}"),
                   title=clean(row.get("title")), tagline=clean(row.get("tagline")),
                   description=description, tech=as_list(row.get("tech_tags")),
                   repo=norm_repo(row.get("repo_url") or (match.group(0) if match else None)),
                   event_key=clean(row.get("hackathon_id") or row.get("hackathon_name") or "manual"),
                   event_name=clean(row.get("hackathon_name")),
                   event_date=parse_date(row.get("hackathon_date")),
                   themes=as_list(row.get("themes")), prizes=prizes,
                   platform=clean(row.get("platform") or "devfolio"))
    records = finalize(store, args.max_age_years, {"dropped_too_old": 0})
    with Path(args.out).open("a", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"appended {len(records)} recognized projects to {args.out}; run load and embed")


def cmd_select_budget(args: argparse.Namespace) -> None:
    """Admission control before loading a corpus larger than a free-tier database."""
    records = list(read_jsonl(Path(args.input)))
    if not records:
        sys.exit(f"No records in {args.input}")
    max_rows = args.max_rows or int(args.budget_mb * 1024 / args.kb_per_row * args.safety)
    keep = set(sorted(range(len(records)), key=lambda i: (
        bool(records[i].get("repo_url")), records[i].get("hackathon_date") or ""
    ), reverse=True)[:max_rows])
    with Path(args.out).open("w", encoding="utf-8") as f:
        for i, row in enumerate(records):
            if i in keep:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"selected {len(keep):,}/{len(records):,} recognized projects into {args.out}; "
          f"retained {sum(bool(records[i].get('repo_url')) for i in keep):,} with repo links")


def cmd_prune(args: argparse.Namespace) -> None:
    with _connect(args.dsn) as conn:
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("INSERT INTO ingestion_runs(trigger_type, status) VALUES ('prune', 'running') RETURNING id")
        run_id = cur.fetchone()[0]
        cur.execute("DELETE FROM repo_cache WHERE expires_at < now()")
        cache_count = cur.rowcount
        cur.execute("DELETE FROM corpus_projects WHERE hackathon_date IS NULL OR hackathon_date < CURRENT_DATE - INTERVAL '10 years'")
        removed = cur.rowcount
        budget = args.budget_mb * 1024 * 1024
        cur.execute("SELECT pg_total_relation_size('corpus_projects')")
        initial_size = cur.fetchone()[0]
        if removed or initial_size > budget:
            needs_vacuum = bool(removed)
            for _ in range(20):
                # VACUUM FULL is necessary: DELETE alone does not lower relation size.
                cur.execute("VACUUM FULL corpus_projects")
                needs_vacuum = False
                cur.execute("SELECT pg_total_relation_size('corpus_projects')")
                size = cur.fetchone()[0]
                if size <= budget:
                    break
                cur.execute("SELECT count(*) FROM corpus_projects")
                total = cur.fetchone()[0]
                if total == 0:
                    break
                batch_size = max(1, min(total, int(total * (size - budget) / size * 1.2) + 1))
                cur.execute("DELETE FROM corpus_projects WHERE id IN (SELECT id FROM corpus_projects "
                            "WHERE dead_link_flag OR repo_url IS NULL ORDER BY hackathon_date NULLS FIRST LIMIT %s)", (batch_size,))
                count = cur.rowcount
                remaining = batch_size - count
                if remaining:
                    cur.execute("SELECT min(hackathon_date) FROM corpus_projects")
                    oldest = cur.fetchone()[0]
                    if oldest:
                        cur.execute("DELETE FROM corpus_projects WHERE hackathon_date < %s::date + INTERVAL '6 months'", (oldest,))
                    else:
                        cur.execute("DELETE FROM corpus_projects WHERE id IN (SELECT id FROM corpus_projects "
                                    "ORDER BY id LIMIT %s)", (remaining,))
                    count += cur.rowcount
                removed += count
                needs_vacuum = count > 0
                if count == 0:
                    break
            if needs_vacuum:
                cur.execute("VACUUM FULL corpus_projects")
        cur.execute('DELETE FROM corpus_hackathons h WHERE NOT EXISTS '
                    '(SELECT 1 FROM corpus_projects p WHERE p.hackathon_id=h.id)')
        cur.execute("SELECT pg_total_relation_size('corpus_projects') / 1048576.0")
        size_mb = cur.fetchone()[0]
        cur.execute("UPDATE ingestion_runs SET finished_at=now(), status=%s, pruned_count=%s, db_size_mb=%s WHERE id=%s",
                    ("complete" if size_mb <= args.budget_mb else "budget_not_reclaimed", removed, size_mb, run_id))
    print(f"expired cache: {cache_count}; projects pruned: {removed}; table+indexes: {size_mb:.2f} MB")


def cmd_takedown(args: argparse.Namespace) -> None:
    """Remove a requested submission and record the admin action."""
    url = norm_url(args.url)
    if not url:
        sys.exit('Pass a valid submission URL')
    with _connect(args.dsn) as conn, conn.cursor() as cur:
        cur.execute('DELETE FROM corpus_projects WHERE submission_url=%s RETURNING repo_url', (url,))
        removed = cur.fetchone()
        if removed and removed[0]:
            cur.execute('DELETE FROM repo_cache WHERE repo_url=%s', (norm_repo(removed[0]),))
        cur.execute('DELETE FROM corpus_hackathons h WHERE NOT EXISTS '
                    '(SELECT 1 FROM corpus_projects p WHERE p.hackathon_id=h.id)')
        cur.execute("INSERT INTO ingestion_runs(trigger_type, source_name, finished_at, pruned_count, status) "
                    "VALUES ('takedown', 'admin_cli', now(), %s, 'complete')", (1 if removed else 0,))
    print('removed' if removed else 'not found', url)


# --------------------------------------------------------------------------- #
# Commands: load / embed / stats  (Postgres + pgvector)
# --------------------------------------------------------------------------- #

SCHEMA_SQL = """
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
    embedding      vector({dim}),
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
CREATE TABLE IF NOT EXISTS ingestion_runs (
 id BIGSERIAL PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ,
 trigger_type TEXT NOT NULL, source_name TEXT, added_count INTEGER NOT NULL DEFAULT 0,
 pruned_count INTEGER NOT NULL DEFAULT 0, db_size_mb NUMERIC(6,2), status TEXT NOT NULL
);
"""


def _connect(dsn: str | None):
    import psycopg
    from dotenv import dotenv_values

    dsn = dsn or os.environ.get("DATABASE_URL") or dotenv_values(Path(__file__).resolve().parents[2] / '.env').get('DATABASE_URL')
    if not dsn:
        sys.exit("Set DATABASE_URL or pass --dsn (postgresql://user:pass@host:5432/dbname)")
    return psycopg.connect(dsn)


def cmd_load(args: argparse.Namespace) -> None:
    from psycopg.types.json import Jsonb

    records = list(read_jsonl(Path(args.corpus)))
    if not records:
        sys.exit(f"{args.corpus} is empty or missing; run `build` first")

    hackathons = {}
    for r in records:
        hackathons[r["hackathon_id"]] = (
            r["hackathon_id"], r["platform"], r.get("hackathon_name"),
            parse_date(r.get("hackathon_date")), r.get("themes") or [],
        )

    with _connect(args.dsn) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.format(dim=args.dim))
        cur.execute('SELECT count(*) FROM corpus_projects')
        before_count = cur.fetchone()[0]
        cur.executemany(
            """INSERT INTO corpus_hackathons (id, platform, name, start_date, themes)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name,
                   start_date = EXCLUDED.start_date, themes = EXCLUDED.themes
               WHERE (corpus_hackathons.name, corpus_hackathons.start_date, corpus_hackathons.themes)
                   IS DISTINCT FROM (EXCLUDED.name, EXCLUDED.start_date, EXCLUDED.themes)""",
            list(hackathons.values()),
        )
        rows = [
            (r["platform"], r["hackathon_id"], r["submission_url"], r["title"], r["tagline"],
             trunc(r["overview"], OVERVIEW_MAX), trunc(r["gap_highlights"], 600),
             r["tech_tags"], " ".join(r["tech_tags"]),
             Jsonb(r["prizes"]), parse_date(r.get("hackathon_date")), r.get("repo_url"), r["source"])
            for r in records
        ]
        cur.executemany(
            """INSERT INTO corpus_projects
                   (platform, hackathon_id, submission_url, title, tagline, overview,
                    gap_highlights, tech_tags, tech_text, prizes, hackathon_date, repo_url, source)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (submission_url) DO UPDATE SET
                   hackathon_id = EXCLUDED.hackathon_id, title = EXCLUDED.title,
                   tagline = EXCLUDED.tagline, overview = EXCLUDED.overview,
                   gap_highlights = EXCLUDED.gap_highlights, tech_tags = EXCLUDED.tech_tags,
                   tech_text = EXCLUDED.tech_text, prizes = EXCLUDED.prizes,
                   hackathon_date = EXCLUDED.hackathon_date, repo_url = EXCLUDED.repo_url,
                   source = EXCLUDED.source,
                   embedding = CASE WHEN (corpus_projects.title, corpus_projects.tagline,
                       corpus_projects.overview, corpus_projects.gap_highlights) IS DISTINCT FROM
                       (EXCLUDED.title, EXCLUDED.tagline, EXCLUDED.overview, EXCLUDED.gap_highlights)
                       THEN NULL ELSE corpus_projects.embedding END
               WHERE (corpus_projects.hackathon_id, corpus_projects.title, corpus_projects.tagline,
                      corpus_projects.overview, corpus_projects.gap_highlights, corpus_projects.tech_tags,
                      corpus_projects.tech_text, corpus_projects.prizes, corpus_projects.hackathon_date,
                      corpus_projects.repo_url, corpus_projects.source)
                   IS DISTINCT FROM
                     (EXCLUDED.hackathon_id, EXCLUDED.title, EXCLUDED.tagline,
                      EXCLUDED.overview, EXCLUDED.gap_highlights, EXCLUDED.tech_tags,
                      EXCLUDED.tech_text, EXCLUDED.prizes, EXCLUDED.hackathon_date,
                      EXCLUDED.repo_url, EXCLUDED.source)""",
            rows,
        )
        cur.execute('SELECT count(*) FROM corpus_projects')
        added_count = cur.fetchone()[0] - before_count
        cur.execute("INSERT INTO ingestion_runs(trigger_type, source_name, finished_at, added_count, db_size_mb, status) "
                    "VALUES ('load', 'corpus_jsonl', now(), %s, "
                    "pg_total_relation_size('corpus_projects') / 1048576.0, 'complete')", (added_count,))
        conn.commit()
    print(f"loaded {len(records):,} projects and {len(hackathons):,} hackathons")
    print("Next: python build_corpus.py embed   (optional)   or   python build_corpus.py stats")


def embed_text(title, tagline, overview, gap) -> str:
    return "\n".join(p for p in (title, tagline, overview, gap) if p)


def cmd_embed(args: argparse.Namespace) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("pip install sentence-transformers  (needed only for the embed step)")

    model = SentenceTransformer(args.model, backend="onnx")
    dim = model.get_embedding_dimension()
    with _connect(args.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = 'corpus_projects'::regclass AND attname = 'embedding'")
        col_dim = cur.fetchone()[0]
        if col_dim != dim:
            sys.exit(f"model gives {dim} dimensions but the column is vector({col_dim}). "
                     f"Re-run `load --dim {dim}` on an empty table, or ALTER the column.")
        # Bulk updates are much faster without maintaining HNSW for every row.
        cur.execute("DROP INDEX IF EXISTS corpus_projects_vec")
        conn.commit()
        total = 0
        while True:
            cur.execute("SELECT id, title, tagline, overview, gap_highlights FROM corpus_projects "
                        "WHERE embedding IS NULL OR embedding_model <> %s ORDER BY id LIMIT %s", (args.model, args.batch))
            batch = cur.fetchall()
            if not batch:
                break
            vecs = model.encode([embed_text(*b[1:]) for b in batch],
                                normalize_embeddings=True, show_progress_bar=False)
            cur.executemany(
                "UPDATE corpus_projects SET embedding = %s::vector, embedding_model = %s WHERE id = %s",
                [("[" + ",".join(f"{x:.6f}" for x in v) + "]", args.model, b[0]) for v, b in zip(vecs, batch)],
            )
            conn.commit()
            total += len(batch)
            print(f"  embedded {total:,}", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS corpus_projects_vec ON corpus_projects "
                    "USING hnsw (embedding vector_cosine_ops)")
        conn.commit()
    print(f"done: {total:,} embeddings with {args.model} ({dim} dims); HNSW index ready")


def cmd_stats(args: argparse.Namespace) -> None:
    with _connect(args.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*), count(embedding), count(NULLIF(gap_highlights, '')), "
                    "count(repo_url), min(hackathon_date), max(hackathon_date) FROM corpus_projects")
        total, emb, gap, repo, dmin, dmax = cur.fetchone()
        cur.execute("SELECT pg_total_relation_size('corpus_projects'), "
                    "pg_size_pretty(pg_total_relation_size('corpus_projects')), "
                    "pg_size_pretty(pg_relation_size('corpus_projects')), "
                    "pg_size_pretty(pg_indexes_size('corpus_projects'))")
        size_b, size_h, heap_h, idx_h = cur.fetchone()
        print(f"projects: {total:,}   with embedding: {emb:,}   with gap text: {gap:,}   with repo: {repo:,}")
        print(f"dates: {dmin} to {dmax}")
        print(f"table total {size_h}  (heap {heap_h}, indexes {idx_h})")
        if total:
            per_row = size_b / total
            print(f"about {per_row / 1024:.1f} KB per project")
            if total < 1000:
                print("  (fewer than 1,000 rows: fixed index overhead inflates this; "
                      "use `build --limit 1000` for a meaningful benchmark)")
            if args.budget_mb:
                fit = int(args.budget_mb * 1024 * 1024 / per_row)
                print(f"at {args.budget_mb} MB this would hold roughly {fit:,} projects "
                      f"(measure with embeddings loaded for a true number)")
        cur.execute("SELECT source, count(*) FROM corpus_projects GROUP BY 1 ORDER BY 2 DESC")
        for source, n in cur.fetchall():
            print(f"  {n:>8,}  {source}")


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Ignis winners corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download the datasets from Hugging Face")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--skip-alvanlii", action="store_true")
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("build", help="filter, dedupe, clean and parse into corpus.jsonl")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default=str(OUT_FILE))
    p.add_argument("--limit", type=int, help="stop after N unique projects (for the 1,000-row benchmark)")
    p.add_argument("--max-age-years", type=int, default=10, help="drop projects older than this (0 = keep all)")
    p.add_argument("--skip-alvanlii", action="store_true")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("load", help="create tables and load corpus.jsonl into Postgres")
    p.add_argument("--corpus", default=str(OUT_FILE))
    p.add_argument("--dsn")
    p.add_argument("--dim", type=int, default=384, help="embedding dimensions of the vector column")
    p.set_defaults(fn=cmd_load)

    p = sub.add_parser("import-manual", help="append curated JSONL/CSV recognized submissions")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default=str(OUT_FILE))
    p.add_argument("--max-age-years", type=int, default=10)
    p.set_defaults(fn=cmd_import_manual)

    p = sub.add_parser("prune", help="prune expired cache and old corpus records")
    p.add_argument("--dsn")
    p.add_argument("--budget-mb", type=int, default=350)
    p.set_defaults(fn=cmd_prune)

    p = sub.add_parser('takedown', help='remove a submission URL on request')
    p.add_argument('--url', required=True)
    p.add_argument('--dsn')
    p.set_defaults(fn=cmd_takedown)

    p = sub.add_parser("select-budget", help="select high-value rows before loading an oversized corpus")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default=str(OUT_FILE))
    p.add_argument("--budget-mb", type=int, default=350)
    p.add_argument("--kb-per-row", type=float, default=11.1)
    p.add_argument("--safety", type=float, default=.9)
    p.add_argument("--max-rows", type=int)
    p.set_defaults(fn=cmd_select_budget)

    p = sub.add_parser("embed", help="(optional) compute embeddings and build the vector index")
    p.add_argument("--dsn")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--batch", type=int, default=256)
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("stats", help="row counts and table size")
    p.add_argument("--dsn")
    p.add_argument("--budget-mb", type=int)
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
