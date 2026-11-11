"""Postgres FTS + vector reciprocal-rank fusion; all ranking is deterministic."""
import re
from app.db.session import database

STOPWORDS = {'app', 'application', 'system', 'platform', 'tool', 'solution', 'user', 'users',
             'using', 'based', 'with', 'from', 'into', 'that', 'this', 'their', 'them',
             'create', 'creator', 'automatic', 'automatically', 'help', 'helps',
             'note', 'notes', 'extractor', 'video', 'educational', 'student', 'students',
             'professional', 'professionals', 'self', 'time', 'inference',
             'question', 'questions', 'answer', 'answers', 'management', 'manager'}


def lexical_query(profile) -> str:
    """Cover several distinct profile angles without turning them into an AND gate."""
    seen, terms = set(), []
    for phrase_index, phrase in enumerate(list(profile.synonyms[:5]) + [profile.core_workflow, profile.problem]):
        added = 0
        for word in re.findall(r'[A-Za-z0-9]+', phrase.lower()):
            if len(word) < 3 or word in STOPWORDS or word in seen:
                continue
            seen.add(word)
            terms.append(word)
            added += 1
            if added >= (3 if phrase_index == 0 else 2) or len(terms) >= 6:
                break
        if len(terms) >= 6:
            break
    return ' OR '.join(terms) or 'project'

TITLE_BOOST = "(1 + CASE WHEN to_tsvector('english', coalesce(title,'')) @@ websearch_to_tsquery('english', %(query)s) THEN .35 ELSE 0 END)"

SQL = '''WITH fts AS (
 SELECT id, row_number() OVER (ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', %(query)s)) * ''' + TITLE_BOOST + ''' DESC, id ASC) AS rank
 FROM corpus_projects
 WHERE search_vector @@ websearch_to_tsquery('english', %(query)s)
 AND hackathon_date >= CURRENT_DATE - INTERVAL '10 years'
 AND (%(platform)s::text IS NULL OR platform = %(platform)s::text)
 AND (%(date_from)s::date IS NULL OR hackathon_date >= %(date_from)s::date)
 ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', %(query)s)) * ''' + TITLE_BOOST + ''' DESC, id ASC LIMIT 40
), vec AS (
 SELECT id, row_number() OVER (ORDER BY embedding <=> %(vector)s::vector) AS rank,
 1 - (embedding <=> %(vector)s::vector) AS similarity
 FROM corpus_projects WHERE embedding IS NOT NULL AND embedding_model = %(model)s
 AND hackathon_date >= CURRENT_DATE - INTERVAL '10 years'
 AND (%(platform)s::text IS NULL OR platform = %(platform)s::text)
 AND (%(date_from)s::date IS NULL OR hackathon_date >= %(date_from)s::date)
 ORDER BY embedding <=> %(vector)s::vector LIMIT 40
), scores AS (
 SELECT coalesce(fts.id, vec.id) AS id,
 coalesce(1.0/(60+fts.rank),0) + coalesce(1.0/(60+vec.rank),0) AS rrf,
 coalesce(vec.similarity,0) AS similarity
 FROM fts FULL OUTER JOIN vec ON fts.id=vec.id
)
SELECT p.*, h.name AS hackathon_name, scores.similarity,
 scores.rrf * (1 + CASE WHEN p.hackathon_date >= CURRENT_DATE - INTERVAL '3 years' THEN .05 ELSE 0 END
 + CASE WHEN %(theme)s::text IS NOT NULL AND EXISTS (
   SELECT 1 FROM unnest(h.themes) AS tag WHERE tag ILIKE '%%' || %(theme)s::text || '%%'
 ) THEN .05 ELSE 0 END) AS score
FROM scores JOIN corpus_projects p ON p.id=scores.id
LEFT JOIN corpus_hackathons h ON h.id=p.hackathon_id
ORDER BY score DESC, p.id ASC LIMIT 24'''


def distinct_projects(rows: list[dict], limit: int = 12) -> list[dict]:
    """Collapse near-identical Devpost submissions with different project URLs."""
    chosen = []
    for row in rows:
        title = (row.get('title') or '').strip().casefold()
        words = set(re.findall(r'[a-z0-9]+', (row.get('overview') or '').casefold()))
        duplicate = False
        if len(title) >= 18 and len(words) >= 10:
            for prior in chosen:
                if title != (prior.get('title') or '').strip().casefold():
                    continue
                other = set(re.findall(r'[a-z0-9]+', (prior.get('overview') or '').casefold()))
                if other and len(words & other) / len(words | other) >= .75:
                    duplicate = True
                    break
        if not duplicate:
            chosen.append(row)
            if len(chosen) >= limit:
                break
    return chosen

async def search(profile, vector: list[float], model: str, *, platform: str | None = None,
                 date_from: str | None = None, theme: str | None = None) -> list[dict]:
    query = lexical_query(profile)
    vec = '[' + ','.join(f'{x:.6f}' for x in vector) + ']'
    async with database() as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL, {'query': query, 'vector': vec, 'model': model,
                                    'platform': platform, 'date_from': date_from, 'theme': theme})
            rows = await cur.fetchall()
    return distinct_projects([dict(row) for row in rows])

async def coverage_stats() -> dict:
    async with database() as conn:
        async with conn.cursor() as cur:
            await cur.execute('SELECT count(*) AS winners, count(DISTINCT hackathon_id) AS hackathons, min(hackathon_date) AS oldest, max(hackathon_date) AS newest FROM corpus_projects')
            return dict(await cur.fetchone())
