import asyncio
import datetime as dt
import re
import time
from app.pipeline.repo_url import norm_repo
import httpx
from app.config import settings
from app.pipeline.embedder import encode, cosine

STEPS = (1000, 300, 100, 30)
_response_cache: dict[tuple[str, int, str], tuple[float, list[dict]]] = {}
_cooldown_until = 0.0
_request_gate = asyncio.Lock()
QUALIFIER = re.compile(r'\b(?:stars|pushed|fork|archived):\S+', re.I)
NOISE_WORDS = {'open', 'source', 'github', 'repository', 'repo', 'project', 'app',
               'application', 'system', 'library', 'api', 'and', 'or', 'not',
               'from', 'for', 'with', 'using', 'the', 'in', 'on', 'to', 'of'}

def named_framework_query(idea: str) -> str | None:
    """Preserve a short hyphenated name stated at the beginning of an idea."""
    normalized = re.sub(r'[‐‑‒–—−]', '-', idea.strip())
    match = re.match(r'^([A-Za-z0-9]{2,15}-[A-Za-z0-9]{2,15})(?=\s|\(|:)', normalized)
    return match.group(1) if match else None

def base_queries(queries: list[str], max_terms: int = 3) -> list[str]:
    seen, cleaned = set(), []
    for query in queries:
        query = re.sub(r'[‐‑‒–—−]', '-', query)
        words = [word for word in QUALIFIER.sub('', query).split()
                 if word.casefold().strip('.,') not in NOISE_WORDS]
        base = ' '.join(words[:max_terms])[:120].strip()
        if base and base.casefold() not in seen:
            seen.add(base.casefold())
            cleaned.append(base)
        if len(cleaned) == 6:
            break
    return cleaned

async def search(queries: list[str], workflow_vector: list[float], excluded: set[str], cutoff: float | None = None,
                 max_terms: int = 2):
    global _cooldown_until
    headers = {'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
    if settings.github_token:
        headers['Authorization'] = f'Bearer {settings.github_token}'
    pushed = (dt.date.today() - dt.timedelta(days=365*3)).isoformat()
    pool = {}
    query_hits: dict[str, set[str]] = {}
    partial = False
    tier = None
    rate_limited = False
    blocked = False
    query_terms = base_queries(queries, max_terms)
    api_calls = cache_hits = 0
    tiers_tried = []
    async def fetch(client, query, threshold):
        global _cooldown_until
        nonlocal blocked
        key = (query, threshold, pushed)
        entry = _response_cache.get(key)
        if entry and entry[0] > time.monotonic():
            return entry[1], False, False, 0, 1
        # GitHub asks integrations to issue search requests serially. The gate
        # also lets us honor the search-specific reset header across analyses.
        async with _request_gate:
            entry = _response_cache.get(key)
            if entry and entry[0] > time.monotonic():
                return entry[1], False, False, 0, 1
            if blocked:
                return [], True, True, 0, 0
            wait = _cooldown_until - time.monotonic()
            if wait > 90:
                blocked = True
                return [], True, True, 0, 0
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                resp = await client.get('https://api.github.com/search/repositories', params={
                    'q': f'{query} stars:>{threshold} pushed:>{pushed} fork:false archived:false',
                    'sort': 'best-match', 'per_page': 30})
                if resp.status_code in (403, 429):
                    retry_after = resp.headers.get('Retry-After')
                    try:
                        delay = float(retry_after) if retry_after else 60.0
                    except ValueError:
                        delay = 60.0
                    if resp.headers.get('x-ratelimit-remaining') == '0':
                        try:
                            delay = max(delay, float(resp.headers['x-ratelimit-reset']) - time.time() + 1)
                        except (KeyError, ValueError):
                            pass
                    delay = max(10.0, min(300.0, delay))
                    _cooldown_until = max(_cooldown_until, time.monotonic() + delay)
                    blocked = True
                    return [], True, True, 1, 0
                resp.raise_for_status()
                if resp.headers.get('x-ratelimit-remaining') == '0':
                    try:
                        delay = max(1.0, float(resp.headers['x-ratelimit-reset']) - time.time() + 1)
                        _cooldown_until = max(_cooldown_until, time.monotonic() + delay)
                    except (KeyError, ValueError):
                        pass
                items = resp.json().get('items', [])
                if len(_response_cache) > 1000:
                    _response_cache.clear()
                _response_cache[key] = (time.monotonic() + 900, items)
                return items, False, False, 1, 0
            except (httpx.HTTPError, ValueError):
                return [], True, False, 1, 0

    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        for threshold in STEPS:
            tier = threshold
            tiers_tried.append(threshold)
            results = await asyncio.gather(*(fetch(client, query, threshold) for query in query_terms))
            for query, (items, incomplete, limited, calls, hits) in zip(query_terms, results):
                api_calls += calls
                cache_hits += hits
                partial |= incomplete
                rate_limited |= limited
                for item in items:
                    url = norm_repo(item.get('html_url'))
                    if url and url not in excluded and not item.get('fork') and not item.get('archived'):
                        pool[url] = item
                        query_hits.setdefault(url, set()).add(query)
            if rate_limited:
                break
            # A broad query can fill the pool with popular generic tools at
            # 1000 stars. Continue through all planned tiers so more specific
            # lower-star repositories still get a chance at semantic ranking.
    candidates = list(pool.values())
    stats = {'raw': len(pool), 'star_tier': tier, 'partial': partial,
             'rate_limited': rate_limited,
             'query_terms': query_terms, 'tiers_tried': tiers_tried,
             'api_calls': api_calls, 'cache_hits': cache_hits}
    if not candidates:
        return [], stats
    texts = [f"{r['name']}: {r.get('description') or ''} {' '.join(r.get('topics') or [])}" for r in candidates]
    scores = [cosine(workflow_vector, v) for v in encode(texts)]
    limit = settings.similarity_cutoff if cutoff is None else cutoff
    exact_names = {query.casefold() for query in query_terms if ' ' not in query and '-' in query}
    eligible = sorted(((r, s, max(s, .95) if (r.get('name') or '').casefold() in exact_names else s)
                       for r, s in zip(candidates, scores) if s >= limit or (r.get('name') or '').casefold() in exact_names),
                      key=lambda x: x[2], reverse=True)
    # Cover distinct Analyzer search angles before filling by cosine. Otherwise
    # one popular subtopic can occupy every slot and hide the rest of the idea.
    selected, selected_urls = [], set()
    for query in query_terms:
        for item in eligible:
            url = norm_repo(item[0].get('html_url'))
            if url not in selected_urls and query in query_hits.get(url, set()):
                selected.append(item)
                selected_urls.add(url)
                break
    query_counts = {query: sum(query in query_hits.get(url, set()) for url in selected_urls)
                    for query in query_terms}
    for item in eligible:
        if len(selected) >= 8:
            break
        url = norm_repo(item[0].get('html_url'))
        hits = query_hits.get(url, set())
        if url not in selected_urls and any(query_counts[query] < 2 for query in hits):
            selected.append(item)
            selected_urls.add(url)
            for query in hits:
                query_counts[query] += 1
    for item in eligible:
        if len(selected) >= 8:
            break
        url = norm_repo(item[0].get('html_url'))
        if url not in selected_urls:
            selected.append(item)
            selected_urls.add(url)
    selected.sort(key=lambda item: item[2], reverse=True)
    return [{**r, 'similarity': s, 'screen_score': rank,
             'matched_queries': sorted(query_hits.get(norm_repo(r.get('html_url')), set()))}
            for r, s, rank in selected[:8]], stats
