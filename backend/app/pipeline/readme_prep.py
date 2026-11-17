import asyncio
import base64
import hashlib
import json
import re
from urllib.parse import urlparse
import httpx
from app.db.session import database
from app.pipeline.repo_url import norm_repo
from app.pipeline.text_cleaner import strip_readme
from app.config import settings

async def fetch_tree(client: httpx.AsyncClient, owner: str, repo: str) -> tuple[list[str], bool]:
    """Fetch every tracked path; expand subtrees if GitHub truncates its recursive response."""
    base = f'https://api.github.com/repos/{owner}/{repo}/git/trees/'
    try:
        response = await client.get(base + 'HEAD', params={'recursive': 1})
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return [], False
    paths = {entry['path'] for entry in data.get('tree', []) if entry.get('path')}
    if not data.get('truncated'):
        return sorted(paths), True
    # GitHub limits a recursive tree to 100,000 entries or 7 MB. Walk each
    # subtree when that response is truncated; retain partial paths on failure.
    pending = [('', 'HEAD')]
    requests = 0
    while pending and requests < 1000:
        prefix, sha = pending.pop()
        try:
            response = await client.get(base + sha)
            response.raise_for_status()
            subtree = response.json()
        except (httpx.HTTPError, ValueError):
            return sorted(paths), False
        requests += 1
        if subtree.get('truncated'):
            return sorted(paths), False
        for entry in subtree.get('tree', []):
            name = entry.get('path')
            if not name:
                continue
            path = f'{prefix}/{name}' if prefix else name
            paths.add(path)
            if entry.get('type') == 'tree' and entry.get('sha'):
                pending.append((path, entry['sha']))
    return sorted(paths), not pending

async def ensure_full_tree(url: str, cached: dict) -> dict:
    """Upgrade old shallow cache rows before explaining them."""
    if cached.get('tree_complete') and len(cached['trimmed_readme']) != 2500:
        return cached
    owner, repo = urlparse(url).path.strip('/').split('/')[:2]
    headers = {'Accept': 'application/vnd.github+json'}
    if settings.github_token:
        headers['Authorization'] = f'Bearer {settings.github_token}'
    async with httpx.AsyncClient(timeout=25, headers=headers) as client:
        tree_result, readme_result = await asyncio.gather(
            fetch_tree(client, owner, repo),
            client.get(f'https://api.github.com/repos/{owner}/{repo}/readme'),
            return_exceptions=True)
    paths, complete = tree_result if isinstance(tree_result, tuple) else ([], False)
    if isinstance(readme_result, httpx.Response) and readme_result.is_success:
        raw = base64.b64decode(readme_result.json().get('content', '')).decode('utf-8', errors='replace')
        cleaned = strip_readme(raw, max_chars=10000)
        if cleaned and cleaned != cached['trimmed_readme']:
            cached['trimmed_readme'] = cleaned
            cached['readme_hash'] = hashlib.sha256(cleaned.encode()).hexdigest()
            cached['explainer_json'] = None
    if paths:
        cached['folder_tree'] = paths
        cached['tree_complete'] = complete
        cached['explainer_json'] = None
    cached['readme_hash'] = hashlib.sha256(cached['trimmed_readme'].encode()).hexdigest()
    async with database() as conn:
        async with conn.cursor() as cur:
            await cur.execute('''UPDATE repo_cache SET folder_tree=%s::jsonb, tree_complete=%s,
                trimmed_readme=%s, readme_hash=%s, explainer_json=%s::jsonb WHERE repo_url=%s''',
                (json.dumps(cached['folder_tree']), bool(cached.get('tree_complete')),
                 cached['trimmed_readme'], cached['readme_hash'],
                 json.dumps(cached['explainer_json']) if cached.get('explainer_json') else None, url))
    return cached

async def cache_get(url: str):
    async with database() as conn:
        async with conn.cursor() as cur:
            await cur.execute('SELECT * FROM repo_cache WHERE repo_url=%s AND expires_at>now()', (url,))
            row = await cur.fetchone()
    cached = dict(row) if row else None
    if cached and cached['readme_present']:
        cleaned = strip_readme(cached['trimmed_readme'], max_chars=10000)
        if cleaned != cached['trimmed_readme']:
            cached['trimmed_readme'] = cleaned
            cached['readme_present'] = bool(cleaned)
            cached['readme_hash'] = hashlib.sha256(cleaned.encode()).hexdigest()
            cached['explainer_json'] = None
            async with database() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('UPDATE repo_cache SET trimmed_readme=%s, readme_present=%s, readme_hash=%s, explainer_json=NULL WHERE repo_url=%s',
                                      (cleaned, cached['readme_present'], cached['readme_hash'], url))
    return cached

async def prepare(url: str, fallback: str = '') -> dict:
    url = norm_repo(url)
    if not url:
        return {'trimmed_readme': fallback[:2500], 'folder_tree': [], 'tree_complete': False, 'license_spdx': None, 'repo_url': None}
    cached = await cache_get(url)
    if cached:
        if not cached['readme_present'] and fallback:
            cached['trimmed_readme'] = fallback[:2500]
        return cached
    parts = urlparse(url).path.strip('/').split('/')
    owner, repo = parts[:2]
    headers = {'Accept': 'application/vnd.github+json'}
    if settings.github_token:
        headers['Authorization'] = f'Bearer {settings.github_token}'
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        readme_resp, tree_result, meta_resp = await asyncio.gather(
            client.get(f'https://api.github.com/repos/{owner}/{repo}/readme'),
            fetch_tree(client, owner, repo),
            client.get(f'https://api.github.com/repos/{owner}/{repo}'), return_exceptions=True)
    raw = ''
    if isinstance(readme_resp, httpx.Response) and readme_resp.is_success:
        data = readme_resp.json()
        raw = base64.b64decode(data.get('content', '')).decode('utf-8', errors='replace')
    tree, tree_complete = tree_result if isinstance(tree_result, tuple) else ([], False)
    license_id = None
    if isinstance(meta_resp, httpx.Response) and meta_resp.is_success:
        license_id = (meta_resp.json().get('license') or {}).get('spdx_id')
    cleaned = strip_readme(raw, max_chars=10000)
    trimmed = cleaned or fallback[:2500]
    result = {'repo_url': url, 'trimmed_readme': trimmed, 'folder_tree': tree, 'tree_complete': tree_complete, 'license_spdx': license_id,
              'readme_present': bool(cleaned),
              'is_copyleft': license_id in {'GPL-2.0', 'GPL-3.0', 'AGPL-3.0', 'LGPL-2.1', 'LGPL-3.0'},
              'readme_hash': hashlib.sha256(trimmed.encode()).hexdigest(), 'explainer_json': None}
    async with database() as conn:
        async with conn.cursor() as cur:
            await cur.execute('''INSERT INTO repo_cache(repo_url, trimmed_readme, folder_tree, tree_complete, readme_hash, readme_present, license_spdx, is_copyleft)
                VALUES (%s,%s,%s::jsonb,%s,%s,%s,%s,%s) ON CONFLICT(repo_url) DO UPDATE SET
                trimmed_readme=excluded.trimmed_readme, folder_tree=excluded.folder_tree, tree_complete=excluded.tree_complete, readme_hash=excluded.readme_hash,
                readme_present=excluded.readme_present, license_spdx=excluded.license_spdx, is_copyleft=excluded.is_copyleft, fetched_at=now(),
                expires_at=now()+interval '7 days', explainer_json=NULL''',
                (url, trimmed, json.dumps(tree), tree_complete, result['readme_hash'], result['readme_present'], license_id, result['is_copyleft']))
    return result
