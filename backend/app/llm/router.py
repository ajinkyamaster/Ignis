"""Two strong-model pools with bounded cross-provider failover and strict JSON validation."""
import asyncio
import json
import logging
import re
import time
from contextvars import ContextVar
from collections.abc import Callable
import httpx
from pydantic import BaseModel, ValidationError
from app.config import settings

class ModelUnavailable(RuntimeError):
    pass

logger = logging.getLogger(__name__)
provider_events: ContextVar[Callable[[str, dict], None] | None] = ContextVar('provider_events', default=None)
_provider_cooldown: dict[str, float] = {}

def emit_provider_event(name: str, payload: dict) -> None:
    callback = provider_events.get()
    if callback:
        callback(name, payload)

def retry_seconds(response: httpx.Response) -> float:
    header = response.headers.get('Retry-After', '')
    try:
        return min(90.0, max(1.0, float(header)))
    except ValueError:
        pass
    match = re.search(r'(?:try again|retry) in\s+([\d.]+)\s*s', response.text, re.I)
    return min(90.0, max(2.0, float(match.group(1)) + 2)) if match else 60.0

def is_daily_quota(response: httpx.Response) -> bool:
    body = response.text.lower()
    return any(marker in body for marker in ('requestsperday', 'requests_per_day', 'per_day', 'daily quota', 'tokens_per_day'))

def compact_reviewer_payload(payload: dict) -> dict:
    profile = payload['profile']
    terms = {word.casefold() for feature in profile['features']
             for word in re.findall(r'[A-Za-z0-9]+', feature['name'] + ' ' + feature['description'])
             if len(word) >= 5 and word.casefold() not in {'using', 'based', 'their', 'users',
                                                            'feature', 'system', 'project', 'provide'}}
    def excerpt(source: str) -> str:
        current = source.split('Stated challenges and future plans', 1)[0]
        if len(current) <= 950:
            return current
        head = current[:400]
        windows = []
        for term in terms:
            for match in list(re.finditer(r'\b' + re.escape(term) + r'\w*\b', current[400:], re.I))[:3]:
                start = max(400, 400 + match.start() - 90)
                snippet = current[start:min(len(current), start + 230)].strip()
                score = sum(bool(re.search(r'\b' + re.escape(other) + r'\w*\b', snippet, re.I))
                            for other in terms)
                windows.append((score, start, snippet))
        chosen = []
        for _, start, snippet in sorted(windows, key=lambda item: (-item[0], item[1])):
            if any(abs(start - prior) < 180 for prior, _ in chosen):
                continue
            chosen.append((start, snippet))
            if len(chosen) >= 2:
                break
        return (head + ''.join('\n… ' + snippet for _, snippet in sorted(chosen)))[:950]
    return {'profile': {key: profile[key] for key in ('problem', 'target_users', 'core_workflow', 'features')},
            'candidates': [{'candidate_id': c['candidate_id'], 'title': c['title'],
                            'evidence_basis': c['evidence_basis'], 'documentation_depth': c['documentation_depth'],
                            'text': excerpt(c['text'])} for c in payload['candidates']]}

def strict_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    def close_objects(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                node['additionalProperties'] = False
                node['required'] = list(node.get('properties', {}))
            for child in node.values():
                close_objects(child)
        elif isinstance(node, list):
            for child in node:
                close_objects(child)
    close_objects(schema)
    return schema

async def structured(role: str, system: str, payload: dict, schema: type[BaseModel],
                     validate_result: Callable[[BaseModel], None] | None = None):
    primary = ([('gemini', settings.reviewer_model), ('groq', settings.reviewer_fallback_model),
                ('groq', settings.reasoning_model)] if role == 'reviewer' else
               [('groq', settings.reasoning_model), ('gemini', settings.reviewer_model),
                ('groq', settings.reviewer_fallback_model)])
    failures = []
    output_schema = strict_schema(schema)
    active_payload = payload
    async with httpx.AsyncClient(timeout=45) as client:
        for provider, model in primary:
            key = settings.gemini_key if provider == 'gemini' else settings.groq_key
            if not key:
                continue
            cooldown = _provider_cooldown.get(provider, 0) - time.monotonic()
            if cooldown > 0:
                # A daily Gemini quota will not recover during this analysis.
                if provider == 'gemini':
                    emit_provider_event('provider_skipped', {'role': role, 'provider': provider, 'reason': 'daily_quota'})
                    continue
                emit_provider_event('provider_retry_wait', {'role': role, 'provider': provider, 'seconds': round(cooldown)})
                await asyncio.sleep(cooldown)
            for attempt in range(2):
                try:
                    logger.info('LLM request role=%s provider=%s model=%s attempt=%s', role, provider, model, attempt + 1)
                    emit_provider_event('provider_call_started', {'role': role, 'provider': provider, 'model': model, 'attempt': attempt + 1})
                    corrections = {
                        'analyzer': 'Use literal Unicode → arrows in core_workflow, contiguous feature IDs, and distinct search queries.',
                        'reviewer': 'Return exactly one judgment for every supplied candidate_id, with no missing, duplicate, or invented IDs.',
                        'synthesizer': 'Copy each addon_clusters.features string exactly from an addons list of a project in seen_in_projects. Never list user features or shared features as add-ons.',
                        'explainer': 'Workflow and concepts must cite README facts only. Do not mention any filename absent from the README there; folder roles may infer paths. Language selection does not prove translation.',
                    }
                    correction = '\nCorrection: ' + corrections.get(role, 'Follow the schema exactly.') if attempt else ''
                    provider_payload = compact_reviewer_payload(active_payload) if role == 'reviewer' else active_payload
                    if provider == 'groq':
                        request_body = {
                            'model': model, 'temperature': 0,
                            'response_format': {'type': 'json_schema', 'json_schema': {'name': schema.__name__, 'strict': True, 'schema': output_schema}},
                            'messages': [{'role': 'system', 'content': system + correction}, {'role': 'user', 'content': json.dumps(provider_payload)}]}
                        response = await client.post('https://api.groq.com/openai/v1/chat/completions', headers={'Authorization': f'Bearer {key}'}, json=request_body)
                        if response.status_code == 400 and 'response_format' in response.text.lower():
                            # Some Groq models reject strict schema mode; JSON object mode
                            # still goes through the same Pydantic and semantic validators.
                            request_body['response_format'] = {'type': 'json_object'}
                            emit_provider_event('provider_format_fallback', {'role': role, 'provider': provider, 'model': model})
                            response = await client.post('https://api.groq.com/openai/v1/chat/completions', headers={'Authorization': f'Bearer {key}'}, json=request_body)
                        response.raise_for_status()
                        raw = response.json()['choices'][0]['message']['content']
                    else:
                        response = await client.post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent', params={'key': key}, json={
                            'systemInstruction': {'parts': [{'text': system + (
                                '\nUse feature IDs 1, 2, 3, etc. in order and a literal input → processing → output workflow.'
                                if role == 'analyzer' else '') + correction}]},
                            'contents': [{'role': 'user', 'parts': [{'text': json.dumps(active_payload)}]}],
                            'generationConfig': {'temperature': 0, 'responseMimeType': 'application/json', 'responseJsonSchema': output_schema}})
                        response.raise_for_status()
                        raw = response.json()['candidates'][0]['content']['parts'][0]['text']
                    result = schema.model_validate_json(raw)
                    if validate_result:
                        validate_result(result)
                    logger.info('LLM validated role=%s provider=%s model=%s', role, provider, model)
                    emit_provider_event('provider_call_succeeded', {'role': role, 'provider': provider, 'model': model})
                    return result
                except (httpx.HTTPError, KeyError, IndexError, ValidationError, ValueError) as exc:
                    status = f' HTTP {exc.response.status_code}' if isinstance(exc, httpx.HTTPStatusError) else ''
                    failures.append(f'{provider}: {type(exc).__name__}{status}')
                    logger.warning('LLM rejected role=%s provider=%s model=%s error=%s%s',
                                   role, provider, model, type(exc).__name__, status)
                    code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                    reason = ('rate_limit' if code == 429 else 'request_too_large' if code == 413
                              else 'provider_error' if code else 'invalid_response')
                    emit_provider_event('provider_call_failed', {'role': role, 'provider': provider,
                                                               'model': model, 'reason': reason, 'attempt': attempt + 1})
                    if code == 429:
                        if provider == 'gemini' and is_daily_quota(exc.response):
                            _provider_cooldown[provider] = time.monotonic() + 6 * 3600
                            emit_provider_event('provider_skipped', {'role': role, 'provider': provider, 'reason': 'daily_quota'})
                            break
                        delay = retry_seconds(exc.response)
                        if provider == 'groq':
                            # A token window can be longer than the short retry hint.
                            delay = max(60.0, delay)
                        _provider_cooldown[provider] = time.monotonic() + delay
                    if code == 413 and role == 'explainer' and attempt == 0:
                        active_payload = dict(active_payload,
                                              trimmed_readme=active_payload['trimmed_readme'][:3500],
                                              folder_tree=active_payload['folder_tree'][:35])
                        emit_provider_event('provider_context_reduced', {'role': role, 'provider': provider,
                                                                         'reason': 'request_too_large'})
                        continue
                    if isinstance(exc, httpx.HTTPStatusError) and code not in (429, 500, 502, 503, 504):
                        break
                    if attempt == 0:
                        delay = max(0.0, _provider_cooldown.get(provider, 0) - time.monotonic()) if code == 429 else 1.0
                        if delay > 1:
                            emit_provider_event('provider_retry_wait', {'role': role, 'provider': provider,
                                                                         'seconds': round(delay)})
                        await asyncio.sleep(delay)
    raise ModelUnavailable('; '.join(failures) or 'No LLM provider key configured')
