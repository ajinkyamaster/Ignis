import asyncio
import logging
from datetime import date
from app.config import settings
from app.llm.contracts import AnalyzerOutput, ReviewerBatchOutput
from app.llm.roles.analyzer import analyze
from app.llm.roles.reviewer import review
from app.llm.roles.synthesizer import synthesize
from app.llm.router import ModelUnavailable
from app.pipeline.embedder import encode
from app.pipeline import corpus_search, github_search, readme_prep
from app.pipeline.matrix_math import build_coverage_matrix, evidence_supports_feature, minimum_strong_coverage

logger = logging.getLogger(__name__)

def retrieval_text(profile: AnalyzerOutput) -> str:
    """Use the explicit task and features, not only the broad problem statement."""
    parts = [profile.core_workflow, profile.problem]
    parts.extend(term[:80] for term in profile.synonyms[:3])
    parts.extend(feature.name[:80] for feature in profile.features[:8])
    return ' '.join(parts)[:1200]


def public_winner(row: dict, candidate_id: str) -> dict:
    return {'id': candidate_id, 'title': row.get('title'), 'tagline': row.get('tagline'),
            'overview': row.get('overview'), 'gap_highlights': row.get('gap_highlights'),
            'hackathon_name': row.get('hackathon_name'), 'hackathon_date': str(row.get('hackathon_date') or ''),
            'platform': row.get('platform'), 'submission_url': row.get('submission_url'),
            'repo_url': row.get('repo_url'), 'prizes': row.get('prizes'), 'source': row.get('source'),
            'award_scope': 'project-level dataset label; event unverified'
                           if str(row.get('source') or '').startswith('alphahack') else 'event-level dataset label',
            'similarity': float(row.get('similarity') or 0), 'score': float(row.get('score') or 0)}

def public_repo(row: dict, candidate_id: str) -> dict:
    return {'id': candidate_id, 'title': row.get('name'), 'overview': row.get('description'),
            'repo_url': row.get('html_url'), 'stargazers_count': row.get('stargazers_count'),
            'pushed_at': row.get('pushed_at'), 'license_spdx': (row.get('license') or {}).get('spdx_id'),
            'similarity': float(row.get('similarity') or 0), 'screen_score': float(row.get('screen_score') or row.get('similarity') or 0),
            'source': 'GitHub API'}

async def run(profile: AnalyzerOutput, idea: str, mode: str, problem_statement: str = '', track: str = ''):
    yield 'waking_db', {'status': 'Connecting to corpus and cache'}
    yield 'embedding_started', {'model': settings.embedding_model}
    vector = (await asyncio.to_thread(encode, [retrieval_text(profile)]))[0]
    yield 'embedding_ready', {'dimensions': len(vector)}
    winners, stats, partial = [], {}, []
    if mode == 'hackathon':
        yield 'corpus_search_started', {'method': 'full-text + vector reciprocal-rank fusion'}
        try:
            winners = await corpus_search.search(profile, vector, settings.embedding_model, theme=track or None)
            stats = await corpus_search.coverage_stats()
        except Exception:
            partial.append('Winner corpus unavailable')
        yield 'corpus_search_finished', {'count': len(winners), 'partial': 'Winner corpus unavailable' in partial}
    excluded = {x for x in (github_search.norm_repo(w.get('repo_url')) for w in winners) if x}
    named_query = github_search.named_framework_query(idea)
    github_queries = ([named_query] if named_query else []) + profile.github_queries
    yield 'github_search_started', {'queries': len(github_search.base_queries(github_queries))}
    try:
        repos, gh_stats = await github_search.search(github_queries, vector, excluded)
    except Exception:
        repos, gh_stats = [], {'raw': 0, 'star_tier': None, 'partial': True}
    if gh_stats['partial']:
        partial.append('GitHub search was incomplete')
    yield 'github_search_finished', {'count': len(repos), 'star_tier': gh_stats['star_tier'], 'partial': gh_stats['partial']}
    if len(repos) < 2 and not gh_stats['partial']:
        yield 'github_retry_started', {'reason': 'Few repository candidates found'}
        try:
            retry = await analyze(idea, mode, problem_statement, track, profile.github_queries)
            more, retry_stats = await github_search.search(retry.github_queries, vector, excluded, max_terms=2)
            known = {github_search.norm_repo(r.get('html_url')) for r in repos}
            repos = sorted(repos + [r for r in more if github_search.norm_repo(r.get('html_url')) not in known],
                           key=lambda r: r.get('similarity') or 0, reverse=True)[:8]
            gh_stats['raw'] += retry_stats['raw']
            gh_stats['partial'] |= retry_stats['partial']
            if retry_stats['partial']:
                partial.append('GitHub retry search was incomplete')
        except Exception:
            partial.append('GitHub retry unavailable')
        yield 'github_retry_finished', {'count': len(repos)}
    screened = ([{'title': w.get('title'), 'url': w.get('submission_url'), 'source': 'winner'} for w in winners]
                + [{'title': r.get('name'), 'url': r.get('html_url'), 'source': 'github'} for r in repos])
    yield 'candidates_screened', {'winners': len(winners), 'github': len(repos), 'star_tier': gh_stats['star_tier'],
                                  'partial': partial, 'screened': screened}
    if not winners and not repos:
        yield 'early_exit', {'message': 'No close prior art found in the available sources. This is not proof that none exists.',
                             'coverage': {'corpus': stats, 'github_raw': gh_stats['raw']}, 'partial': partial}
        return
    candidates = {}
    work = []
    for i, row in enumerate(winners):
        cid = f'cor_{i+1}'
        item = public_winner(row, cid)
        candidates[cid] = item
        fallback = ('Current project overview:\n' + (row.get('overview') or '') +
                    '\n\nStated challenges and future plans (not implemented features):\n' +
                    (row.get('gap_highlights') or ''))
        work.append((cid, row.get('repo_url'), fallback))
    for i, row in enumerate(repos):
        cid = f'gh_{i+1}'
        candidates[cid] = public_repo(row, cid)
        work.append((cid, row.get('html_url'), row.get('description') or ''))
    yield 'evidence_fetch_started', {'count': len(work)}
    prepared = await asyncio.gather(*(readme_prep.prepare(url, fallback) for _, url, fallback in work), return_exceptions=True)
    evidence = []
    for (cid, _, fallback), result in zip(work, prepared):
        if isinstance(result, Exception):
            result = {'trimmed_readme': fallback[:2500], 'folder_tree': [], 'license_spdx': None}
            partial.append(f'{cid} README unavailable')
        if cid.startswith('gh_') and not result.get('readme_present'):
            candidates.pop(cid, None)
            continue
        candidates[cid]['license_spdx'] = result.get('license_spdx') or candidates[cid].get('license_spdx')
        candidates[cid]['is_copyleft'] = candidates[cid]['license_spdx'] in {
            'GPL-2.0', 'GPL-3.0', 'AGPL-3.0', 'LGPL-2.1', 'LGPL-3.0'
        }
        candidates[cid]['evidence_basis'] = 'GitHub README' if result.get('readme_present') else 'Submission overview'
        candidates[cid]['evidence_excerpt'] = result['trimmed_readme'][:500]
        evidence.append({'candidate_id': cid, 'title': candidates[cid]['title'], 'source': candidates[cid]['source'],
                         'text': result['trimmed_readme'][:2500], 'documentation_depth': min(len(result['trimmed_readme']), 2500),
                         'evidence_basis': candidates[cid]['evidence_basis']})
    yield 'evidence_fetch_finished', {'usable': len(evidence), 'requested': len(work)}
    if not evidence:
        yield 'early_exit', {'message': 'No close prior art with usable documentation found in the available sources.',
                             'coverage': {'corpus': stats, 'github_raw': gh_stats['raw']}, 'partial': partial}
        return
    # The planned path uses one Reviewer call for Normal mode and at most two
    # for the usual 12 winner + 8 repository Hackathon shortlist. The router
    # compacts fallback evidence; split further only if a provider rejects a
    # large or invalid batch.
    batches = [evidence[i:i + 12] for i in range(0, len(evidence), 12)]
    judgments = []
    evidence_by_id = {item['candidate_id']: item for item in evidence}
    batch_index = 0
    while batch_index < len(batches):
        batch = batches[batch_index]
        if not batch:
            batch_index += 1
            continue
        yield 'review_batch_started', {'batch': batch_index + 1, 'total_batches': len(batches), 'candidates': len(batch)}
        try:
            output: ReviewerBatchOutput = await review(profile, batch)
        except ModelUnavailable as exc:
            retryable_size = any(marker in str(exc) for marker in ('HTTP 413', 'HTTP 400', 'ValueError', 'ValidationError'))
            if len(batch) > 6 and retryable_size:
                middle = len(batch) // 2
                batches[batch_index:batch_index + 1] = [batch[:middle], batch[middle:]]
                yield 'review_batch_split', {'candidates': len(batch), 'new_batches': 2,
                                             'reason': 'Provider rejected the larger Reviewer batch'}
                continue
            logger.exception('Reviewer providers unavailable after batch %s', batch_index + 1)
            matrix = build_coverage_matrix(profile.features, judgments, candidates, mode)
            yield 'review_unavailable', {'message': 'Reviewer providers are rate-limited or could not accept the request. '
                                                    'Screened projects are listed below, but have not all been reviewed. Try again after provider quota resets.',
                                         'matrix': matrix, 'reviewed_candidates': len(judgments),
                                         'screened_candidates': len(evidence), 'partial': partial}
            return
        allowed = {x['candidate_id'] for x in batch}
        seen = set()
        for j in output.judgments:
            if j.candidate_id not in allowed or j.candidate_id in seen:
                continue
            document = evidence_by_id[j.candidate_id]
            proposed_ids = set(j.covered_feature_ids)
            j.covered_feature_ids = sorted(f.id for f in profile.features
                                           if f.id in proposed_ids
                                           and evidence_supports_feature(f.name, document['text'], f.description))
            if j.label == 'Strong' and not minimum_strong_coverage(len(j.covered_feature_ids), len(profile.features)):
                j.label = 'Related'
                j.one_line_reason = ('The supplied evidence does not establish substantially the same full workflow. '
                                     + j.one_line_reason)
            rejected = [f.name for f in profile.features
                        if f.id in proposed_ids and f.id not in j.covered_feature_ids]
            if rejected:
                supported = [f.name for f in profile.features if f.id in j.covered_feature_ids]
                unconfirmed = [f.name for f in profile.features if f.id not in j.covered_feature_ids]
                j.one_line_reason = (
                    ('The supplied evidence supports ' + ', '.join(supported) + '. ' if supported else '')
                    + 'It does not establish ' + ', '.join(unconfirmed) + '.'
                )
            if document['evidence_basis'] != 'GitHub README' or document['documentation_depth'] < 200:
                j.confidence = 'Low'
            elif document['documentation_depth'] < 800 and j.confidence == 'High':
                j.confidence = 'Medium'
            judgments.append(j)
            seen.add(j.candidate_id)
        yield 'review_batch_finished', {'batch': batch_index + 1, 'total_batches': len(batches), 'reviewed': len(judgments)}
        batch_index += 1
    matrix = build_coverage_matrix(profile.features, judgments, candidates, mode)
    yield 'review_ready', {'cards': matrix['cards'], 'counts': matrix['counts'],
                           'uncovered_features': matrix['uncovered_features'], 'partial': partial,
                           'screened_candidates': len(screened), 'reviewed_candidates': len(judgments),
                           'confirmed_matches': len(matrix['cards'])}
    if not matrix['cards']:
        yield 'early_exit', {'message': 'No close prior art was confirmed in the reviewed evidence. This is not proof that none exists.',
                             'coverage': {'corpus': stats, 'github_raw': gh_stats['raw'], 'reviewed_candidates': len(evidence)},
                             'partial': partial}
        return
    compact = [{k: c[k] for k in ('id', 'label', 'confidence', 'addons', 'shared')} |
               {'title': c['project'].get('title'), 'evidence_basis': c['project'].get('evidence_basis'),
                'reason': c['reason'][:240],
                'not_mentioned_in_evidence': c['lacking'],
                'gap_highlights': (c['project'].get('gap_highlights') or '')[:180],
                'license_spdx': c['project'].get('license_spdx')} for c in matrix['cards']]
    yield 'synthesis_started', {'reviewed_cards': len(matrix['cards'])}
    try:
        verdict = await synthesize({'mode': mode, 'profile': {key: getattr(profile, key) for key in
                                                             ('problem', 'target_users', 'core_workflow')} |
                                    {'features': [feature.model_dump() for feature in profile.features]},
                                    'strong_match_count': sum(c['label'] == 'Strong' for c in matrix['cards']),
                                    'coverage_counts': matrix['counts'],
                                    'uncovered_features': matrix['uncovered_features'], 'candidates': compact})
    except ModelUnavailable:
        logger.exception('Synthesizer providers unavailable')
        yield 'synthesis_unavailable', {'message': 'Projects were reviewed, but the verdict model is temporarily unavailable. '
                                                  'The evidence cards and coverage matrix remain available.',
                                        'matrix': matrix, 'partial': partial}
        return
    yield 'synthesis_finished', {'direction': verdict.verdict_direction}
    verdict_data = verdict.model_dump()
    names = {candidate['id']: candidate['title'] or candidate['id'] for candidate in compact}
    for cluster in verdict_data['addon_clusters']:
        cluster['seen_in_projects'] = [names[candidate_id] for candidate_id in cluster['seen_in_projects']]
    yield 'complete', {'verdict': verdict_data, 'matrix': matrix, 'partial': partial,
                       'coverage': {'corpus': stats, 'github_raw': gh_stats['raw']}}
