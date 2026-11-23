import re
import math

from app.llm.contracts import NumberedFeature, CandidateJudgment

TIERS = {'Strong': 0, 'Related': 1}
GENERIC_FEATURE_WORDS = {'data', 'feature', 'features', 'system', 'engine', 'interface',
                         'integration', 'processing', 'generation', 'automatic', 'automated',
                         'app', 'application', 'support', 'platform', 'option', 'module', 'tool'}


def minimum_strong_coverage(covered: int, total: int) -> bool:
    """A Strong label needs substantial verified feature coverage at minimum."""
    return covered >= math.ceil(total * .6)


def compound_feature_supported(name: str, evidence: str) -> bool:
    """Require source text for each side of an explicitly combined feature name."""
    parts = re.split(r'\s+(?:&|and|\+)\s+', name, flags=re.I)
    if len(parts) < 2:
        return True
    words = re.findall(r'[a-z0-9]+', evidence.lower())
    for part in parts:
        terms = [word for word in re.findall(r'[a-z0-9]+', part.lower())
                 if len(word) >= 3 and word not in GENERIC_FEATURE_WORDS]
        if not terms:
            continue
        if not any(
            (term == 'lms' and (re.search(r'\blms\b|learning management system|\bcanvas\b|\bmoodle\b|\bblackboard\b', evidence, re.I)))
            or any(word.startswith(term[:5]) for word in words)
            for term in terms
        ):
            return False
    return True


def evidence_supports_feature(name: str, evidence: str, description: str = '') -> bool:
    """Conservative checks for capabilities often inferred from adjacent features."""
    current_evidence = evidence.split('Stated challenges and future plans', 1)[0]
    if not compound_feature_supported(name, current_evidence):
        return False
    feature = name.casefold()
    source = current_evidence.casefold()
    requested = (name + ' ' + description).casefold()
    if re.search(r'\b(?:safe|safety)\b', requested) and not re.search(
            r'\b(?:check\w*|validat\w*|verif\w*|screen\w*|enforc\w*)\b'
            r'[^.!?\n]{0,50}\b(?:expir\w*|freshness|food safety|safe to eat|temperature)\b'
            r'|\b(?:expir\w*|freshness|food safety|safe to eat|temperature)\b'
            r'[^.!?\n]{0,50}\b(?:check\w*|validat\w*|verif\w*|screen\w*|enforc\w*)\b'
            r'|\b(?:food safety compliance|temperature control)\b', source):
        # Fresh produce or a freshness listing does not prove safety checks.
        return False
    if re.search(r'\bpredicted?\b', requested) and not re.search(
            r'\b(?:predict\w*|forecast\w*)\b', source):
        return False
    if (re.search(r'\b(?:surplus|excess)\b', requested)
            and re.search(r'\b(?:predict\w*|forecast\w*)\b', requested)
            and not (re.search(r'\b(?:surplus|excess)\b', source)
                     and re.search(r'\b(?:predict\w*|forecast\w*)\b', source))):
        # Forecasting demand or order quantities is adjacent, but does not
        # establish forecasting surplus food available for donation.
        return False
    if re.search(r'\binventory\b', requested) and re.search(r'\bscan\w*\b', requested):
        if not re.search(
                r'\b(?:inventory|stock)\b[^.!?\n]{0,80}\b(?:scan\w*|barcode|camera|image\w*)\b'
                r'|\b(?:scan\w*|barcode|camera|image\w*)\b[^.!?\n]{0,80}\b(?:inventory|stock)\b', source):
            return False
    if re.search(r'\bflashcards?\b', requested) and re.search(r'\b(?:generat\w*|automat\w*|creat\w*)\b', requested):
        # Manual card entry and generic "knowledge cards" do not prove
        # automatic flashcard creation from the user's materials.
        if not re.search(
            r'\b(?:generat\w*|automat\w*|ai[- ]powered|nlp[- ]powered|convert\w*)\b'
            r'[^.!?\n]{0,90}\bflashcards?\b'
            r'|\bflashcards?\b[^.!?\n]{0,90}\b(?:generat\w*|automat\w*|generator)\b'
            r'|\bflashcard generator\b', source):
            return False
    if re.search(r'\bremind\w*\b', requested) and not re.search(
            r'\b(?:remind\w*|notif\w*|alert\w*)\b', source):
        # Spaced-repetition scheduling alone does not establish reminders.
        return False
    if ('wheelchair' in requested and re.search(r'\b(?:route|routing|journey|trip|path)\b', requested)
            and not re.search(r'\b(?:wheelchair|step[- ]free|curb[- ]cut)\b', source)):
        # Generic transit routing does not prove wheelchair-usable journeys.
        return False
    if (re.search(r'\btranscri\w*\b', requested)
            and re.search(r'\b(?:audio|speech|videos?|recordings?)\b', requested)
            and re.search(r'\b(?:creat\w*|convert\w*|generat\w*|transcri\w*)\b', requested)):
        # Downloading pre-existing captions is not evidence of speech-to-text.
        non_negated = re.sub(r'\b(?:no|without|not using)\s+whisper\b', '', source)
        asr = re.search(
            r'\b(?:speech[- ]to[- ]text|automatic speech recognition|asr|speech recognition|whisper)\b'
            r'|\btranscrib\w*.{0,80}\b(?:audio|speech|video|recording|files?)\b'
            r'|\b(?:audio|speech|video|recording|files?)\b.{0,80}\btranscrib\w*\b'
            r'|\bgenerat\w* transcripts? from (?:audio|video)\b', non_negated, re.S)
        if not asr:
            return False
    if 'concept' in feature and ('extract' in feature or 'identif' in feature):
        if not re.search(r'\b(?:key concepts?|main ideas?|key ideas?|keywords?|concept extraction)\b', source):
            return False
    if 'search' in feature and ('transcript' in feature or 'caption' in feature):
        if 'keyword' in requested and not re.search(
                r'\bkeywords?\b|\bfull[- ]text\b|\bexact[- ]match\b', source):
            # Semantic search over derived content is not evidence of the
            # specifically requested keyword search.
            return False
        if not re.search(r'\b(?:search|searchable|find|lookup|query)\b', source):
            return False
        if not re.search(r'\b(?:transcript|caption|subtitle)s?\b', source):
            return False
        # A searchable app and a transcript mentioned elsewhere do not prove
        # that the transcript itself can be searched.
        direct_search = re.finditer(
            r'\b(?:search(?:able)?|find|lookup|query)\b([^.!?\n]{0,80})\b(?:transcript|caption|subtitle)s?\b'
            r'|\b(?:transcript|caption|subtitle)s?\b([^.!?\n]{0,80})\b(?:search(?:able)?|find|lookup|query)\b',
            source,
        )
        if not any(not re.search(r'\b(?:notes?|markdown|into)\b',
                                 match.group(1) or match.group(2) or '')
                   for match in direct_search):
            return False
    return True


def scope_reason(reason: str) -> str:
    """Avoid claiming an undocumented capability is definitely absent."""
    reason = re.sub(r'\b(?:lacks|doesn.t handle|does not handle|doesn.t offer|does not offer|does not include)\b',
                    'is not shown by the supplied evidence to provide', reason, flags=re.I)
    reason = re.sub(r'\bbut not the full\b',
                    'but the supplied evidence does not establish the full', reason, flags=re.I)
    return reason

def build_coverage_matrix(features: list[NumberedFeature], judgments: list[CandidateJudgment], candidates: dict[str, dict], mode: str):
    ids = {f.id for f in features}
    kept = [j for j in judgments if j.label in TIERS and j.candidate_id in candidates
            and ids.intersection(j.covered_feature_ids)]
    def rank(j):
        c = candidates[j.candidate_id]
        return (TIERS[j.label], -len(ids.intersection(j.covered_feature_ids)),
                -float(c.get('screen_score') or c.get('similarity') or c.get('score') or 0),
                -int(c.get('stargazers_count') or 0))
    winners = sorted((j for j in kept if j.candidate_id.startswith('cor_')), key=rank)[:10] if mode == 'hackathon' else []
    repos = sorted((j for j in kept if j.candidate_id.startswith('gh_')), key=rank)[:5]
    active = winners + repos
    cards = []
    for j in active:
        covered = ids.intersection(j.covered_feature_ids)
        item = candidates[j.candidate_id]
        cards.append({'id': j.candidate_id, 'source': 'winner' if j.candidate_id.startswith('cor_') else 'github',
                      'label': j.label, 'confidence': j.confidence, 'reason': scope_reason(j.one_line_reason),
                      'shared': [f.name for f in features if f.id in covered],
                      'lacking': [f.name for f in features if f.id not in covered],
                      'covered_feature_ids': sorted(covered), 'addons': j.addons, 'project': item})
    counts = {f.id: sum(f.id in set(j.covered_feature_ids) for j in active) for f in features}
    return {'cards': cards, 'counts': counts,
            'uncovered_features': [f.name for f in features if counts[f.id] == 0]}
