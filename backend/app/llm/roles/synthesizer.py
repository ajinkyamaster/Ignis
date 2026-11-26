import re

from app.llm.contracts import SynthesizerOutput
from app.llm.router import structured

SYSTEM = '''You are the Ignis Synthesizer. Use only structured profile, computed coverage counts, strong_match_count, kept candidate summaries, gaps, and addons. Cluster synonymous addons. Each addon_clusters.features string must be copied exactly from one candidate's addons list; never put a user-requested feature or a shared feature there. In seen_in_projects, use only the exact candidate IDs whose addons contain the listed feature. Explain crowdedness, evidence-backed opportunities, and a practical direction. In Hackathon mode state the number of strong matches in natural prose; never print JSON field names. In Normal mode write a concise gap summary. Describe only the reviewed sample; do not call it the entire market or landscape. A feature listed as not_mentioned_in_evidence or having zero coverage was not found in reviewed evidence; never claim it is definitively absent from the project or the wider world. Identify suggestions as suggestions. If suggesting code reuse, explicitly say that it depends on each repository's license and the applicable hackathon rules. Never change counts or claim code was reviewed.'''

async def synthesize(payload: dict) -> SynthesizerOutput:
    known = {candidate['id'] for candidate in payload['candidates']}
    addons_by_id = {candidate['id']: {feature.strip().casefold() for feature in candidate.get('addons', [])}
                    for candidate in payload['candidates']}
    title_to_id = {candidate['title']: candidate['id'] for candidate in payload['candidates']
                   if candidate.get('title') and sum(other.get('title') == candidate['title']
                                                     for other in payload['candidates']) == 1}
    def validate(result: SynthesizerOutput) -> None:
        for cluster in result.addon_clusters:
            normalized = [item if item in known else title_to_id.get(item) for item in cluster.seen_in_projects]
            if any(item is None for item in normalized):
                raise ValueError('Synthesizer referenced a project not in the reviewed set')
            cluster.seen_in_projects = list(dict.fromkeys(normalized))
            for feature in cluster.features:
                if not any(feature.strip().casefold() in addons_by_id[project_id]
                           for project_id in cluster.seen_in_projects):
                    raise ValueError('Synthesizer addon was absent from the cited projects')
        result.crowdedness_summary = result.crowdedness_summary.replace(
            'no single project offers', 'no reviewed project is evidenced to offer'
        ).replace('No single project offers', 'No reviewed project is evidenced to offer')
        result.crowdedness_summary = result.crowdedness_summary.replace(
            'The market is ', 'The reviewed sample is '
        ).replace('The landscape is ', 'The reviewed sample is ')
        result.crowdedness_summary = re.sub(
            r'\bthe (?:market|landscape)\b', 'the reviewed sample',
            result.crowdedness_summary, flags=re.I,
        )
        result.crowdedness_summary = re.sub(
            r'\bclear gap\b', 'gap in this reviewed sample',
            result.crowdedness_summary, flags=re.I,
        )
        result.crowdedness_summary = re.sub(
            r'\bmany projects\b', 'some reviewed projects',
            result.crowdedness_summary, flags=re.I,
        )
        result.crowdedness_summary = re.sub(
            r'\bno project\b', 'no reviewed project', result.crowdedness_summary, flags=re.I
        )
        result.crowdedness_summary = re.sub(
            r'\bno reviewed project (?:provides|offers|has)\b',
            'no reviewed project is evidenced to provide',
            result.crowdedness_summary, flags=re.I,
        )
        result.crowdedness_summary = re.sub(
            r'\b(is|are) absent\b',
            lambda match: ('was' if match.group(1).lower() == 'is' else 'were')
                          + ' not established in the reviewed evidence',
            result.crowdedness_summary, flags=re.I,
        )
        result.crowdedness_summary = re.sub(
            r'([.!?]\s+)([a-z])', lambda match: match.group(1) + match.group(2).upper(),
            result.crowdedness_summary,
        )
        if not result.crowdedness_summary.casefold().startswith(('among the reviewed', 'in the reviewed', 'the reviewed sample')):
            result.crowdedness_summary = 'Among the reviewed projects, ' + result.crowdedness_summary[0].lower() + result.crowdedness_summary[1:]
        result.crowdedness_summary = result.crowdedness_summary[0].upper() + result.crowdedness_summary[1:]
        if len(payload['candidates']) == 1:
            candidate = payload['candidates'][0]
            shared = ', '.join(candidate.get('shared') or []) or 'some of the requested features'
            gaps = ', '.join(candidate.get('not_mentioned_in_evidence') or [])
            result.crowdedness_summary = (
                f'The one reviewed project has evidence for {shared}. '
                + (f'The supplied evidence does not establish {gaps}. ' if gaps else '')
                + 'One reviewed project is insufficient to judge how crowded this area is.'
            )
    return await structured('synthesizer', SYSTEM, payload, SynthesizerOutput, validate)
