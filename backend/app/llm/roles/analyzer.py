import re

from app.llm.contracts import AnalyzerOutput
from app.llm.router import structured

SYSTEM = '''You are the Ignis Analyzer. Understand a finalized software idea. Return only schema-valid JSON. Identify the problem, target users, only the core capabilities explicitly stated by the user, and a canonical input → processing → output workflow. Before writing features, map every distinct stated action and output to a feature, and check that every feature can be supported by a short verbatim phrase in the user's idea or problem statement. Preserve stated constraints such as safety, privacy, accessibility, and timing in the relevant feature description. Include all separately stated capabilities; for example searchable transcripts require search, not transcription alone. One to eight numbered features is acceptable; never pad a short idea with a dashboard, visualization, export, integration, offline mode, security method, notification, or platform feature unless the user requested it. Make each numbered feature one independently checkable capability. Split distinct stated capabilities instead of combining them with "and" or "&". Make 4–6 distinct GitHub search queries from workflow, use case, and technology angles. Use 2–3 distinctive keywords per query: GitHub search ANDs words, so long phrases often return nothing. Queries must contain search terms only; no stars, dates, fork, or archived qualifiers. If a rough idea lacks details, keep the feature list short and describe assumptions as assumptions in feature descriptions. Optional hackathon track is context, not proof of uniqueness.'''

async def analyze(idea: str, mode: str, problem_statement: str = '', track: str = '', failed_queries: list[str] | None = None) -> AnalyzerOutput:
    source = idea + ' ' + problem_statement
    searchable_transcripts = bool(re.search(
        r'\bsearch(?:able)?\b.{0,35}\btranscripts?\b|\btranscripts?\b.{0,35}\bsearch(?:able)?\b',
        source, re.I))
    safety_required = bool(re.search(r'\b(?:safe|safety)\b', source, re.I))
    system = SYSTEM + (' Broaden queries from failed terms to 1–2 core nouns; keep the same idea and features.'
                       if failed_queries else '')
    if searchable_transcripts:
        system += ' This idea explicitly asks for searchable transcripts. Make transcript creation and transcript search two separate numbered features; a searchable transcription feature merges two independently checkable capabilities.'
    if not re.search(r'\bconcepts?\b', source, re.I):
        system += ' Do not add key-concept extraction to a flashcard feature unless the user actually specified concept extraction.'
    if not re.search(r'\b(?:timestamp|jump|seek)\w*\b', source, re.I):
        system += ' Searching text does not imply timestamp links, video seeking, or jumping to video segments; do not add those actions unless stated.'
    if safety_required:
        system += ' The user explicitly mentions safety. Preserve that constraint in the relevant feature description.'

    def validate(result: AnalyzerOutput) -> None:
        if searchable_transcripts:
            has_search_feature = any(
                re.search(r'\b(?:search(?:able)?|lookup|query|index)\b', f.name + ' ' + f.description, re.I)
                and re.search(r'\btranscripts?\b', f.name + ' ' + f.description, re.I)
                and not re.search(r'\btranscrib\w*\b', f.name, re.I)
                for f in result.features)
            if not has_search_feature:
                raise ValueError('Searchable transcripts require a separate transcript-search feature')
        if not re.search(r'\bconcepts?\b', source, re.I) and any(
            re.search(r'\b(?:extract|identify|detect)\b.{0,20}\b(?:key )?concepts?\b',
                      f.name + ' ' + f.description, re.I) for f in result.features):
            raise ValueError('Analyzer added concept extraction absent from user input')
        if not re.search(r'\b(?:timestamp|jump|seek)\w*\b', source, re.I) and any(
            re.search(r'\b(?:timestamp|jump|seek)\w*\b', f.name + ' ' + f.description, re.I)
            for f in result.features):
            raise ValueError('Analyzer added video navigation absent from user input')
        if safety_required and not any(re.search(r'\b(?:safe|safety)\b',
                                               f.name + ' ' + f.description, re.I)
                                       for f in result.features):
            raise ValueError('Analyzer omitted an explicit safety constraint')

    return await structured('analyzer', system,
                            {'idea': idea, 'mode': mode, 'problem_statement': problem_statement,
                             'track': track, 'failed_queries': failed_queries or []},
                            AnalyzerOutput, validate)
