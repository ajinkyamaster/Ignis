import re
BADGE = re.compile(r'\[!\[.*?\]\(.*?\)\]\(.*?\)')
HTML = re.compile(r'<[^>]+>')
NOISE_HEADING = re.compile(r"^(installation|getting started|quick start|prerequisites|environment variables|contributing|contributors|team|acknowledg(?:e)?ments|license|setup|(?:backend|frontend|local|development) setup|roadmap|future work|future plans|planned features|what['’]s next)\b", re.I)
HEADING = re.compile(r'^(#{1,6})\s+')

def strip_readme(raw: str, max_chars: int = 2500) -> str:
    lines, skip_level = [], None
    for line in raw.splitlines():
        stripped = line.strip()
        heading = HEADING.match(stripped)
        if skip_level is not None and (not heading or len(heading.group(1)) > skip_level):
            continue
        skip_level = None
        heading_text = re.sub(r'^[^A-Za-z0-9]+', '', stripped[len(heading.group(0)):]) if heading else ''
        if heading and NOISE_HEADING.match(heading_text):
            skip_level = len(heading.group(1)) if heading else 6
            continue
        if BADGE.search(line) or line.lstrip().startswith(('npm install', 'docker run', 'pip install')):
            continue
        line = HTML.sub(' ', line)
        if line.strip():
            lines.append(line.strip())
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines))[:max_chars]
