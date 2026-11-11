from urllib.parse import urlparse

def norm_repo(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    parts = parsed.path.strip('/').split('/')
    if parsed.netloc.lower() != 'github.com' or len(parts) < 2:
        return None
    return f'https://github.com/{parts[0].lower()}/{parts[1].removesuffix(".git").lower()}'
