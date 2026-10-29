from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / '.env')

@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv('DATABASE_URL', '').strip()
    github_token: str = os.getenv('GITHUB_TOKEN', '').strip()
    groq_key: str = os.getenv('GROQ_API_KEY', '').strip()
    gemini_key: str = os.getenv('GEMINI_API_KEY', os.getenv('GOOGLE_API_KEY', '')).strip()
    auth_secret: str = os.getenv('AUTH_SECRET', '')
    reviewer_model: str = os.getenv('REVIEWER_MODEL', 'gemini-2.5-flash')
    reviewer_fallback_model: str = os.getenv('REVIEWER_FALLBACK_MODEL', 'qwen/qwen3.8-27b')
    reasoning_model: str = os.getenv('REASONING_MODEL', 'openai/gpt-oss-120b')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
    similarity_cutoff: float = float(os.getenv('SIMILARITY_CUTOFF', '0.48'))
    analysis_daily_limit: int = int(os.getenv('ANALYSIS_DAILY_LIMIT', '10'))
    explain_daily_limit: int = int(os.getenv('EXPLAIN_DAILY_LIMIT', '20'))

settings = Settings()
