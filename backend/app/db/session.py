"""Reusable async Postgres connections for API requests and pipeline stages."""
import asyncio
from contextlib import asynccontextmanager

from psycopg.rows import dict_row
from psycopg.conninfo import conninfo_to_dict
from psycopg import ProgrammingError
from psycopg_pool import AsyncConnectionPool

from app.config import settings

_pool: AsyncConnectionPool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        if not settings.database_url:
            raise RuntimeError('DATABASE_URL is required')
        try:
            conninfo_to_dict(settings.database_url)
        except ProgrammingError:
            # The driver's parse error can echo a URL containing its password.
            raise RuntimeError('DATABASE_URL has invalid connection string syntax') from None
        async with _pool_lock:
            if _pool is None:
                pool = AsyncConnectionPool(
                    conninfo=settings.database_url,
                    kwargs={'row_factory': dict_row, 'connect_timeout': 8},
                    # Hosted Postgres may close idle SSL sockets behind the pool.
                    # Verify a borrowed connection before charging quota or querying.
                    check=AsyncConnectionPool.check_connection,
                    min_size=1,
                    max_size=4,
                    timeout=15,
                    max_idle=300,
                    open=False,
                )
                try:
                    await pool.open(wait=True, timeout=15)
                except Exception:
                    await pool.close()
                    raise
                _pool = pool
    return _pool


@asynccontextmanager
async def database():
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn


async def close_pool() -> None:
    global _pool
    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None
