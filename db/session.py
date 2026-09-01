from typing import AsyncIterator, Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine, async_sessionmaker

from core.config import settings

engine:AsyncEngine = create_async_engine(settings.date_source_url,echo=True,pool_pre_ping=True)

AsyncSessionLocal:async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False
)

async def get_session()->AsyncIterator[AsyncSession]:
    #
    async with AsyncSessionLocal() as session:
        yield session


DbSession = Annotated[AsyncSession,Depends(get_session)]