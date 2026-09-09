from typing import AsyncIterator, Annotated, Iterator

from fastapi import Depends
from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker

from core.config import settings

engine:AsyncEngine = create_async_engine(settings.date_source_url,echo=True,pool_pre_ping=True)

# 异步
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




engine:Engine = create_engine(settings.date_source_url,echo=True,pool_pre_ping=True)

SessionLocal:sessionmaker[Session] = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False
)
