import uuid

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Agent, Passenger
from app.security import AGENT, PASSENGER, decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)

_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_passenger(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Passenger:
    if credentials is None:
        raise _CREDENTIALS_ERROR
    try:
        token = decode_access_token(credentials.credentials)
    except jwt.InvalidTokenError:
        raise _CREDENTIALS_ERROR from None
    if token.get("role") != PASSENGER:
        raise _CREDENTIALS_ERROR
    try:
        passenger_id = uuid.UUID(token["sub"])
    except (KeyError, ValueError):
        raise _CREDENTIALS_ERROR from None
    result = await db.execute(select(Passenger).where(Passenger.id == passenger_id))
    passenger = result.scalar_one_or_none()
    if passenger is None:
        raise _CREDENTIALS_ERROR
    return passenger


async def get_current_agent(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Agent:
    if credentials is None:
        raise _CREDENTIALS_ERROR
    try:
        token = decode_access_token(credentials.credentials)
    except jwt.InvalidTokenError:
        raise _CREDENTIALS_ERROR from None
    if token.get("role") != AGENT:
        raise _CREDENTIALS_ERROR
    try:
        agent_id = uuid.UUID(token["sub"])
    except (KeyError, ValueError):
        raise _CREDENTIALS_ERROR from None
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if agent is None:
        raise _CREDENTIALS_ERROR
    return agent