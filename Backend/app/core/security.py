# core/security.py
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, Request
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


IS_PRODUCTION = os.environ.get("ENVIRONMENT", "development") == "production"
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
if not SECRET_KEY:
    if IS_PRODUCTION:
        raise RuntimeError("JWT_SECRET_KEY environment variable is required in production.")
    SECRET_KEY = "insecure_dev_secret_key_change_me_in_production"
ALGORITHM = "HS256"

# Short-lived — kept in the client's memory, never persisted, so a short
# expiry limits damage if it's ever exposed (e.g. logged accidentally).
ACCESS_TOKEN_EXPIRE_MINUTES = 15
# Long-lived — safe in an httpOnly cookie since JS can never read it.
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def _create_token(user_id: int, expires_delta: timedelta, token_type: str) -> str:
    # "sub" (subject) and "exp" (expiry) are standard JWT claim names —
    # "type" is our own addition, so a refresh token can never be
    # accidentally accepted where an access token is expected, or vice versa.
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + expires_delta,
        "type": token_type,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: int) -> str:
    return _create_token(user_id, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES), "access")


def create_refresh_token(user_id: int) -> str:
    return _create_token(user_id, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS), "refresh")


def decode_token(token: str, expected_type: str) -> int:
    """Returns the user_id if the token is valid and of the expected
    type; raises HTTPException(401) otherwise."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if payload.get("type") != expected_type:
        raise HTTPException(status_code=401, detail="Wrong token type")

    return int(payload["sub"])


# One shared instance — this is what makes Swagger show the Authorize
# button and know to send whatever token you paste in as a Bearer header.
security = HTTPBearer()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> int:
    """FastAPI dependency — protects a route by requiring a valid access
    token in the Authorization header. Usage on a route:
        async def my_route(user_id: int = Depends(get_current_user_id)):
    """
    # HTTPBearer already validated the header looks like "Bearer <token>"
    # and raises its own 401 if it's missing entirely — credentials.credentials
    # is just the raw token string, no manual prefix-stripping needed.
    token = credentials.credentials
    return decode_token(token, expected_type="access")
