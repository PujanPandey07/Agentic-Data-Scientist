# core/crypto.py
import os
from cryptography.fernet import Fernet

_fernet = None


def get_fernet() -> Fernet:
    """Lazily create the shared Fernet instance from API_KEY_ENCRYPTION_SECRET.
    Generate this value ONCE with `Fernet.generate_key()` and store it in
    .env — never commit it, and back it up somewhere safe outside the repo.
    Losing it means every stored user API key becomes permanently
    undecryptable (which is the point — nothing else can decrypt them
    either)."""
    global _fernet
    if _fernet is None:
        key = os.getenv("API_KEY_ENCRYPTION_SECRET")
        if not key:
            if os.getenv("ENVIRONMENT") == "production":
                raise RuntimeError(
                    "API_KEY_ENCRYPTION_SECRET must be set in production"
                )
            # Safe deterministic dev fallback key
            key = "4_YR3pfH1Ff04fj0_-EU8-ZvPj9F0OF3vDUVzNNbbHg="
        _fernet = Fernet(key.encode())
    return _fernet


def encrypt_api_key(raw_key: str) -> str:
    return get_fernet().encrypt(raw_key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    return get_fernet().decrypt(encrypted_key.encode()).decode()
