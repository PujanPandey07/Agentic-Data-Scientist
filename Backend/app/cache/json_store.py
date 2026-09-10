import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonCacheStore:
    """Small persistent JSON store used behind typed cache interfaces."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for(key)
        if not path.is_file():
            logger.info("Cache miss: %s", key)
            return None

        try:
            with path.open("r", encoding="utf-8") as file:
                value = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Unable to read cache entry %s: %s", key, error)
            return None

        logger.info("Cache hit: %s", key)
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path_for(key)
        temporary_path = path.with_suffix(".tmp")

        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(value, file, indent=2, ensure_ascii=True)
            temporary_path.replace(path)
        except OSError as error:
            logger.exception("Unable to write cache entry %s: %s", key, error)

    def invalidate(self, key: str) -> None:
        path = self._path_for(key)
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            logger.warning(
                "Unable to invalidate cache entry %s: %s", key, error)

    def _path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"
