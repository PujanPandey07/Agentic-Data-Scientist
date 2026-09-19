from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import redis


# Where Parquet files actually live on disk — Redis only ever stores a
# short file-path string, never the DataFrame's actual bytes. Keeping the
# real data on disk (not in Redis, not in process memory) is what lets a
# separate background worker process access the same cached DataFrame a
# web request cached earlier — the whole point of this change.
CACHE_DIR = Path(__file__).resolve().parent.parent / "runtime_cache"
CACHE_DIR.mkdir(exist_ok=True)


class RuntimeStateStore:
    """Shared runtime cache for large graph artifacts (DataFrames).

    Previously an in-process dict — fine for one process, invisible to a
    separate worker process. Now: the actual DataFrame is saved to a
    Parquet file on disk, and Redis holds only the file path, keyed by
    dataset_id. Any process (the API, or a future background worker) can
    look up the same path in Redis and read the same file from disk —
    that's what makes this genuinely shared, not just faster.
    """

    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6380/0")
        # decode_responses=True — without it, Redis hands back raw bytes
        # instead of Python strings, which is awkward for what are just
        # short file-path strings here.
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def _dataset_path(self, dataset_id: str) -> Path:
        return CACHE_DIR / f"{dataset_id}.parquet"

    def _train_path(self, dataset_id: str) -> Path:
        return CACHE_DIR / f"{dataset_id}_train.parquet"

    def _test_path(self, dataset_id: str) -> Path:
        return CACHE_DIR / f"{dataset_id}_test.parquet"

    def get_dataset(self, dataset_id: str | None) -> Any | None:
        if not dataset_id:
            return None
        path = self._redis.get(f"dataset:{dataset_id}")
        if not path or not os.path.exists(path):
            return None
        return pd.read_parquet(path)

    def set_dataset(self, dataset_id: str | None, dataframe: Any) -> None:
        if not dataset_id:
            return
        path = self._dataset_path(dataset_id)
        dataframe.to_parquet(path)
        self._redis.set(f"dataset:{dataset_id}", str(path))

    def clear_dataset(self, dataset_id: str | None) -> None:
        if not dataset_id:
            return
        path = self._dataset_path(dataset_id)
        if path.exists():
            path.unlink()
        self._redis.delete(f"dataset:{dataset_id}")

    def get_train_test(self, dataset_id: str | None) -> tuple[Any | None, Any | None]:
        if not dataset_id:
            return None, None
        train_path = self._redis.get(f"train:{dataset_id}")
        test_path = self._redis.get(f"test:{dataset_id}")
        if not train_path or not test_path:
            return None, None
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            return None, None
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    def set_train_test(self, dataset_id: str | None, train_df: Any, test_df: Any) -> None:
        if not dataset_id:
            return
        train_path = self._train_path(dataset_id)
        test_path = self._test_path(dataset_id)
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)
        self._redis.set(f"train:{dataset_id}", str(train_path))
        self._redis.set(f"test:{dataset_id}", str(test_path))

    def clear_train_test(self, dataset_id: str | None) -> None:
        if not dataset_id:
            return
        for path_key, path_fn in (("train", self._train_path), ("test", self._test_path)):
            path = path_fn(dataset_id)
            if path.exists():
                path.unlink()
            self._redis.delete(f"{path_key}:{dataset_id}")

    def clear_dataset_related(self, dataset_id: str | None) -> None:
        self.clear_dataset(dataset_id)
        self.clear_train_test(dataset_id)


runtime_state_store = RuntimeStateStore()
