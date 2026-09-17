from __future__ import annotations

from threading import RLock
from typing import Any


class RuntimeStateStore:
    """In-memory runtime cache for large graph artifacts.

    These values are intentionally kept out of the LangGraph checkpoint state so
    the persisted state stays lightweight. The cache is keyed by dataset_id so the
    same dataset can be reused across a run without copying the full dataframe
    into every checkpoint payload.
    """

    def __init__(self):
        self._lock = RLock()
        self._datasets: dict[str, Any] = {}
        self._train_tests: dict[str, tuple[Any, Any]] = {}

    def get_dataset(self, dataset_id: str | None) -> Any | None:
        if not dataset_id:
            return None
        with self._lock:
            return self._datasets.get(dataset_id)

    def set_dataset(self, dataset_id: str | None, dataframe: Any) -> None:
        if not dataset_id:
            return
        with self._lock:
            self._datasets[dataset_id] = dataframe

    def clear_dataset(self, dataset_id: str | None) -> None:
        if not dataset_id:
            return
        with self._lock:
            self._datasets.pop(dataset_id, None)

    def get_train_test(self, dataset_id: str | None) -> tuple[Any | None, Any | None]:
        if not dataset_id:
            return None, None
        with self._lock:
            train_df, test_df = self._train_tests.get(dataset_id, (None, None))
            return train_df, test_df

    def set_train_test(self, dataset_id: str | None, train_df: Any, test_df: Any) -> None:
        if not dataset_id:
            return
        with self._lock:
            self._train_tests[dataset_id] = (train_df, test_df)

    def clear_train_test(self, dataset_id: str | None) -> None:
        if not dataset_id:
            return
        with self._lock:
            self._train_tests.pop(dataset_id, None)

    def clear_dataset_related(self, dataset_id: str | None) -> None:
        self.clear_dataset(dataset_id)
        self.clear_train_test(dataset_id)


runtime_state_store = RuntimeStateStore()
