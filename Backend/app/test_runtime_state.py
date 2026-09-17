import pandas as pd

from services.runtime_state import runtime_state_store


def test_runtime_state_keeps_large_frames_out_of_state():
    state = {"dataset_id": "runtime-test"}
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    runtime_state_store.set_dataset(state["dataset_id"], df)
    state["dataframe"] = None

    assert state["dataframe"] is None
    assert runtime_state_store.get_dataset(state["dataset_id"]).equals(df)
