# utilis/sanitize.py
"""Recursively convert numpy scalar types into native Python types.

Pandas/numpy operations (.describe(), .value_counts(), correlation
matrices, etc.) commonly return numpy.int64/float64/bool_ instead of
plain Python int/float/bool. Python's json module tolerates this via a
fallback (see ReportingService's json.dump(..., default=str)), but
Pydantic's model_dump(mode="json") has no such fallback and raises
PydanticSerializationError the moment it hits one. Call this on any
dict built from pandas/numpy results before handing it to a Pydantic
model or anywhere else that needs guaranteed-native types.
"""
import numpy as np


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    return value
