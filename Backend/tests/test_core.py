import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add Backend/app to python path so imports resolve
app_dir = Path(__file__).resolve().parent.parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import pandas as pd
from pydantic import ValidationError

from core.crypto import encrypt_api_key, decrypt_api_key
from schema.analysis_plan import AnalysisPlan
from services.dataset_service import read_file_to_dataframe


class TestCoreServices(unittest.TestCase):

    def test_crypto_roundtrip(self):
        secret = "sk-test-fake-api-key-12345"
        encrypted = encrypt_api_key(secret)
        self.assertNotEqual(secret, encrypted)
        decrypted = decrypt_api_key(encrypted)
        self.assertEqual(secret, decrypted)

    def test_analysis_plan_schema_valid_types(self):
        for ptype in ["classification", "regression", "clustering", "time_series", None]:
            plan = AnalysisPlan(
                user_intent="Test intent",
                problem_type=ptype,
                target_column="target",
                tasks=["cleaning", "eda"],
                reasoning="Test reasoning",
            )
            self.assertEqual(plan.problem_type, ptype)

    def test_analysis_plan_schema_invalid_type(self):
        with self.assertRaises(ValidationError):
            AnalysisPlan(
                user_intent="Test intent",
                problem_type="unsupported_type_name",
                target_column="target",
                tasks=["cleaning"],
                reasoning="Invalid",
            )

    def test_multi_format_dataset_reading(self):
        df = pd.DataFrame({"feature_1": [10, 20, 30], "feature_2": ["a", "b", "c"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            formats = [
                ("test.csv", lambda p: df.to_csv(p, index=False)),
                ("test.tsv", lambda p: df.to_csv(p, sep="\t", index=False)),
                ("test.parquet", lambda p: df.to_parquet(p, index=False)),
                ("test.json", lambda p: df.to_json(p, orient="records")),
                ("test.feather", lambda p: df.to_feather(p)),
                ("test.xlsx", lambda p: df.to_excel(p, index=False)),
            ]

            for fname, saver in formats:
                file_path = tmp_path / fname
                saver(file_path)
                loaded_df = read_file_to_dataframe(file_path)
                self.assertEqual(len(loaded_df), 3, f"Row count failed for {fname}")
                self.assertEqual(len(loaded_df.columns), 2, f"Column count failed for {fname}")

    def test_empty_file_guardrail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "empty.csv"
            empty_file.write_text("")
            with self.assertRaises(ValueError):
                read_file_to_dataframe(empty_file)


if __name__ == "__main__":
    unittest.main()
