import csv
import hashlib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "MANIFEST.csv"
TEXT_EXTENSIONS = {".cff", ".csv", ".html", ".ipynb", ".json", ".jsonl", ".md", ".py", ".tex", ".txt", ".yml", ".yaml"}


def artifact_sha256(path):
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_EXTENSIONS:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


class ManifestHashTests(unittest.TestCase):
    def load_records(self):
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_committed_artifact_hashes_match_manifest(self):
        records = self.load_records()

        self.assertTrue(records, "manifest must contain artifact records")
        for record in records:
            expected = record["sha256"].strip()
            status = record["status"].strip()
            if not expected or status.startswith("retired_") or status.startswith("not_in_repository"):
                continue

            artifact = REPO_ROOT / record["file"]
            self.assertTrue(artifact.is_file(), f"missing manifest artifact: {record['file']}")
            actual = artifact_sha256(artifact)
            self.assertEqual(actual, expected, f"hash mismatch: {record['file']}")

    def test_current_data_artifacts_are_manifested(self):
        records = self.load_records()
        recorded = {record["file"] for record in records}
        current_data = REPO_ROOT / "data" / "current"
        artifacts = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in current_data.rglob("*")
            if path.is_file() and path.name != "README.md"
        }
        self.assertEqual(artifacts - recorded, set(), "unmanifested current data artifacts")

    def test_roundness_source_input_counts_and_class_summary(self):
        base = REPO_ROOT / "data" / "current" / "roundness_exploratory"
        with (base / "confirmatory_target_level_contrasts.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 552)
        with (base / "confirmatory_prompt_level_sequence_scores.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 3864)

        with (base / "exploratory_class_arm_decomposition.csv").open(encoding="utf-8", newline="") as handle:
            summary = {row["class_short"]: int(row["n_targets"]) for row in csv.DictReader(handle)}
        self.assertEqual(summary, {"nonround": 276, "round_target_operand_ends5": 138, "round_target_operand_ends0": 138})


if __name__ == "__main__":
    unittest.main()
