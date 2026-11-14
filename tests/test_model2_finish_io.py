import csv
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile

from model2_io import (CONDITIONS, FILES, audit_data, checksums, import_runs,
                       prepare_data, read_json, report, safe_extract, write_json)


def fixture(root):
    root = Path(root)
    clean = root / "clean"
    clean.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        (clean / FILES[split]).write_bytes(pickle.dumps(([[1], [1, 2]], [2, 3])))
    (clean / FILES["item_mapping"]).write_bytes(pickle.dumps({"a": 1, "b": 2, "c": 3}))
    write_json(clean / FILES["summary"], {"dataset": "toy", "num_items": 3, "protocol_version": "toy_clean_v1"})
    hashes = checksums(clean)
    write_json(root / "pipeline_manifest.json", {"clean": {"checksums": hashes}})
    for name in CONDITIONS:
        d = root / "noisy" / name
        d.mkdir(parents=True)
        for filename in FILES.values():
            (d / filename).write_bytes((clean / filename).read_bytes())
        write_json(d / FILES["summary"], {"noise": {"source_clean_checksums": hashes,
                   "corruption_type": name.split("_")[0]}})
    config = root.parent / "config.toml"
    config.write_text('[data]\nexpected_protocol_version="toy_clean_v1"\nartifact_repository="toy/data"\nartifact_revision="abc"\n[data.expected_checksums]\n' +
                      ''.join(f'{key}="{value}"\n' for key, value in hashes.items()) +
                      '\n[model]\ngnn_steps=1\nhybrid=true\n[training]\nlearning_rate=0.001\nweight_decay=0.00001\nlr_decay_step=3\nlr_decay_gamma=0.1\n')
    return config


class FinishIOTests(unittest.TestCase):
    def test_wrong_folder_summary_repair_retains_exact_manifest_hash(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            config = fixture(root / "source/pipeline")
            p = root / "source/pipeline"
            (p / "clean/preprocessing_summary.json").rename(p / "preprocessing_summary.json")
            z = root / "data.zip"
            with zipfile.ZipFile(z, "w") as f:
                for x in p.rglob("*"):
                    if x.is_file(): f.write(x, x.relative_to(p.parent))
            dst = prepare_data(z, root / "staged")
            self.assertEqual(audit_data(dst, config)["examples"]["test"], 2)
            self.assertFalse((p / "clean/preprocessing_summary.json").exists())

    def test_wrong_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            fixture(root / "source/pipeline")
            p = root / "source/pipeline"
            (p / "clean/preprocessing_summary.json").unlink()
            write_json(p / "preprocessing_summary.json", {"wrong": True})
            z = root / "data.zip"
            with zipfile.ZipFile(z, "w") as f:
                for x in p.rglob("*"):
                    if x.is_file(): f.write(x, x.relative_to(p.parent))
            with self.assertRaises(ValueError): prepare_data(z, root / "staged")

    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            z = Path(t) / "bad.zip"
            with zipfile.ZipFile(z, "w") as f: f.writestr("../escaped.txt", "bad")
            with self.assertRaises(ValueError): safe_extract(z, Path(t) / "out")

    def test_changed_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "pipeline"
            config = fixture(root)
            (root / "noisy/deletion_0.05/test.txt").write_bytes(pickle.dumps(([[1], [1, 2]], [3, 3])))
            with self.assertRaises(ValueError): audit_data(root, config)

    def test_report_does_not_fabricate_missing_metrics(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            write_json(root / "experiments/toy/results.json", {"dataset": "toy", "seed": 2028})
            write_json(root / "experiments/toy/robustness_results.json", {"clean": {"recall@20": .5, "examples": 2}})
            output = report(root / "experiments", root / "report.csv")
            with output.open(newline="") as f: rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["recall@20"], "0.5")
            self.assertEqual(rows[0]["ndcg@20"], "")


if __name__ == "__main__": unittest.main()
