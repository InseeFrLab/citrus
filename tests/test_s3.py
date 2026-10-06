import tempfile
import unittest
from pathlib import Path

from src.s3 import S3Sync, S3SyncError, download_files, split_s3_uri


class FakeS3:
    """Client boto3 minimal : garde les objets en mémoire, peut refuser les envois."""

    def __init__(self, fail=False):
        self.objects = {}
        self.uploads = []
        self.fail = fail

    def upload_file(self, filename, bucket, key):
        if self.fail:
            raise RuntimeError("ExpiredToken")
        self.objects[(bucket, key)] = Path(filename).read_bytes()
        self.uploads.append(key)

    def download_file(self, bucket, key, filename):
        if (bucket, key) not in self.objects:
            raise RuntimeError("NoSuchKey")
        Path(filename).write_bytes(self.objects[(bucket, key)])


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SplitUriTest(unittest.TestCase):
    def test_bucket_and_key(self):
        self.assertEqual(split_s3_uri("s3://b/a/c/"), ("b", "a/c"))
        self.assertEqual(split_s3_uri("s3://b"), ("b", ""))

    def test_rejects_non_s3(self):
        with self.assertRaises(ValueError):
            split_s3_uri("/tmp/x")
        with self.assertRaises(ValueError):
            split_s3_uri("s3:///x")


class S3SyncTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in ("results.jsonl", "meta.json", "operations.parquet"):
            (self.dir / name).write_text(name, encoding="utf-8")
        self.client = FakeS3()
        self.clock = FakeClock()

    def tearDown(self):
        self.tmp.cleanup()

    def sync(self, **kwargs):
        return S3Sync(
            "s3://projet/evaluation/batch",
            self.dir,
            live_files=("results.jsonl", "meta.json"),
            client=self.client,
            clock=self.clock,
            **kwargs,
        )

    def test_uri_is_normalized(self):
        self.assertEqual(self.sync().uri, "s3://projet/evaluation/batch/")

    def test_pushes_every_n_records(self):
        sync = self.sync(every=3, interval_s=1e9)
        for _ in range(2):
            sync.maybe_push()
        self.assertEqual(self.client.uploads, [])
        sync.maybe_push()
        self.assertEqual(
            self.client.uploads, ["evaluation/batch/results.jsonl", "evaluation/batch/meta.json"]
        )

    def test_pushes_after_interval(self):
        sync = self.sync(every=1000, interval_s=60)
        sync.maybe_push()
        self.assertEqual(self.client.uploads, [])
        self.clock.now = 61
        sync.maybe_push()
        self.assertEqual(len(self.client.uploads), 2)

    def test_push_all_sends_every_file(self):
        self.assertTrue(self.sync().push_all())
        self.assertEqual(
            sorted(key for _, key in self.client.objects),
            [
                "evaluation/batch/meta.json",
                "evaluation/batch/operations.parquet",
                "evaluation/batch/results.jsonl",
            ],
        )

    def test_failed_push_is_counted_not_raised(self):
        self.client.fail = True
        sync = self.sync(every=1)
        with self.assertLogs("citrus", level="WARNING") as logs:
            sync.maybe_push()
            sync.maybe_push()
            self.assertFalse(sync.push_all())
        self.assertEqual(sync.failures, 3)
        self.assertIn("ExpiredToken", logs.output[0])

    def test_check_raises_when_refused(self):
        self.client.fail = True
        with self.assertRaises(S3SyncError):
            self.sync().check("meta.json")

    def test_download_round_trip(self):
        self.sync().push_all()
        target = self.dir / "copie"
        download_files("s3://projet/evaluation/batch/", target, ["meta.json"], client=self.client)
        self.assertEqual((target / "meta.json").read_text(encoding="utf-8"), "meta.json")
        with self.assertRaises(S3SyncError):
            download_files("s3://projet/autre", target, ["meta.json"], client=self.client)


if __name__ == "__main__":
    unittest.main()
