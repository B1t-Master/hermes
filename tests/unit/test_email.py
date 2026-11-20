import tempfile
import unittest
from pathlib import Path

from services import email


class SendEmailTest(unittest.TestCase):
    def test_writes_outbox_file_and_returns_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = email.send_email("a@b.com", "Hello world", "Body text", outbox_dir=tmp)
            self.assertEqual(record["to"], "a@b.com")
            self.assertEqual(record["subject"], "Hello world")
            self.assertIn("sent_at", record)
            path = Path(record["path"])
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("To: a@b.com", text)
            self.assertIn("Body text", text)

    def test_unwritable_outbox_does_not_raise(self):
        # A path that is a file, not a directory, must not break the caller.
        with tempfile.NamedTemporaryFile() as handle:
            record = email.send_email("a@b.com", "s", "b", outbox_dir=handle.name)
        self.assertEqual(record["to"], "a@b.com")
        self.assertNotIn("path", record)


if __name__ == "__main__":
    unittest.main()
