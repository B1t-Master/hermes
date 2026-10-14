import unittest
from unittest import mock

from app.agents import graph


class FormatContextTest(unittest.TestCase):
    def test_no_cap_keeps_full_content(self):
        chunks = [{"title": "T", "content": "x" * 5000}]
        with mock.patch.object(graph.settings, "context_max_chars", 0):
            context, titles = graph._format_context(chunks)
        self.assertEqual(titles, "T")
        self.assertEqual(len(context), len("[1] T\n") + 5000)

    def test_cap_truncates_long_chunks(self):
        chunks = [{"title": "T", "content": "x" * 5000}]
        with mock.patch.object(graph.settings, "context_max_chars", 100):
            context, _ = graph._format_context(chunks)
        self.assertIn("\u2026", context)
        self.assertLess(len(context), 200)


if __name__ == "__main__":
    unittest.main()
