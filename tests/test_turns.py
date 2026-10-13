import unittest

from app.agents.turns import citations_from_chunks, strip_source_urls


class StripSourceUrlsTest(unittest.TestCase):
    def test_strips_bare_https_url(self):
        text = "See https://www.kenya-airways.com/baggage for details."
        self.assertEqual(strip_source_urls(text), "See for details.")

    def test_strips_www_url(self):
        self.assertEqual(strip_source_urls("More at www.example.com now"), "More at now")

    def test_keeps_markdown_link_label(self):
        text = "Read [Baggage rules](https://example.com/baggage) carefully."
        self.assertEqual(strip_source_urls(text), "Read [Baggage rules] carefully.")

    def test_leaves_plain_text_untouched(self):
        text = "Call +254 20 327 4000 or email us."
        self.assertEqual(strip_source_urls(text), text)

    def test_empty_text(self):
        self.assertEqual(strip_source_urls(""), "")


class CitationsTest(unittest.TestCase):
    def test_dedupes_and_rounds_score(self):
        chunks = [
            {"title": "Baggage", "source_url": "url::x", "score": 0.123456},
            {"title": "Baggage", "source_url": "url::x", "score": 0.9},
            {"title": "FAQ", "source_url": None, "score": None},
        ]
        citations = citations_from_chunks(chunks)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["score"], 0.1235)
        self.assertNotIn("score", citations[1])


if __name__ == "__main__":
    unittest.main()
