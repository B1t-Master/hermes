import unittest
from unittest import mock

from app.agents import graph
from app.agents.turns import strip_source_section, strip_source_urls


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
        self.assertIn("…", context)
        self.assertLess(len(context), 200)


class StripSourceUrlsTest(unittest.TestCase):
    def test_citation_urls_are_removed_but_the_label_stays(self):
        text = "Cite [KQ conditions](https://example.com/doc.pdf) please."
        cleaned = strip_source_urls(text)
        self.assertNotIn("example.com", cleaned)
        self.assertIn("KQ conditions", cleaned)

    def test_official_self_service_link_is_kept(self):
        url = "https://digital.kenya-airways.com/servicing/self-reaccommodation/retrieve"
        cleaned = strip_source_urls(f"You can re-book yourself at {url}")
        self.assertIn(url, cleaned)

    def test_official_markdown_link_is_kept_intact(self):
        url = "https://www.kenya-airways.com/en-ke/help/"
        cleaned = strip_source_urls(f"See [the help centre]({url}) please.")
        self.assertIn(f"[the help centre]({url})", cleaned)

    def test_no_dangling_link_bracket(self):
        cleaned = strip_source_urls("See [the doc](https://example.com/x).")
        self.assertNotIn("(", cleaned)

    def test_any_kenya_airways_page_is_kept(self):
        url = "https://www.kenya-airways.com/en-ke/book-manage/manage-booking/online-refunds/"
        cleaned = strip_source_urls(f"Go to {url} and fill in the form.")
        self.assertIn(url, cleaned)
        self.assertIn("and fill in the form.", cleaned)

    def test_partner_tracking_link_is_kept(self):
        url = "https://www.worldtracer.aero"
        cleaned = strip_source_urls(f"Track your bag at {url}")
        self.assertIn(url, cleaned)

    def test_removed_link_does_not_leave_a_dangling_connector(self):
        cleaned = strip_source_urls("Go to https://example.com/docs and submit the form.")
        self.assertNotIn("go to", cleaned.lower())
        self.assertNotIn("example.com", cleaned)
        self.assertIn("submit the form.", cleaned)


class StripSourceSectionTest(unittest.TestCase):
    def test_trailing_sources_block_is_dropped(self):
        cleaned = strip_source_section("Your answer.\n\nSources: [a](https://x.com)")
        self.assertEqual(cleaned, "Your answer.")


if __name__ == "__main__":
    unittest.main()
