import unittest

from evals.scoring import score_case


class ScoreCaseTest(unittest.TestCase):
    def test_passes_when_all_checks_match(self):
        case = {
            "id": "x",
            "question": "q",
            "expect_contains": ["free"],
            "expect_escalate": False,
        }
        result = score_case(case, answer="The service is free.", escalated=False, sources=[])
        self.assertTrue(result.passed)
        self.assertEqual(result.failed_checks, [])

    def test_fails_on_missing_term(self):
        case = {"id": "x", "question": "q", "expect_contains": ["free"]}
        result = score_case(case, answer="It costs money.", escalated=False, sources=[])
        self.assertFalse(result.passed)
        self.assertIn("contains:free", result.failed_checks)

    def test_any_of_needs_one(self):
        case = {"id": "x", "question": "q", "expect_any": ["booking reference", "e-ticket"]}
        ok = score_case(case, answer="Use your e-ticket number.", escalated=False, sources=[])
        bad = score_case(case, answer="Use the app.", escalated=False, sources=[])
        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)

    def test_absent_term_fails(self):
        case = {"id": "x", "question": "q", "expect_none": ["paris"]}
        result = score_case(case, answer="Paris is the capital.", escalated=False, sources=[])
        self.assertFalse(result.passed)
        self.assertIn("absent:paris", result.failed_checks)

    def test_escalate_expectation(self):
        case = {"id": "x", "question": "q", "expect_escalate": True}
        self.assertTrue(score_case(case, answer="", escalated=True, sources=[]).passed)
        self.assertFalse(score_case(case, answer="", escalated=False, sources=[]).passed)

    def test_cite_expectation(self):
        case = {"id": "x", "question": "q", "expect_cite": True}
        cited = score_case(case, answer="a", escalated=False, sources=[{"title": "FAQ"}])
        uncited = score_case(case, answer="a", escalated=False, sources=[])
        self.assertTrue(cited.passed)
        self.assertFalse(uncited.passed)

    def test_error_fails_case(self):
        case = {"id": "x", "question": "q"}
        result = score_case(case, answer="", escalated=False, sources=[], error="boom")
        self.assertFalse(result.passed)
        self.assertEqual(result.error, "boom")

    def test_matches_across_non_breaking_spaces(self):
        case = {"id": "x", "question": "q", "expect_contains": ["7 days"]}
        result = score_case(case, answer="up to 7\u00a0days before", escalated=False, sources=[])
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
