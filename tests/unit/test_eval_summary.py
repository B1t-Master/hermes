import json
import tempfile
import unittest
from pathlib import Path

from evals.summary import amain, build_summary, parse_args


def _functional_case(case_id: str, **overrides) -> dict:
    case = {
        "id": case_id,
        "question": f"question {case_id}",
        "passed": True,
        "escalated": False,
        "cited": True,
        "latency_ms": 1200.0,
        "failed_checks": [],
        "error": None,
    }
    case.update(overrides)
    return case


def _recall_case(case_id: str, **overrides) -> dict:
    case = {
        "id": case_id,
        "question": f"question {case_id}",
        "ground_truth": "adjudicated",
        "total_relevant": 2,
        "retrieved_in_top_k": 2,
        "relevant_ranks": [1, 2],
        "hits": {"1": 1, "2": 2, "3": 2},
        "first_relevant_rank": 1,
        "top_scores": [{"rank": 1, "title": "FAQ", "score": 0.9, "relevant": True}],
    }
    case.update(overrides)
    return case


def _rag_case(case_id: str, **overrides) -> dict:
    case = {
        "id": case_id,
        "question": f"question {case_id}",
        "recall_at_3": 1.0,
        "precision_at_3": 0.667,
        "hit_rate_at_3": 1.0,
        "mrr": 1.0,
        "faithfulness": 0.75,
        "answer_relevance": 0.8,
        "judge_used": True,
    }
    case.update(overrides)
    return case


class BuildSummaryTest(unittest.TestCase):
    def test_merges_all_sources_into_one_case_row(self):
        reports = {
            "functional": {
                "passed": 1,
                "total": 1,
                "mean_latency_ms": 1200.0,
                "wall_clock_ms": 1300.0,
                "cases": [_functional_case("a")],
            },
            "recall": {
                "mode": "vector",
                "corpus_chunks": 65,
                "cases_scored": 1,
                "server_retriever_k": 3,
                "recall_by_k": {"3": 1.0},
                "case_hit_rate_at_server_k": 1.0,
                "mrr_first_relevant": 1.0,
                "cases": [_recall_case("a")],
            },
            "rag_metrics": {
                "k": 3,
                "retrieval_mode": "vector",
                "judge_model_calls": 1,
                "judge_coverage": 1.0,
                "summary": {"faithfulness": 0.75, "answer_relevance": 0.8},
                "cases": [_rag_case("a")],
            },
        }

        summary = build_summary(reports, full=True)

        self.assertEqual(len(summary["cases"]), 1)
        row = summary["cases"][0]
        self.assertEqual(row["id"], "a")
        self.assertEqual(row["functional"]["passed"], True)
        self.assertEqual(row["recall"]["first_relevant_rank"], 1)
        self.assertEqual(row["rag"]["faithfulness"], 0.75)
        self.assertEqual(row["missing"], [])
        # Bulky source-only fields stay out of the summary.
        self.assertNotIn("answer", row["functional"])
        self.assertNotIn("top_scores", row["recall"])

        overall = summary["overall"]
        self.assertEqual(overall["pass_rate"], 1.0)
        self.assertEqual(overall["retrieval"]["corpus_chunks"], 65)
        self.assertEqual(overall["rag"]["summary"]["faithfulness"], 0.75)

    def test_default_summary_is_compact_aggregates_only(self):
        reports = {
            "functional": {
                "passed": 1,
                "total": 1,
                "mean_latency_ms": 1200.0,
                "wall_clock_ms": 1300.0,
                "cases": [_functional_case("a")],
            },
            "recall": {
                "mode": "vector",
                "corpus_chunks": 65,
                "server_retriever_k": 3,
                "recall_by_k": {"3": 1.0},
                "case_hit_rate_at_server_k": 1.0,
                "mrr_first_relevant": 1.0,
                "cases": [_recall_case("a")],
            },
            "rag_metrics": {
                "k": 3,
                "summary": {"faithfulness": 0.75, "answer_relevance": 0.8},
                "cases": [_rag_case("a")],
            },
        }

        summary = build_summary(reports)

        self.assertNotIn("cases", summary)
        self.assertNotIn("sources", summary)
        self.assertEqual(summary["functional"]["passed"], 1)
        self.assertEqual(summary["functional"]["pass_rate"], 1.0)
        self.assertEqual(summary["retrieval"]["recall_at_k"], 1.0)
        self.assertEqual(summary["rag"]["faithfulness"], 0.75)

    def test_case_missing_from_recall_is_null_and_reported(self):
        reports = {
            "functional": {
                "passed": 1,
                "total": 1,
                "cases": [_functional_case("escalate")],
            },
            "recall": {
                "skipped_ids": ["escalate"],
                "cases": [],
            },
            "rag_metrics": {
                "skipped_ids": ["escalate"],
                "cases": [],
            },
        }

        row = build_summary(reports, full=True)["cases"][0]

        self.assertIsNone(row["recall"])
        self.assertIsNone(row["rag"])
        self.assertEqual(row["missing"], ["recall", "rag_metrics"])

    def test_order_follows_functional_report_then_extras(self):
        reports = {
            "functional": {
                "passed": 2,
                "total": 2,
                "cases": [_functional_case("b"), _functional_case("a")],
            },
            "recall": None,
            "rag_metrics": {"cases": [_rag_case("c")]},
        }

        summary = build_summary(reports, full=True)

        self.assertEqual([row["id"] for row in summary["cases"]], ["b", "a", "c"])
        self.assertIn("c present in rag_metrics but not in functional report", summary["warnings"])
        self.assertTrue(summary["cases"][2]["missing"] == ["functional", "recall"])

    def test_missing_optional_source_degrades_with_warning(self):
        reports = {
            "functional": {"passed": 1, "total": 1, "cases": [_functional_case("a")]},
            "recall": None,
            "rag_metrics": None,
        }

        summary = build_summary(reports, full=True)

        self.assertIsNone(summary["overall"]["retrieval"])
        self.assertIsNone(summary["overall"]["rag"])
        self.assertIsNone(summary["cases"][0]["recall"])
        self.assertIsNone(summary["cases"][0]["rag"])
        self.assertIn(
            "recall report is missing; retrieval metrics are null", summary["warnings"]
        )

    def test_undeclared_case_gap_warns_but_declared_skip_does_not(self):
        reports = {
            "functional": {
                "passed": 2,
                "total": 2,
                "cases": [_functional_case("a"), _functional_case("b")],
            },
            "recall": {
                "skipped_ids": ["b"],
                "cases": [_recall_case("a")],
            },
            "rag_metrics": None,
        }

        summary = build_summary(reports, full=True)

        self.assertFalse(
            any("missing from recall" in warning for warning in summary["warnings"])
        )


class AmainTest(unittest.TestCase):
    def test_writes_summary_with_missing_optional_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "report.json"
            out_path = tmp_path / "summary.json"
            report_path.write_text(
                json.dumps(
                    {
                        "passed": 1,
                        "total": 1,
                        "cases": [_functional_case("a")],
                    }
                ),
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--report",
                    str(report_path),
                    "--recall",
                    str(tmp_path / "missing_recall.json"),
                    "--rag",
                    str(tmp_path / "missing_rag.json"),
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(amain(args), 0)

            summary = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["functional"]["passed"], 1)
            self.assertIsNone(summary["retrieval"])
            self.assertIsNone(summary["rag"])
            self.assertNotIn("cases", summary)

    def test_missing_functional_report_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "summary.json"
            args = parse_args(
                [
                    "--report",
                    str(tmp_path / "missing_report.json"),
                    "--recall",
                    str(tmp_path / "missing_recall.json"),
                    "--rag",
                    str(tmp_path / "missing_rag.json"),
                    "--out",
                    str(out_path),
                ]
            )
            self.assertEqual(amain(args), 1)
            self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
