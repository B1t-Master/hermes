import unittest
import uuid

from fastapi.testclient import TestClient

from app import metrics
from app.api.agents import _mean, _p95
from app.main import app


class _FakeSession:
    """Captures objects passed to session.add() without a DB."""

    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj) -> None:
        self.added.append(obj)


class AddMetricTest(unittest.TestCase):
    def test_queues_row_with_fields(self):
        session = _FakeSession()
        cid = uuid.uuid4()

        metrics.add_metric(session, cid, metrics.RESPONSE_TIME_MS, 123.4)

        self.assertEqual(len(session.added), 1)
        row = session.added[0]
        self.assertEqual(row.conversation_id, cid)
        self.assertEqual(row.metric, metrics.RESPONSE_TIME_MS)
        self.assertEqual(row.value, 123.4)

    def test_keeps_int_as_float(self):
        session = _FakeSession()
        metrics.add_metric(session, uuid.uuid4(), metrics.RETRIEVAL_HITS, 3)
        self.assertEqual(session.added[0].value, 3.0)

    def test_ignores_unknown_metric(self):
        session = _FakeSession()
        metrics.add_metric(session, uuid.uuid4(), "bogus", 1)
        self.assertEqual(session.added, [])

    def test_all_metric_names_accepted(self):
        for name in metrics.METRIC_NAMES:
            session = _FakeSession()
            metrics.add_metric(session, uuid.uuid4(), name, 1)
            self.assertEqual(len(session.added), 1, name)


class MetricStatsTest(unittest.TestCase):
    def test_mean_empty_and_values(self):
        self.assertIsNone(_mean([]))
        self.assertEqual(_mean([2, 4]), 3.0)

    def test_p95(self):
        self.assertIsNone(_p95([]))
        self.assertEqual(_p95([7]), 7)
        self.assertEqual(_p95(list(range(1, 101))), 95)


class MetricsRouteTest(unittest.TestCase):
    """The aggregation endpoint is agent-only."""

    def setUp(self):
        self.client = TestClient(app)

    def test_requires_auth(self):
        self.assertIn(self.client.get("/agents/metrics").status_code, (401, 403))


if __name__ == "__main__":
    unittest.main()
