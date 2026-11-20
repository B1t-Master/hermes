import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas import CSATIn


class CSATSchemaTest(unittest.TestCase):
    def test_accepts_whole_range(self):
        for rating in (1, 3, 5):
            self.assertEqual(CSATIn(rating=rating).rating, rating)

    def test_rejects_out_of_range(self):
        for rating in (0, 6):
            with self.assertRaises(ValidationError):
                CSATIn(rating=rating)


class CSATRouteTest(unittest.TestCase):
    """The rating route is passenger-owned: unauthenticated calls are rejected."""

    def setUp(self):
        self.client = TestClient(app)

    def test_requires_auth(self):
        resp = self.client.post(
            "/chat/conversations/00000000-0000-0000-0000-000000000000/csat",
            json={"rating": 5},
        )
        self.assertIn(resp.status_code, (401, 403))


if __name__ == "__main__":
    unittest.main()
