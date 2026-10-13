import unittest

from fastapi.testclient import TestClient

from app.main import app


class KBUnitTest(unittest.TestCase):
    """KB routes are agent-only: anonymous requests must be rejected."""

    def setUp(self):
        self.client = TestClient(app)

    def test_kb_sources_requires_auth(self):
        resp = self.client.get("/kb/sources")
        self.assertIn(resp.status_code, (401, 403))

    def test_kb_reset_requires_auth(self):
        resp = self.client.post("/kb/reset")
        self.assertIn(resp.status_code, (401, 403))


if __name__ == "__main__":
    unittest.main()
