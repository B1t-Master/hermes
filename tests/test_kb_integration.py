import io
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.deps import get_current_agent
from app.main import app
from ingest.pipeline import IngestReport


class _FakeAgent:
    id = "00000000-0000-0000-0000-000000000000"
    name = "test-agent"
    email = "test@example.com"


class _FakeStore:
    async def ensure_schema(self):
        return None

    async def close(self):
        return None


class KBIntegrationTest(unittest.TestCase):
    """Upload/URL validation + a hermetic happy-path ingest.

    Auth is overridden (no DB) and the ingest store/pipeline are patched so
    the suite never touches Postgres or downloads models.
    """

    def setUp(self):
        self.client = TestClient(app)
        app.dependency_overrides[get_current_agent] = lambda: _FakeAgent()
        self.headers = {"Authorization": "Bearer test"}

    def tearDown(self):
        app.dependency_overrides.clear()
        for path in Path("data_sources").glob("test_*"):
            try:
                path.unlink()
            except OSError:
                pass

    def test_upload_rejects_non_pdf_docx(self):
        data = io.BytesIO(b"hello world")
        resp = self.client.post(
            "/kb/upload",
            files={"files": ("test.txt", data, "text/plain")},
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 400)

    def test_urls_validation(self):
        resp = self.client.post(
            "/kb/urls", json={"urls": ["not-a-url"]}, headers=self.headers
        )
        self.assertEqual(resp.status_code, 400)

    def test_upload_rejects_oversized_file(self):
        data = io.BytesIO(b"%PDF-1.4\n" + b"x" * 64)
        with patch("app.api.kb.MAX_UPLOAD_BYTES", 16):
            resp = self.client.post(
                "/kb/upload",
                files={"files": ("test.pdf", data, "application/pdf")},
                headers=self.headers,
            )
        self.assertEqual(resp.status_code, 400)

    def test_upload_accepts_pdf_and_ingests(self):
        data = io.BytesIO(b"%PDF-1.4\n%fake")
        report = IngestReport()
        report.new = 1
        with (
            patch("ingest.store.get_store", new=AsyncMock(return_value=_FakeStore())),
            patch("ingest.pipeline.ingest_sources", new=AsyncMock(return_value=report)) as ingest,
        ):
            resp = self.client.post(
                "/kb/upload?embed=false",
                files={"files": ("test.pdf", data, "application/pdf")},
                headers=self.headers,
            )
        self.assertEqual(resp.status_code, 200)
        ingest.assert_awaited_once()
        self.assertEqual(resp.json()["new"], 1)


if __name__ == "__main__":
    unittest.main()
