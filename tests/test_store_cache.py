import asyncio
import os
import tempfile
import unittest

from ingest.store import (
    CachedStore,
    Fragment,
    SqliteStore,
    Store,
    bump_corpus_version,
    rank_fragments,
)


def _frag(content, embedding=None, title="Doc", source_url="file::x", topic="T", idx=0):
    return Fragment(
        content=content,
        source_url=source_url,
        title=title,
        topic=topic,
        chunk_index=idx,
        embedding=embedding,
    )


class RankFragmentsTest(unittest.TestCase):
    def test_vector_mode_orders_by_cosine_similarity(self):
        fragments = [
            _frag("flight delay rebooking", [1.0, 0.0]),
            _frag("baggage allowance", [0.0, 1.0]),
        ]
        results = rank_fragments(fragments, [1.0, 0.0], "ignored", k=5)
        self.assertEqual(results[0].content, "flight delay rebooking")
        self.assertGreater(results[0].score, results[1].score)

    def test_unembedded_chunks_are_skipped_in_vector_mode(self):
        fragments = [_frag("no vector", None), _frag("has vector", [1.0, 0.0])]
        results = rank_fragments(fragments, [1.0, 0.0], "q", k=5)
        self.assertEqual([r.content for r in results], ["has vector"])

    def test_keyword_fallback_scores_term_overlap(self):
        fragments = [
            _frag("refund and rebooking policy"),
            _frag("refund only"),
            _frag("unrelated text"),
        ]
        results = rank_fragments(fragments, None, "refund rebooking", k=5)
        self.assertEqual(results[0].content, "refund and rebooking policy")
        self.assertNotIn("unrelated text", [r.content for r in results])

    def test_empty_query_text_returns_nothing(self):
        self.assertEqual(rank_fragments([_frag("a")], None, "..", k=5), [])


class _FakeStore(Store):
    """Counts dump calls so cache behaviour is observable without a DB."""

    def __init__(self, fragments):
        self._fragments = fragments
        self.dump_calls = 0

    async def get_hash(self, source_key):
        return None

    async def replace_source(self, **kwargs):
        return "new"

    async def list_documents(self):
        return []

    async def delete_source(self, source_key):
        return False

    async def dump_fragments(self):
        self.dump_calls += 1
        return list(self._fragments)

    async def close(self):
        return None

    async def search(self, query_embedding, query_text, k=5):
        raise AssertionError("CachedStore must not delegate search to the inner store")


class CachedStoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_loads_once_then_serves_from_memory(self):
        inner = _FakeStore([_frag("alpha", [1.0, 0.0])])
        store = CachedStore(inner, ttl_seconds=0)
        await store.search([1.0, 0.0], "alpha", k=5)
        await store.search([1.0, 0.0], "alpha", k=5)
        self.assertEqual(inner.dump_calls, 1)

    async def test_corpus_version_bump_forces_reload(self):
        inner = _FakeStore([_frag("alpha", [1.0, 0.0])])
        store = CachedStore(inner, ttl_seconds=0)
        await store.search([1.0, 0.0], "alpha", k=5)
        bump_corpus_version()
        await store.search([1.0, 0.0], "alpha", k=5)
        self.assertEqual(inner.dump_calls, 2)

    async def test_invalidate_forces_reload(self):
        inner = _FakeStore([_frag("alpha", [1.0, 0.0])])
        store = CachedStore(inner, ttl_seconds=0)
        await store.search([1.0, 0.0], "alpha", k=5)
        store.invalidate()
        await store.search([1.0, 0.0], "alpha", k=5)
        self.assertEqual(inner.dump_calls, 2)

    async def test_ttl_expiry_forces_reload(self):
        inner = _FakeStore([_frag("alpha", [1.0, 0.0])])
        store = CachedStore(inner, ttl_seconds=0.01)
        await store.search([1.0, 0.0], "alpha", k=5)
        await asyncio.sleep(0.03)
        await store.search([1.0, 0.0], "alpha", k=5)
        self.assertEqual(inner.dump_calls, 2)

    async def test_mutations_invalidate(self):
        inner = _FakeStore([_frag("alpha", [1.0, 0.0])])
        store = CachedStore(inner, ttl_seconds=0)
        await store.search([1.0, 0.0], "alpha", k=5)
        await store.replace_source()
        await store.search([1.0, 0.0], "alpha", k=5)
        self.assertEqual(inner.dump_calls, 2)


class SqliteDumpFragmentsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fd, self._path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        self.store = SqliteStore(self._path)
        await self.store.connect()

    async def asyncTearDown(self):
        await self.store.close()
        try:
            os.unlink(self._path)
        except OSError:
            pass

    async def _seed(self):
        await self.store.replace_source(
            source_key="file::a.pdf",
            title="A",
            topic="Flights",
            doc_type="pdf",
            content_hash="h1",
            chunks=["flight delay rebooking policy", "baggage allowance limits"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            source_url=None,
        )

    async def test_dump_round_trips_embeddings(self):
        await self._seed()
        fragments = await self.store.dump_fragments()
        self.assertEqual(len(fragments), 2)
        self.assertEqual(fragments[0].embedding, [1.0, 0.0])
        self.assertEqual(fragments[0].title, "A")

    async def test_search_vector_and_keyword(self):
        await self._seed()
        vector = await self.store.search([1.0, 0.0], "q", k=1)
        self.assertEqual(vector[0].content, "flight delay rebooking policy")

        keyword = await self.store.search(None, "baggage", k=5)
        self.assertEqual([r.content for r in keyword], ["baggage allowance limits"])


if __name__ == "__main__":
    unittest.main()
