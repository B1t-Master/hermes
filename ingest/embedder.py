from app.config import settings

EMBEDDING_DIM = 384  # bge-small-en-v1.5 outputs 384 dims (not 512)


class Embedder:
    def __init__(self, model_name: str = settings.embedding_model):
        self.model_name = model_name
        self._model = None

    @property
    def available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed in this venv. "
                "Run: pip install torch sentence-transformers "
                "(see README 'Cost notes')."
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        return [vec.tolist() for vec in model.encode(texts, normalize_embeddings=True)]