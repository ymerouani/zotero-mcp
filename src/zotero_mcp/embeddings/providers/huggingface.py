"""Local HuggingFace embedding function, backed by sentence-transformers."""

import logging
import os
from typing import Any

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import BaseEmbeddingFunction

logger = logging.getLogger(__name__)


@register_embedding_function
class HuggingFaceEmbeddingFunction(BaseEmbeddingFunction):
    """Custom HuggingFace embedding function for ChromaDB using sentence-transformers.

    Registered under the name "huggingface" so ChromaDB rebuilds it (rather than
    its own incompatible built-in of the same name) when reloading a persisted
    collection's config (see OpenAIEmbeddingFunction for details).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str | None = None):
        self.model_name = model_name
        # Device placement: explicit config wins, then ZOTERO_EMBEDDING_DEVICE,
        # then sentence-transformers' own auto-detection (upstream default).
        # Set "cpu" to keep the CUDA device free for other GPU work (e.g. a
        # local LLM server); a 0.6B embedder runs happily on CPU and the
        # vector space is identical.
        self.device = device or os.environ.get("ZOTERO_EMBEDDING_DEVICE") or None

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name} (device={self.device or 'auto'})")
            if self.device:
                self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
            else:
                self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except ImportError:
            raise ImportError("sentence-transformers package is required for HuggingFace embeddings. Install with: pip install sentence-transformers")

        # Read limit from model metadata; conservative fallback
        self.max_input_tokens = getattr(self.model, "max_seq_length", 500)

    @staticmethod
    def name() -> str:
        return "huggingface"

    def get_config(self) -> dict[str, Any]:
        config = {
            "model_name": self.model_name,
            # ChromaDB's built-in "huggingface" EF requires api_key_env_var in
            # addition to model_name and asserts without it. Persisting the key
            # keeps the config buildable by either class (issue #382); our own
            # build_from_config ignores it (we embed locally, no API key).
            "api_key_env_var": "HUGGINGFACE_API_KEY",
        }
        if self.device:
            config["device"] = self.device
        return config

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HuggingFaceEmbeddingFunction":
        return HuggingFaceEmbeddingFunction(
            model_name=config.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
            device=config.get("device"),
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using HuggingFace model."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate using the model's own tokenizer."""
        tokenizer = getattr(self.model, 'tokenizer', None)
        if tokenizer is not None:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) > max_tokens:
                encoded = encoded[:max_tokens]
                text = tokenizer.decode(encoded)
        else:
            max_chars = max_tokens * 2
            if len(text) > max_chars:
                text = text[:max_chars]
        return text
