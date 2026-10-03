import os
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """
    Embedding Generator using Jina AI API (Week 3 concept).
    Supports task-specific embeddings (retrieval.passage vs retrieval.query).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-embeddings-v3",
        timeout: int = 30,
    ):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY is not set. Please set it in your .env file or pass it to constructor."
            )
        self.model = model
        self.timeout = timeout
        self.api_url = "https://api.jina.ai/v1/embeddings"

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        results = self._call_api(
            input_texts=[text], task="retrieval.query"
        )
        return results[0]

    def embed_documents(
        self, texts: list[str], batch_size: int = 16
    ) -> list[list[float]]:
        """
        Embed a list of document passages in batches.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._call_api(
                input_texts=batch, task="retrieval.passage"
            )
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _call_api(
        self, input_texts: list[str], task: str = "retrieval.passage"
    ) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "task": task,
            "dimensions": 1024,
            "input": input_texts,
        }

        response = requests.post(
            self.api_url, headers=headers, json=payload, timeout=self.timeout
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Jina API error (HTTP {response.status_code}): {response.text}"
            )

        data = response.json().get("data", [])
        # Preserve original order by index
        sorted_items = sorted(data, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_items]
