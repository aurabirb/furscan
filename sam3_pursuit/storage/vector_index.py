"""FAISS vector index for similarity search."""

import os
from typing import Optional

import faiss
import numpy as np

from sam3_pursuit.config import Config


class VectorIndex:
    """FAISS vector index for embedding similarity search.

    Uses HNSW (Hierarchical Navigable Small World) index for
    fast approximate nearest neighbor search.
    """

    def __init__(
        self,
        index_path: str = Config.INDEX_PATH,
        embedding_dim: int = Config.EMBEDDING_DIM,
        hnsw_m: int = Config.HNSW_M,
        ef_construction: int = Config.HNSW_EF_CONSTRUCTION,
        ef_search: int = Config.HNSW_EF_SEARCH
    ):
        """Initialize the vector index.

        Args:
            index_path: Path to save/load the FAISS index.
            embedding_dim: Dimension of embeddings.
            hnsw_m: HNSW M parameter (number of connections).
            ef_construction: HNSW efConstruction parameter.
            ef_search: HNSW efSearch parameter.
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> faiss.Index:
        """Load existing index or create a new one.

        Returns:
            FAISS index object.
        """
        if os.path.exists(self.index_path):
            print(f"Loading existing index from {self.index_path}")
            index = faiss.read_index(self.index_path)
            print(f"Index loaded with {index.ntotal} vectors")
        else:
            print(f"Creating new HNSW index with dimension {self.embedding_dim}")
            index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search

        return index

    def add(self, embeddings: np.ndarray) -> int:
        """Add embeddings to the index.

        Args:
            embeddings: Array of embeddings, shape (n, embedding_dim).

        Returns:
            Starting ID of added embeddings.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        embeddings = embeddings.astype(np.float32)
        start_id = self.index.ntotal
        self.index.add(embeddings)

        return start_id

    def search(
        self,
        query: np.ndarray,
        top_k: int = Config.DEFAULT_TOP_K
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings.

        Args:
            query: Query embedding, shape (embedding_dim,) or (n, embedding_dim).
            top_k: Number of results to return.

        Returns:
            Tuple of (distances, indices) arrays.
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = query.astype(np.float32)
        distances, indices = self.index.search(query, top_k)

        return distances, indices

    def save(self):
        """Save index to disk."""
        print(f"Saving index to {self.index_path}")
        faiss.write_index(self.index, self.index_path)

    @property
    def size(self) -> int:
        """Get number of vectors in the index."""
        return self.index.ntotal

    def reset(self):
        """Reset the index (remove all vectors)."""
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_m)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
