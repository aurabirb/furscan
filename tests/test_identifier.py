"""Basic tests for the SAM3 fursuit recognition system."""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.storage.database import Database, Detection
from sam3_pursuit.storage.vector_index import VectorIndex


class TestDatabase(unittest.TestCase):
    """Tests for the database module."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_file.close()
        self.db = Database(self.temp_file.name)

    def tearDown(self):
        """Clean up temporary database."""
        os.unlink(self.temp_file.name)

    def test_add_detection(self):
        """Test adding a detection record."""
        detection = Detection(
            id=None,
            post_id="12345",
            character_name="TestChar",
            embedding_id=0,
            bbox_x=10,
            bbox_y=20,
            bbox_width=100,
            bbox_height=100,
            confidence=0.95
        )

        row_id = self.db.add_detection(detection)
        self.assertIsNotNone(row_id)

    def test_get_detection_by_embedding_id(self):
        """Test retrieving a detection by embedding ID."""
        detection = Detection(
            id=None,
            post_id="12345",
            character_name="TestChar",
            embedding_id=42,
            bbox_x=10,
            bbox_y=20,
            bbox_width=100,
            bbox_height=100,
            confidence=0.95
        )

        self.db.add_detection(detection)
        retrieved = self.db.get_detection_by_embedding_id(42)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.post_id, "12345")
        self.assertEqual(retrieved.character_name, "TestChar")

    def test_get_stats(self):
        """Test getting database statistics."""
        # Add some detections
        for i in range(5):
            detection = Detection(
                id=None,
                post_id=f"post_{i}",
                character_name="TestChar",
                embedding_id=i,
                bbox_x=0,
                bbox_y=0,
                bbox_width=100,
                bbox_height=100,
                confidence=0.9
            )
            self.db.add_detection(detection)

        stats = self.db.get_stats()
        self.assertEqual(stats["total_detections"], 5)
        self.assertEqual(stats["unique_characters"], 1)
        self.assertEqual(stats["unique_posts"], 5)


class TestVectorIndex(unittest.TestCase):
    """Tests for the vector index module."""

    def setUp(self):
        """Create a temporary index for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".index")
        self.temp_file.close()
        os.unlink(self.temp_file.name)  # Remove so VectorIndex creates new
        self.index = VectorIndex(
            index_path=self.temp_file.name,
            embedding_dim=768
        )

    def tearDown(self):
        """Clean up temporary index."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_add_and_search(self):
        """Test adding embeddings and searching."""
        # Create random embeddings
        embeddings = np.random.randn(10, 768).astype(np.float32)
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to index
        start_id = self.index.add(embeddings)
        self.assertEqual(start_id, 0)
        self.assertEqual(self.index.size, 10)

        # Search with first embedding
        query = embeddings[0]
        distances, indices = self.index.search(query, top_k=3)

        # First result should be the same embedding (distance ~0)
        self.assertEqual(indices[0][0], 0)
        self.assertLess(distances[0][0], 0.01)

    def test_save_and_load(self):
        """Test saving and loading index."""
        embeddings = np.random.randn(5, 768).astype(np.float32)
        self.index.add(embeddings)
        self.index.save()

        # Load in new instance
        new_index = VectorIndex(
            index_path=self.temp_file.name,
            embedding_dim=768
        )
        self.assertEqual(new_index.size, 5)


class TestConfig(unittest.TestCase):
    """Tests for the config module."""

    def test_get_device(self):
        """Test device selection."""
        device = Config.get_device()
        self.assertIn(device, ["cuda", "mps", "cpu"])

    def test_paths(self):
        """Test path configurations."""
        self.assertTrue(Config.DB_PATH.endswith(".db"))
        self.assertTrue(Config.INDEX_PATH.endswith(".index"))


if __name__ == "__main__":
    unittest.main()
