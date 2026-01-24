"""Test the SAM3 system with NFC25 fursuit database."""

import json
import os
import random
import tempfile

from PIL import Image

# NFC25 database paths
NFC25_DIR = "/media/user/SSD2TB/nfc25-fursuits"
NFC25_JSON = os.path.join(NFC25_DIR, "nfc25-fursuit-list.json")
NFC25_IMAGES = os.path.join(NFC25_DIR, "fursuit_images")


def load_nfc25_mapping() -> dict[str, dict]:
    """Load mapping from image filename to character info."""
    with open(NFC25_JSON) as f:
        data = json.load(f)

    mapping = {}
    for fursuit in data["FursuitList"]:
        image_url = fursuit.get("ImageUrl", "")
        if image_url:
            filename = image_url.split("/")[-1]
            mapping[filename] = {
                "nickname": fursuit.get("NickName"),
                "species": fursuit.get("Species"),
                "worn_by": fursuit.get("WornBy"),
                "country": fursuit.get("CountryName"),
            }
    return mapping


def get_available_images(mapping: dict) -> list[tuple[str, dict]]:
    """Get list of (filepath, info) for images that exist on disk."""
    available = []
    for filename, info in mapping.items():
        filepath = os.path.join(NFC25_IMAGES, filename)
        if os.path.exists(filepath):
            available.append((filepath, info))
    return available


def test_embedding_similarity():
    """Test that embeddings of the same character are similar."""
    from sam3_pursuit.models.embedder import FursuitEmbedder
    import numpy as np

    print("Loading embedder...")
    embedder = FursuitEmbedder()

    mapping = load_nfc25_mapping()
    images = get_available_images(mapping)

    # Pick 5 random images
    random.seed(42)
    sample = random.sample(images, 5)

    print(f"\nGenerating embeddings for {len(sample)} images...")
    embeddings = []
    for filepath, info in sample:
        img = Image.open(filepath)
        emb = embedder.embed(img)
        embeddings.append((info["nickname"], emb))
        print(f"  {info['nickname']}: embedding shape {emb.shape}")

    # Compute pairwise cosine similarities
    print("\nPairwise cosine similarities:")
    for i, (name1, emb1) in enumerate(embeddings):
        for j, (name2, emb2) in enumerate(embeddings):
            if i < j:
                # Cosine similarity (embeddings are already L2 normalized)
                sim = np.dot(emb1, emb2)
                print(f"  {name1} <-> {name2}: {sim:.4f}")


def test_indexing_and_search():
    """Test indexing images and searching for matches."""
    from sam3_pursuit import SAM3FursuitIdentifier

    # Use temporary database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        index_path = os.path.join(tmpdir, "test.index")

        print("Initializing identifier...")
        identifier = SAM3FursuitIdentifier(db_path=db_path, index_path=index_path)

        mapping = load_nfc25_mapping()
        images = get_available_images(mapping)

        # Index 20 random characters
        random.seed(42)
        sample = random.sample(images, 20)

        print(f"\nIndexing {len(sample)} images...")
        for filepath, info in sample:
            identifier.add_images(
                character_name=info["nickname"],
                image_paths=[filepath]
            )

        print(f"\nIndex now contains {identifier.index.size} embeddings")

        # Test: Search for one of the indexed images
        # It should find itself as the top match
        print("\n--- Self-identification test ---")
        test_img_path, test_info = sample[0]
        test_img = Image.open(test_img_path)

        results = identifier.identify(test_img, top_k=5)

        print(f"Query: {test_info['nickname']}")
        print("Results:")
        for i, r in enumerate(results, 1):
            match = "✓" if r.character_name == test_info["nickname"] else " "
            print(f"  {i}. [{match}] {r.character_name} (confidence: {r.confidence:.2%})")

        # Test with a different image from the set
        print("\n--- Cross-identification test ---")
        test_img_path2, test_info2 = sample[5]
        test_img2 = Image.open(test_img_path2)

        results2 = identifier.identify(test_img2, top_k=5)

        print(f"Query: {test_info2['nickname']}")
        print("Results:")
        for i, r in enumerate(results2, 1):
            match = "✓" if r.character_name == test_info2["nickname"] else " "
            print(f"  {i}. [{match}] {r.character_name} (confidence: {r.confidence:.2%})")


def test_accuracy_batch():
    """Test identification accuracy on a batch of images."""
    from sam3_pursuit import SAM3FursuitIdentifier

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        index_path = os.path.join(tmpdir, "test.index")

        print("Initializing identifier...")
        identifier = SAM3FursuitIdentifier(db_path=db_path, index_path=index_path)

        mapping = load_nfc25_mapping()
        images = get_available_images(mapping)

        # Split into index set and test set
        random.seed(123)
        random.shuffle(images)

        index_set = images[:100]  # Index these
        test_set = images[:20]    # Test with subset of indexed images

        print(f"\nIndexing {len(index_set)} images...")
        for filepath, info in index_set:
            identifier.add_images(info["nickname"], [filepath])

        # Test accuracy
        print(f"\nTesting identification on {len(test_set)} images...")
        correct_top1 = 0
        correct_top5 = 0

        for filepath, info in test_set:
            img = Image.open(filepath)
            results = identifier.identify(img, top_k=5)

            top1_correct = results[0].character_name == info["nickname"] if results else False
            top5_correct = any(r.character_name == info["nickname"] for r in results)

            if top1_correct:
                correct_top1 += 1
            if top5_correct:
                correct_top5 += 1

        print(f"\nResults:")
        print(f"  Top-1 accuracy: {correct_top1}/{len(test_set)} ({100*correct_top1/len(test_set):.1f}%)")
        print(f"  Top-5 accuracy: {correct_top5}/{len(test_set)} ({100*correct_top5/len(test_set):.1f}%)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_nfc25.py [embedding|search|accuracy]")
        print()
        print("Tests:")
        print("  embedding  - Test embedding generation and similarity")
        print("  search     - Test indexing and search")
        print("  accuracy   - Test identification accuracy on batch")
        sys.exit(1)

    test_name = sys.argv[1]

    if test_name == "embedding":
        test_embedding_similarity()
    elif test_name == "search":
        test_indexing_and_search()
    elif test_name == "accuracy":
        test_accuracy_batch()
    else:
        print(f"Unknown test: {test_name}")
        sys.exit(1)
