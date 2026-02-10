"""Tests for combine and split CLI commands."""
import os
import tempfile

import numpy as np
import pytest

from sam3_pursuit.config import Config
from sam3_pursuit.storage.database import Database, Detection
from sam3_pursuit.storage.vector_index import VectorIndex
from sam3_pursuit.api.cli import (
    _copy_detections,
    _copy_dataset_files,
    _get_dataset_dir,
    _get_dataset_paths,
    _get_source_subdirs,
)


@pytest.fixture
def tmp_dir(tmp_path):
    """Use tmp_path as the base for all dataset files and image dirs."""
    import sam3_pursuit.api.cli as cli_mod
    from pathlib import Path

    orig_get_paths = cli_mod._get_dataset_paths
    orig_get_dir = cli_mod._get_dataset_dir

    def _patched_get_dataset_paths(dataset):
        return (
            str(tmp_path / f"{dataset}.db"),
            str(tmp_path / f"{dataset}.index"),
        )

    def _patched_get_dataset_dir(dataset):
        if dataset == Config.DEFAULT_DATASET:
            return tmp_path
        return tmp_path / "datasets" / dataset

    cli_mod._get_dataset_paths = _patched_get_dataset_paths
    cli_mod._get_dataset_dir = _patched_get_dataset_dir
    yield tmp_path
    cli_mod._get_dataset_paths = orig_get_paths
    cli_mod._get_dataset_dir = orig_get_dir


def _make_dataset(tmp_path, name, detections_data):
    """Create a dataset with given detections.

    detections_data: list of (post_id, character_name, source, preprocessing_info) tuples
    Returns (db, index, embeddings_dict) where embeddings_dict maps embedding_id -> embedding vector.
    """
    db_path = str(tmp_path / f"{name}.db")
    index_path = str(tmp_path / f"{name}.index")

    db = Database(db_path)
    index = VectorIndex(index_path)

    embeddings = {}
    for i, (post_id, char_name, source, preproc) in enumerate(detections_data):
        # Create a deterministic embedding based on content
        rng = np.random.RandomState(hash((post_id, char_name, source, preproc)) % (2**31))
        emb = rng.randn(Config.EMBEDDING_DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # normalize

        emb_id = index.add(emb.reshape(1, -1))
        embeddings[emb_id] = emb

        det = Detection(
            id=None,
            post_id=post_id,
            character_name=char_name,
            embedding_id=emb_id,
            bbox_x=0, bbox_y=0, bbox_width=100, bbox_height=100,
            confidence=0.95,
            segmentor_model="test",
            source=source,
            preprocessing_info=preproc,
            git_version="test",
        )
        db.add_detection(det)

    index.save()
    return db, index, embeddings


def _get_all_detections(db):
    """Get all detections from a database."""
    conn = db._connect()
    cursor = conn.cursor()
    cursor.execute(f"SELECT {db._SELECT_FIELDS} FROM detections ORDER BY embedding_id")
    return [db._row_to_detection(row) for row in cursor.fetchall()]


class TestCopyDetections:
    def test_basic_copy(self, tmp_path):
        """Test copying detections from one dataset to another."""
        source_db, source_index, source_embs = _make_dataset(tmp_path, "src", [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post2", "CharB", "manual", "solid_128"),
        ])

        target_db = Database(str(tmp_path / "tgt.db"))
        target_index = VectorIndex(str(tmp_path / "tgt.index"))

        detections = _get_all_detections(source_db)
        copied, skipped = _copy_detections(detections, source_index, target_db, target_index)

        assert copied == 2
        assert skipped == 0
        assert target_index.size == 2

        # Verify detections in target
        target_dets = _get_all_detections(target_db)
        assert len(target_dets) == 2
        assert target_dets[0].post_id == "post1"
        assert target_dets[1].post_id == "post2"

        # Verify embeddings match
        for det in target_dets:
            src_emb = np.zeros(Config.EMBEDDING_DIM, dtype=np.float32)
            tgt_emb = np.zeros(Config.EMBEDDING_DIM, dtype=np.float32)
            # Find original embedding_id for same post
            src_det = [d for d in detections if d.post_id == det.post_id][0]
            source_index.index.reconstruct(src_det.embedding_id, src_emb)
            target_index.index.reconstruct(det.embedding_id, tgt_emb)
            np.testing.assert_array_almost_equal(src_emb, tgt_emb)

        target_index.save()
        source_db.close()
        target_db.close()

    def test_multiple_characters_same_post(self, tmp_path):
        """Test that multiple characters from the same post are all copied."""
        source_db, source_index, _ = _make_dataset(tmp_path, "src_multi", [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post1", "CharB", "furtrack", "solid_128"),
        ])

        target_db = Database(str(tmp_path / "tgt_multi.db"))
        target_index = VectorIndex(str(tmp_path / "tgt_multi.index"))

        detections = _get_all_detections(source_db)
        copied, skipped = _copy_detections(detections, source_index, target_db, target_index)

        assert copied == 2
        assert skipped == 0

        target_dets = _get_all_detections(target_db)
        assert len(target_dets) == 2
        assert {d.character_name for d in target_dets} == {"CharA", "CharB"}

        target_index.save()
        source_db.close()
        target_db.close()

    def test_dedup(self, tmp_path):
        """Test that duplicates are skipped."""
        source_db, source_index, _ = _make_dataset(tmp_path, "src2", [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post2", "CharB", "manual", "solid_128"),
        ])

        # Pre-populate target with post1
        target_db, target_index, _ = _make_dataset(tmp_path, "tgt2", [
            ("post1", "CharA", "furtrack", "solid_128"),
        ])

        detections = _get_all_detections(source_db)
        copied, skipped = _copy_detections(detections, source_index, target_db, target_index)

        assert copied == 1  # only post2
        assert skipped == 1  # post1 was duplicate
        assert target_index.size == 2  # 1 original + 1 copied

        source_db.close()
        target_db.close()


class TestCombineCommand:
    def test_combine_two_datasets(self, tmp_path, tmp_dir):
        """Test combining two datasets."""
        import sys
        from sam3_pursuit.api.cli import combine_command
        import argparse

        _make_dataset(tmp_path, "ds_a", [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post2", "CharA", "furtrack", "solid_128"),
        ])
        _make_dataset(tmp_path, "ds_b", [
            ("post3", "CharB", "manual", "solid_128"),
        ])

        # Close the datasets (they'll be reopened by combine_command)
        args = argparse.Namespace(
            datasets=["ds_a", "ds_b"],
            output="combined",
        )
        combine_command(args)

        # Verify combined dataset
        combined_db = Database(str(tmp_path / "combined.db"))
        combined_index = VectorIndex(str(tmp_path / "combined.index"))

        dets = _get_all_detections(combined_db)
        assert len(dets) == 3
        assert combined_index.size == 3
        assert {d.post_id for d in dets} == {"post1", "post2", "post3"}

        combined_db.close()

    def test_combine_dedup_across_datasets(self, tmp_path, tmp_dir):
        """Test that combine deduplicates across datasets."""
        import argparse
        from sam3_pursuit.api.cli import combine_command

        _make_dataset(tmp_path, "ds_c", [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post2", "CharA", "furtrack", "solid_128"),
        ])
        _make_dataset(tmp_path, "ds_d", [
            ("post1", "CharA", "furtrack", "solid_128"),  # duplicate
            ("post3", "CharB", "manual", "solid_128"),
        ])

        args = argparse.Namespace(datasets=["ds_c", "ds_d"], output="combined2")
        combine_command(args)

        combined_db = Database(str(tmp_path / "combined2.db"))
        dets = _get_all_detections(combined_db)
        assert len(dets) == 3  # post1 only once
        combined_db.close()


class TestSplitCommand:
    def test_split_by_source(self, tmp_path, tmp_dir):
        """Test splitting by source."""
        import argparse
        from sam3_pursuit.api.cli import split_command

        _make_dataset(tmp_path, "ds_split", [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post2", "CharA", "furtrack", "solid_128"),
            ("post3", "CharB", "manual", "solid_128"),
            ("post4", "CharC", "barq", "solid_128"),
        ])

        args = argparse.Namespace(
            source_dataset="ds_split",
            output="ft_only",
            by_source="furtrack",
            by_character=None,
            shards=1,
        )
        split_command(args)

        result_db = Database(str(tmp_path / "ft_only.db"))
        result_index = VectorIndex(str(tmp_path / "ft_only.index"))
        dets = _get_all_detections(result_db)

        assert len(dets) == 2
        assert result_index.size == 2
        assert all(d.source == "furtrack" for d in dets)
        result_db.close()

    def test_split_by_character(self, tmp_path, tmp_dir):
        """Test splitting by character name."""
        import argparse
        from sam3_pursuit.api.cli import split_command

        _make_dataset(tmp_path, "ds_split2", [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post2", "CharB", "manual", "solid_128"),
            ("post3", "CharC", "barq", "solid_128"),
        ])

        args = argparse.Namespace(
            source_dataset="ds_split2",
            output="chars_ab",
            by_source=None,
            by_character="CharA,CharC",
            shards=1,
        )
        split_command(args)

        result_db = Database(str(tmp_path / "chars_ab.db"))
        dets = _get_all_detections(result_db)
        assert len(dets) == 2
        assert {d.character_name for d in dets} == {"CharA", "CharC"}
        result_db.close()

    def test_split_with_shards(self, tmp_path, tmp_dir):
        """Test splitting into multiple shards."""
        import argparse
        from sam3_pursuit.api.cli import split_command

        _make_dataset(tmp_path, "ds_shard", [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post2", "CharA", "furtrack", "solid_128"),
            ("post3", "CharA", "furtrack", "solid_128"),
            ("post4", "CharA", "furtrack", "solid_128"),
            ("post5", "CharA", "furtrack", "solid_128"),
            ("post6", "CharA", "furtrack", "solid_128"),
        ])

        args = argparse.Namespace(
            source_dataset="ds_shard",
            output="sharded",
            by_source="furtrack",
            by_character=None,
            shards=3,
        )
        split_command(args)

        # All 6 detections should be distributed across 3 shards
        total = 0
        all_post_ids = set()
        for i in range(3):
            shard_db = Database(str(tmp_path / f"sharded_{i}.db"))
            dets = _get_all_detections(shard_db)
            total += len(dets)
            all_post_ids.update(d.post_id for d in dets)
            shard_db.close()

        assert total == 6
        assert all_post_ids == {"post1", "post2", "post3", "post4", "post5", "post6"}

    def test_split_shards_deterministic(self, tmp_path, tmp_dir):
        """Test that shard assignment is deterministic."""
        import argparse
        from sam3_pursuit.api.cli import split_command

        data = [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post2", "CharA", "furtrack", "solid_128"),
            ("post3", "CharA", "furtrack", "solid_128"),
        ]

        # Run twice
        for run in range(2):
            name = f"ds_det_{run}"
            _make_dataset(tmp_path, name, data)
            args = argparse.Namespace(
                source_dataset=name,
                output=f"det_out_{run}",
                by_source="furtrack",
                by_character=None,
                shards=2,
            )
            split_command(args)

        # Compare shard contents
        for i in range(2):
            db0 = Database(str(tmp_path / f"det_out_0_{i}.db"))
            db1 = Database(str(tmp_path / f"det_out_1_{i}.db"))
            dets0 = {d.post_id for d in _get_all_detections(db0)}
            dets1 = {d.post_id for d in _get_all_detections(db1)}
            assert dets0 == dets1
            db0.close()
            db1.close()


class TestSplitAndRecombine:
    def test_split_then_combine_roundtrip(self, tmp_path, tmp_dir):
        """Test that splitting and recombining produces the same data."""
        import argparse
        from sam3_pursuit.api.cli import split_command, combine_command

        original_data = [
            ("post1", "CharA", "furtrack", "solid_128"),
            ("post2", "CharB", "furtrack", "solid_128"),
            ("post3", "CharC", "manual", "solid_128"),
            ("post4", "CharD", "barq", "solid_128"),
            ("post5", "CharA", "furtrack", "solid_128"),
            ("post6", "CharB", "manual", "solid_128"),
        ]
        orig_db, orig_index, _ = _make_dataset(tmp_path, "original", original_data)

        # Split by source into furtrack and non-furtrack
        split_command(argparse.Namespace(
            source_dataset="original",
            output="ft_split",
            by_source="furtrack",
            by_character=None,
            shards=1,
        ))
        split_command(argparse.Namespace(
            source_dataset="original",
            output="manual_split",
            by_source="manual",
            by_character=None,
            shards=1,
        ))
        split_command(argparse.Namespace(
            source_dataset="original",
            output="barq_split",
            by_source="barq",
            by_character=None,
            shards=1,
        ))

        # Recombine
        combine_command(argparse.Namespace(
            datasets=["ft_split", "manual_split", "barq_split"],
            output="recombined",
        ))

        # Verify recombined matches original
        recombined_db = Database(str(tmp_path / "recombined.db"))
        recombined_index = VectorIndex(str(tmp_path / "recombined.index"))

        orig_dets = _get_all_detections(orig_db)
        recom_dets = _get_all_detections(recombined_db)

        assert len(recom_dets) == len(orig_dets)
        assert {d.post_id for d in recom_dets} == {d.post_id for d in orig_dets}
        assert recombined_index.size == orig_index.size

        # Verify embeddings match
        for orig_det in orig_dets:
            recom_det = [d for d in recom_dets if d.post_id == orig_det.post_id
                         and d.source == orig_det.source
                         and d.preprocessing_info == orig_det.preprocessing_info][0]

            orig_emb = np.zeros(Config.EMBEDDING_DIM, dtype=np.float32)
            recom_emb = np.zeros(Config.EMBEDDING_DIM, dtype=np.float32)
            orig_index.index.reconstruct(orig_det.embedding_id, orig_emb)
            recombined_index.index.reconstruct(recom_det.embedding_id, recom_emb)
            np.testing.assert_array_almost_equal(orig_emb, recom_emb)

        # Verify source unchanged
        orig_dets_after = _get_all_detections(orig_db)
        assert len(orig_dets_after) == len(original_data)

        orig_db.close()
        recombined_db.close()

    def test_shard_then_combine_roundtrip(self, tmp_path, tmp_dir):
        """Test that sharding and recombining preserves all data."""
        import argparse
        from sam3_pursuit.api.cli import split_command, combine_command

        original_data = [
            (f"post{i}", "CharA", "furtrack", "solid_128")
            for i in range(10)
        ]
        orig_db, orig_index, _ = _make_dataset(tmp_path, "orig_shard", original_data)

        # Split into 3 shards
        split_command(argparse.Namespace(
            source_dataset="orig_shard",
            output="shard",
            by_source="furtrack",
            by_character=None,
            shards=3,
        ))

        # Recombine shards
        combine_command(argparse.Namespace(
            datasets=["shard_0", "shard_1", "shard_2"],
            output="unshard",
        ))

        unshard_db = Database(str(tmp_path / "unshard.db"))
        unshard_index = VectorIndex(str(tmp_path / "unshard.index"))

        orig_dets = _get_all_detections(orig_db)
        unshard_dets = _get_all_detections(unshard_db)

        assert len(unshard_dets) == len(orig_dets)
        assert unshard_index.size == orig_index.size
        assert {d.post_id for d in unshard_dets} == {d.post_id for d in orig_dets}

        orig_db.close()
        unshard_db.close()


def _make_image_files(base_dir, source_name, characters, is_default=False):
    """Create dummy image files for a dataset.

    characters: dict of {char_name: [post_id, ...]}
    Returns list of created file paths.
    """
    if is_default:
        source_dir = base_dir / f"{source_name}_images"
    else:
        source_dir = base_dir / source_name
    paths = []
    for char_name, post_ids in characters.items():
        char_dir = source_dir / char_name
        char_dir.mkdir(parents=True, exist_ok=True)
        for post_id in post_ids:
            img = char_dir / f"{post_id}.jpg"
            img.write_bytes(b"fake image data")
            paths.append(img)
    return paths


class TestCopyDatasetFiles:
    def test_basic_copy(self, tmp_path, tmp_dir):
        """Test copying all files from a non-default dataset."""
        src_dir = tmp_path / "datasets" / "ds_a"
        _make_image_files(src_dir, "furtrack", {"CharA": ["p1", "p2"], "CharB": ["p3"]})

        copied = _copy_dataset_files("ds_a", "ds_out")
        assert copied == 3

        out_dir = tmp_path / "datasets" / "ds_out"
        assert (out_dir / "furtrack" / "CharA" / "p1.jpg").exists()
        assert (out_dir / "furtrack" / "CharA" / "p2.jpg").exists()
        assert (out_dir / "furtrack" / "CharB" / "p3.jpg").exists()

    def test_filter_by_source(self, tmp_path, tmp_dir):
        """Test filtering by source subdir."""
        src_dir = tmp_path / "datasets" / "ds_b"
        _make_image_files(src_dir, "furtrack", {"CharA": ["p1"]})
        _make_image_files(src_dir, "barq", {"CharA": ["p2"]})

        copied = _copy_dataset_files("ds_b", "ds_out2", by_source="furtrack")
        assert copied == 1

        out_dir = tmp_path / "datasets" / "ds_out2"
        assert (out_dir / "furtrack" / "CharA" / "p1.jpg").exists()
        assert not (out_dir / "barq").exists()

    def test_filter_by_character(self, tmp_path, tmp_dir):
        """Test filtering by character name."""
        src_dir = tmp_path / "datasets" / "ds_c"
        _make_image_files(src_dir, "furtrack", {"CharA": ["p1"], "CharB": ["p2"], "CharC": ["p3"]})

        copied = _copy_dataset_files("ds_c", "ds_out3", by_character="CharA,CharC")
        assert copied == 2

        out_dir = tmp_path / "datasets" / "ds_out3"
        assert (out_dir / "furtrack" / "CharA" / "p1.jpg").exists()
        assert not (out_dir / "furtrack" / "CharB").exists()
        assert (out_dir / "furtrack" / "CharC" / "p3.jpg").exists()

    def test_sharding(self, tmp_path, tmp_dir):
        """Test that sharding splits files deterministically."""
        src_dir = tmp_path / "datasets" / "ds_d"
        posts = [f"post{i}" for i in range(10)]
        _make_image_files(src_dir, "furtrack", {"CharA": posts})

        all_copied = set()
        for shard_idx in range(3):
            copied = _copy_dataset_files(
                "ds_d", f"ds_shard_{shard_idx}",
                shard_idx=shard_idx, shards=3,
            )
            out_dir = tmp_path / "datasets" / f"ds_shard_{shard_idx}" / "furtrack" / "CharA"
            if out_dir.exists():
                shard_files = {f.stem for f in out_dir.iterdir()}
                all_copied.update(shard_files)

        assert all_copied == set(posts)

    def test_skip_existing(self, tmp_path, tmp_dir):
        """Test that existing files are not overwritten."""
        src_dir = tmp_path / "datasets" / "ds_e"
        _make_image_files(src_dir, "furtrack", {"CharA": ["p1"]})

        # Pre-create target file
        out_dir = tmp_path / "datasets" / "ds_out4" / "furtrack" / "CharA"
        out_dir.mkdir(parents=True)
        existing = out_dir / "p1.jpg"
        existing.write_bytes(b"original")

        copied = _copy_dataset_files("ds_e", "ds_out4")
        assert copied == 0
        assert existing.read_bytes() == b"original"

    def test_default_dataset_dirs(self, tmp_path, tmp_dir):
        """Test copying from default dataset (source_images dirs at root)."""
        _make_image_files(tmp_path, "furtrack", {"CharA": ["p1"]}, is_default=True)
        _make_image_files(tmp_path, "barq", {"CharB": ["p2"]}, is_default=True)

        copied = _copy_dataset_files(Config.DEFAULT_DATASET, "ds_from_default")
        assert copied == 2

        out_dir = tmp_path / "datasets" / "ds_from_default"
        assert (out_dir / "furtrack" / "CharA" / "p1.jpg").exists()
        assert (out_dir / "barq" / "CharB" / "p2.jpg").exists()

    def test_copy_to_default_dataset(self, tmp_path, tmp_dir):
        """Test copying into default dataset (creates source_images dirs)."""
        src_dir = tmp_path / "datasets" / "ds_f"
        _make_image_files(src_dir, "furtrack", {"CharA": ["p1"]})

        copied = _copy_dataset_files("ds_f", Config.DEFAULT_DATASET)
        assert copied == 1
        assert (tmp_path / "furtrack_images" / "CharA" / "p1.jpg").exists()

    def test_empty_source(self, tmp_path, tmp_dir):
        """Test copying from nonexistent source dir."""
        copied = _copy_dataset_files("nonexistent", "ds_out5")
        assert copied == 0


class TestCombineWithFiles:
    def test_combine_files_only(self, tmp_path, tmp_dir):
        """Test combining datasets that have only image files, no DB."""
        import argparse
        from sam3_pursuit.api.cli import combine_command

        src_a = tmp_path / "datasets" / "files_a"
        src_b = tmp_path / "datasets" / "files_b"
        _make_image_files(src_a, "furtrack", {"CharA": ["p1"]})
        _make_image_files(src_b, "barq", {"CharB": ["p2"]})

        args = argparse.Namespace(datasets=["files_a", "files_b"], output="files_merged")
        combine_command(args)

        out_dir = tmp_path / "datasets" / "files_merged"
        assert (out_dir / "furtrack" / "CharA" / "p1.jpg").exists()
        assert (out_dir / "barq" / "CharB" / "p2.jpg").exists()

    def test_combine_db_and_files(self, tmp_path, tmp_dir):
        """Test combining datasets with both DB and image files."""
        import argparse
        from sam3_pursuit.api.cli import combine_command

        # Create DB datasets
        _make_dataset(tmp_path, "dbfiles_a", [
            ("p1", "CharA", "furtrack", "solid_128"),
        ])
        _make_dataset(tmp_path, "dbfiles_b", [
            ("p2", "CharB", "barq", "solid_128"),
        ])

        # Create image files
        src_a = tmp_path / "datasets" / "dbfiles_a"
        src_b = tmp_path / "datasets" / "dbfiles_b"
        _make_image_files(src_a, "furtrack", {"CharA": ["p1"]})
        _make_image_files(src_b, "barq", {"CharB": ["p2"]})

        args = argparse.Namespace(datasets=["dbfiles_a", "dbfiles_b"], output="dbfiles_merged")
        combine_command(args)

        # Verify DB
        merged_db = Database(str(tmp_path / "dbfiles_merged.db"))
        dets = _get_all_detections(merged_db)
        assert len(dets) == 2
        merged_db.close()

        # Verify files
        out_dir = tmp_path / "datasets" / "dbfiles_merged"
        assert (out_dir / "furtrack" / "CharA" / "p1.jpg").exists()
        assert (out_dir / "barq" / "CharB" / "p2.jpg").exists()


class TestSplitWithFiles:
    def test_split_files_only(self, tmp_path, tmp_dir):
        """Test splitting dataset with only image files, no DB."""
        import argparse
        from sam3_pursuit.api.cli import split_command

        src = tmp_path / "datasets" / "files_src"
        _make_image_files(src, "furtrack", {"CharA": ["p1", "p2"]})
        _make_image_files(src, "barq", {"CharB": ["p3"]})

        args = argparse.Namespace(
            source_dataset="files_src",
            output="files_ft",
            by_source="furtrack",
            by_character=None,
            shards=1,
        )
        split_command(args)

        out_dir = tmp_path / "datasets" / "files_ft"
        assert (out_dir / "furtrack" / "CharA" / "p1.jpg").exists()
        assert (out_dir / "furtrack" / "CharA" / "p2.jpg").exists()
        assert not (out_dir / "barq").exists()

    def test_split_files_by_character(self, tmp_path, tmp_dir):
        """Test splitting files by character name."""
        import argparse
        from sam3_pursuit.api.cli import split_command

        src = tmp_path / "datasets" / "char_src"
        _make_image_files(src, "furtrack", {"CharA": ["p1"], "CharB": ["p2"], "CharC": ["p3"]})

        args = argparse.Namespace(
            source_dataset="char_src",
            output="char_out",
            by_source=None,
            by_character="CharA,CharC",
            shards=1,
        )
        split_command(args)

        out_dir = tmp_path / "datasets" / "char_out"
        assert (out_dir / "furtrack" / "CharA" / "p1.jpg").exists()
        assert not (out_dir / "furtrack" / "CharB").exists()
        assert (out_dir / "furtrack" / "CharC" / "p3.jpg").exists()

    def test_split_source_unchanged(self, tmp_path, tmp_dir):
        """Test that source files are unchanged after split."""
        import argparse
        from sam3_pursuit.api.cli import split_command

        src = tmp_path / "datasets" / "src_intact"
        _make_image_files(src, "furtrack", {"CharA": ["p1", "p2"]})

        args = argparse.Namespace(
            source_dataset="src_intact",
            output="split_out",
            by_source="furtrack",
            by_character=None,
            shards=1,
        )
        split_command(args)

        # Source files still there
        assert (src / "furtrack" / "CharA" / "p1.jpg").exists()
        assert (src / "furtrack" / "CharA" / "p2.jpg").exists()
