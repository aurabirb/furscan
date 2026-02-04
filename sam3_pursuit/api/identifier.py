import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

from sam3_pursuit.config import Config
from sam3_pursuit.models.preprocessor import IsolationConfig
from sam3_pursuit.models.segmentor import SAM3FursuitSegmentor, FullImageSegmentor, SegmentationResult
from sam3_pursuit.models.embedder import FursuitEmbedder
from sam3_pursuit.models.preprocessor import BackgroundIsolator
from sam3_pursuit.storage.database import Database, Detection
from sam3_pursuit.storage.vector_index import VectorIndex
from sam3_pursuit.storage.mask_storage import MaskStorage


@dataclass
class ProcessingResult:
    segmentation: SegmentationResult
    embedding: np.ndarray
    isolated_crop: Optional[Image.Image] = None
    segmentor_model: str = "unknown"
    segmentor_concept: Optional[str] = None
    mask_reused: bool = False


@dataclass
class IdentificationResult:
    character_name: Optional[str]
    confidence: float
    distance: float
    post_id: str
    bbox: tuple[int, int, int, int]
    segmentor_model: str = "unknown"
    source: Optional[str] = None


@dataclass
class SegmentResults:
    segment_index: int
    segment_bbox: tuple[int, int, int, int]
    segment_confidence: float
    matches: list[IdentificationResult]

class FursuitIdentifier:
    def __init__(
        self,
        db_path: str = Config.DB_PATH,
        index_path: str = Config.INDEX_PATH,
        device: Optional[str] = None,
        isolation_config: Optional[IsolationConfig] = None,
        segmentor_model_name: Optional[str] = "",
        segmentor_concept: Optional[str] = "",
    ):
        self.device = device or Config.get_device()
        self.segmentor_device = Config.get_segmentor_device()
        if segmentor_model_name == Config.SAM3_MODEL:
            self.segmentor = SAM3FursuitSegmentor(device=self.segmentor_device, concept=segmentor_concept)
        else:
            self.segmentor = FullImageSegmentor()
        self.segmentor_model_name = self.segmentor.model_name
        self.db = Database(db_path)
        self.index = VectorIndex(index_path)
        self._sync_index_and_db()
        self.mask_storage = MaskStorage()
        self.segmentor_concept = segmentor_concept
        self.embedder_model_name = Config.DINOV2_MODEL
        self.embedder = FursuitEmbedder(device=self.device, model_name=self.embedder_model_name)
        self.isolator = BackgroundIsolator(isolation_config)


    def _sync_index_and_db(self):
        """Ensure FAISS index and database are in sync (crash recovery)."""
        max_valid_id = self.index.size - 1
        max_db_id = self.db.get_next_embedding_id() - 1
        if max_db_id > max_valid_id:
            deleted = self.db.delete_orphaned_detections(max_valid_id)
            if deleted > 0:
                print(f"Sync: deleted {deleted} orphaned detections (embedding_id > {max_valid_id})")

    def _short_embedder_name(self) -> str:
        emb = self.embedder_model_name
        if "dinov2-base" in emb:
            return "dv2b"
        elif "dinov2-large" in emb:
            return "dv2l"
        elif "dinov2-giant" in emb:
            return "dv2g"
        return emb.split("/")[-1][:8]

    def _build_preprocessing_info(self) -> str:
        """Build fingerprint for segmented crops."""
        iso = self.isolator.config
        mode_map = {"solid": "s", "blur": "b", "none": "n"}
        parts = [
            "v2",
            f"seg:{self.segmentor_model_name}",
            f"con:{(self.segmentor_concept or '').replace('|', '.')}",
            f"bg:{mode_map.get(iso.mode, 'n')}",
        ]
        if iso.mode == "solid":
            r, g, b = iso.background_color
            parts.append(f"bgc:{r:02x}{g:02x}{b:02x}")
        elif iso.mode == "blur":
            parts.append(f"br:{iso.blur_radius}")
        parts += [f"emb:{self._short_embedder_name()}", f"tsz:{Config.TARGET_IMAGE_SIZE}"]
        return "|".join(parts)

    def _build_full_preprocessing_info(self) -> str:
        return f"v2|seg:full|emb:{self._short_embedder_name()}|tsz:{Config.TARGET_IMAGE_SIZE}"

    def identify(
        self,
        image: Image.Image,
        top_k: int = Config.DEFAULT_TOP_K,
        save_crops: bool = False,
        crop_prefix: str = "query",
    ) -> list[SegmentResults]:
        if self.index.size == 0:
            print("Warning: Index is empty, no matches possible")
            return []

        proc_results = self.process(image)
        segment_results = []
        for i, proc_result in enumerate(proc_results):
            if save_crops and proc_result.isolated_crop:
                self._save_debug_crop(proc_result.isolated_crop, f"{crop_prefix}_{i}", source="search")
            matches = self._search_embedding(proc_result.embedding, top_k)
            segment_results.append(SegmentResults(
                segment_index=i,
                segment_bbox=proc_result.segmentation.bbox,
                segment_confidence=proc_result.segmentation.confidence,
                matches=matches,
            ))
        return segment_results

    def _save_debug_crop(
        self,
        image: Image.Image,
        name: str,
        source: Optional[str] = None,
    ) -> None:
        """Save crop image for debugging (optional, use --save-crops)."""
        crops_dir = Path(Config.CROPS_INGEST_DIR) / (source or "unknown")
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crops_dir / f"{name}.jpg"
        image.convert("RGB").save(crop_path, quality=90)

    def _search_embedding(self, embedding: np.ndarray, top_k: int) -> list[IdentificationResult]:
        distances, indices = self.index.search(embedding, top_k * 2)
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            detection = self.db.get_detection_by_embedding_id(int(idx))
            if detection is None:
                continue
            confidence = max(0.0, 1.0 - distance / 2.0)
            results.append(IdentificationResult(
                character_name=detection.character_name,
                confidence=confidence,
                distance=float(distance),
                post_id=detection.post_id,
                bbox=(detection.bbox_x, detection.bbox_y,
                      detection.bbox_width, detection.bbox_height),
                segmentor_model=detection.segmentor_model,
                source=detection.source,
            ))
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]

    def _resize_to_patch_multiple(self, image: Image.Image, target_size: int = 630) -> Image.Image:
        w, h = image.size
        if w >= h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        new_w = max(Config.PATCH_SIZE, (new_w // Config.PATCH_SIZE) * Config.PATCH_SIZE)
        new_h = max(Config.PATCH_SIZE, (new_h // Config.PATCH_SIZE) * Config.PATCH_SIZE)
        # print(f"Resizing image from ({w}, {h}) to ({new_w}, {new_h})")
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _load_segments_for_post(self, post_id: str, source: str, model: str, concept: str, image: Image.Image) -> list[SegmentationResult]:
        if self.mask_storage.has_no_segments_marker(post_id, source, model, concept, image):
            return FullImageSegmentor().segment(image)
        masks = self.mask_storage.load_masks_for_post(post_id, source, model, concept)
        segmentations = [
            SegmentationResult.from_mask(image, mask, segmentor=self.segmentor_model_name) for mask in masks
        ]
        return segmentations

    def _save_segments_for_post(self, post_id: str, source: str, model: str, concept: str, segmentations: list[SegmentationResult]) -> None:
        masks = [seg.mask for seg in segmentations if seg.mask is not None]
        if masks:
            self.mask_storage.save_masks_for_post(post_id, source, self.segmentor_model_name, self.segmentor_concept, masks)
        else:
            self.mask_storage.save_no_segments_marker(post_id, source, self.segmentor_model_name, self.segmentor_concept)

    def process(self, image: Image.Image) -> list[ProcessingResult]:
        proc_results = []
        segmentations = self.segmentor.segment(image)
        for seg in segmentations:
            isolated = self.isolator.isolate(seg.crop, seg.crop_mask)
            isolated_crop = self._resize_to_patch_multiple(isolated)
            embedding = self.embedder.embed(isolated_crop)
            proc_results.append(ProcessingResult(
                segmentation=seg,
                embedding=embedding,
                isolated_crop=isolated_crop,
                segmentor_model=seg.segmentor,
                segmentor_concept=self.segmentor_concept,
                mask_reused=False,
            ))
        return proc_results


    def add_images(
        self,
        character_names: list[str],
        image_paths: list[str],
        save_crops: bool = False,
        source: Optional[str] = None,
        uploaded_by: Optional[str] = None,
        add_full_image: bool = True,
        batch_size: int = 100,
        skip_non_fursuit: bool = False,
        classify_threshold: float = Config.DEFAULT_CLASSIFY_THRESHOLD,
        post_ids: Optional[list[str]] = None,
    ) -> int:
        
        assert len(character_names) == len(image_paths)
        character_names = [name.lower().replace(" ", "_") for name in character_names]

        if post_ids is not None:
            assert len(post_ids) == len(image_paths)
        else:
            post_ids = [self._extract_post_id(p) for p in image_paths]

        seg_preproc = self._build_preprocessing_info()
        full_preproc = self._build_full_preprocessing_info()
        posts_need_full = self.db.get_posts_needing_update(post_ids, full_preproc, source)
        posts_need_seg = self.db.get_posts_needing_update(post_ids, seg_preproc, source)
        posts_to_process = posts_need_seg if not add_full_image else posts_need_full | posts_need_seg

        print(f"Processing {len(posts_to_process)} posts ({len(posts_need_full)} need full, {len(posts_need_seg)} need seg)")
        filtered_indices = [i for i, pid in enumerate(post_ids) if pid in posts_to_process]
        if not filtered_indices:
            return 0

        classifier = None
        if skip_non_fursuit:
            from sam3_pursuit.models.classifier import ImageClassifier
            classifier = ImageClassifier(device=self.device)
            print(f"Using classifier to skip non-fursuit images (threshold: {classify_threshold})")

        total = len(filtered_indices)
        added_count = 0
        skipped_count = 0
        masks_reused_count = 0
        masks_generated_count = 0
        pending_embeddings: list[np.ndarray] = []
        pending_detections: list[Detection] = []

        def new_detection(post_id, character_name, bbox, confidence, segmentor_model, filename, preproc_info):
            return Detection(
                id=None, post_id=post_id, character_name=character_name, embedding_id=-1,
                bbox_x=bbox[0], bbox_y=bbox[1], bbox_width=bbox[2], bbox_height=bbox[3],
                confidence=confidence, segmentor_model=segmentor_model,
                source=source, uploaded_by=uploaded_by, source_filename=filename,
                preprocessing_info=preproc_info,
            )

        def flush_batch():
            """Commit DB then save FAISS. If interrupted, _sync_index_and_db cleans up orphans."""
            nonlocal added_count
            if not pending_embeddings:
                return
            print(f"  Saving batch of {len(pending_detections)} embeddings to database and index...")
            start_id = self.index.add(np.vstack(pending_embeddings).astype(np.float32))
            for i, detection in enumerate(pending_detections):
                detection.embedding_id = start_id + i
            self.db.add_detections_batch(pending_detections)
            self.index.save(backup=True)
            added_count += len(pending_detections)
            print(f"  Batch saved: {len(pending_detections)} embeddings (index: {self.index.size})")
            pending_embeddings.clear()
            pending_detections.clear()

        for i, idx in enumerate(filtered_indices):
            character_name = character_names[idx]
            img_path = image_paths[idx]
            post_id = self._extract_post_id(img_path)
            filename = os.path.basename(img_path)

            try:
                image = self._load_image(img_path)
            except Exception as e:
                print(f"[{i+1}/{total}] Failed to load {filename}: {e}")
                continue

            if classifier and not classifier.is_fursuit(image, threshold=classify_threshold):
                skipped_count += 1
                print(f"[{i+1}/{total}] Skipped {filename} (not fursuit)")
                continue

            try:
                segmentations = self._load_segments_for_post(post_id, source, self.segmentor_model_name, self.segmentor_concept, image)
            except Exception as e:
                print(f"[{i + 1}/{total}] Failed to load segments for {filename}: {e}")
                segmentations = []

            if segmentations:
                mask_reused = True
                masks_reused_count += len(segmentations)
            else:
                mask_reused = False
                segmentations = self.segmentor.segment(image) # Long operation
                masks_generated_count += len(segmentations)
                try:
                    self._save_segments_for_post(post_id, source, self.segmentor_model_name, self.segmentor_concept, segmentations)
                except Exception as e:
                    print(f"[{i+1}/{total}] Failed to save segments for {filename}: {e}")

            if not segmentations:
                print(f"[{i+1}/{total}] No segments found for {filename}, adding full image as fallback")
                segmentations = FullImageSegmentor().segment(image)

            proc_results = []
            for seg in segmentations:
                isolated = self.isolator.isolate(seg.crop, seg.crop_mask)
                isolated_crop = self._resize_to_patch_multiple(isolated)
                embedding = self.embedder.embed(isolated_crop)
                proc_results.append(ProcessingResult(
                    segmentation=seg,
                    embedding=embedding,
                    isolated_crop=isolated_crop,
                    segmentor_model=seg.segmentor,
                    segmentor_concept=self.segmentor_concept,
                    mask_reused=mask_reused,
                )) # TODO: can share processing result over network for distributed ingestion

            for j, proc_result in enumerate(proc_results):
                if save_crops and proc_result.isolated_crop:
                    seg_name = f"{post_id}_seg_{j}"
                    self._save_debug_crop(proc_result.isolated_crop, seg_name, source=source)
                pending_embeddings.append(proc_result.embedding.reshape(1, -1))
                pending_detections.append(new_detection(
                    post_id, character_name, proc_result.segmentation.bbox,
                    proc_result.segmentation.confidence, proc_result.segmentor_model,
                    filename, seg_preproc))

            mask_msg = " (masks reused)" if mask_reused else ""
            print(f"[{i+1}/{total}] {character_name}: {len(proc_results)} segments{mask_msg}")

            if len(pending_embeddings) >= batch_size:
                flush_batch()

        flush_batch()

        skip_msg = f", {skipped_count} skipped (not fursuit)" if skipped_count else ""
        mask_msg = f", masks: {masks_reused_count} reused/{masks_generated_count} generated" if masks_reused_count or masks_generated_count else ""
        print(f"Ingestion complete: {added_count} embeddings added{skip_msg}{mask_msg} (index: {self.index.size})")
        return added_count

        # print(f"\nMask check: {ok} ok, {regenerated} regenerated, {failed} failed (total: {total})")
        # return {"ok": ok, "regenerated": regenerated, "failed": failed, "total": total}

    def _load_image(self, img_path: str) -> Image.Image:
        img_path = str(img_path)
        if img_path.startswith(('http://', 'https://')):
            response = requests.get(img_path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            if not Path(img_path).exists():
                raise FileNotFoundError()
            img = Image.open(img_path)
        if img is None:
            raise ValueError()
        return img

    def _extract_post_id(self, img_path: str) -> str:
        basename = os.path.basename(img_path)
        return os.path.splitext(basename)[0]

    def get_stats(self) -> dict:
        db_stats = self.db.get_stats()
        db_stats["index_size"] = self.index.size
        return db_stats
