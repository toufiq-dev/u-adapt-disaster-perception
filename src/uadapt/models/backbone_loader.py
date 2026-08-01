"""Backbone loader.

Loads a frozen open-vocabulary detector and exposes a uniform ``predict``
interface that returns, per image, the top-k proposals with box features and
per-class text similarities. The primary backbone is Grounding DINO Swin-T;
OWL-ViT and YOLOE26 are cross-backbone ablations; YOLO-World-small and
YOLO11-small are Colab-friendly fallbacks (proposal §5 Phase 1, §7.3).

Guarded imports keep this module importable (and unit-testable) without
torch / transformers / ultralytics installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


def torch_no_grad():
    """Decorator that enables torch.inference_mode() around predict() if torch
    is available (no-op otherwise, keeping unit tests dependency-free)."""
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                import torch
            except ImportError:
                return fn(*args, **kwargs)
            with torch.inference_mode():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


@dataclass
class Proposal:
    """A single candidate detection from the frozen backbone."""

    image_id: str
    bbox: np.ndarray            # (x1, y1, x2, y2)
    score: float                # detector confidence
    class_name: str             # winning open-vocabulary class
    visual_feature: Optional[np.ndarray] = None   # box feature (d,)
    text_similarities: Optional[np.ndarray] = None  # per-class similarities (C,)
    classes: List[str] = field(default_factory=list)  # class list used


class Backbone(Protocol):
    """Uniform frozen-detector interface."""

    name: str

    def predict(self, image: np.ndarray, classes: List[str], **kw: Any) -> List[Proposal]:
        """Run the frozen detector once; return proposals (already top-k limited)."""
        ...


def load_backbone(config: Dict[str, Any], device: str = "cuda") -> Backbone:
    """Load a backbone from a model config dict (see configs/models/*.yaml)."""
    name = config["name"]
    if name == "grounding_dino_swinT":
        return _GroundingDINOBackbone(config, device)
    if name == "owl_vit":
        return _OWLViTBackbone(config, device)
    if name in ("yolo_world_small", "yolo11_small"):
        return _YOLOBackbone(config, device)
    if name == "yolo_e26":
        return _YOLOEBackbone(config, device)
    raise ValueError(f"Unknown backbone: {name}")


class _GroundingDINOBackbone:
    """Grounding DINO Swin-T via huggingface/transformers (Apache-2.0)."""

    name = "grounding_dino_swinT"

    def __init__(self, config: Dict[str, Any], device: str = "cuda") -> None:
        from transformers import (
            GroundingDinoForObjectDetection,
            GroundingDinoProcessor,
        )

        self.checkpoint = config["checkpoint"]
        self.device = device
        self.processor = GroundingDinoProcessor.from_pretrained(self.checkpoint)
        self.model = GroundingDinoForObjectDetection.from_pretrained(self.checkpoint)
        self.model.to(device).eval()
        self.box_threshold = config["inference"].get("box_threshold", 0.25)
        self.text_threshold = config["inference"].get("text_threshold", 0.25)
        self.top_k = config["inference"].get("top_k", 100)

    @torch_no_grad()
    def predict(self, image: np.ndarray, classes: List[str], **kw: Any) -> List[Proposal]:
        texts = [[f"a {c}" for c in classes]]
        inputs = self.processor(images=image, text=texts, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Post-process to boxes + logits, keep top-k by confidence.
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.text_input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.shape[0], image.shape[1])],
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        proposals = [
            Proposal(
                image_id=kw.get("image_id", "img"),
                bbox=boxes[i],
                score=float(scores[i]),
                class_name=str(labels[i]),
            )
            for i in range(len(boxes))
        ]
        return limit_top_k(proposals, self.top_k)

    def __repr__(self) -> str:  # pragma: no cover
        return f"GroundingDINO(Swin-T, {self.checkpoint})"


class _OWLViTBackbone:
    """OWL-ViT via huggingface/transformers (Apache-2.0)."""

    name = "owl_vit"

    def __init__(self, config: Dict[str, Any], device: str = "cuda") -> None:
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.checkpoint = config["checkpoint"]
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(self.checkpoint)
        self.model = OwlViTForObjectDetection.from_pretrained(self.checkpoint)
        self.model.to(device).eval()
        self.score_threshold = config["inference"].get("score_threshold", 0.1)
        self.top_k = config["inference"].get("top_k", 100)

    @torch_no_grad()
    def predict(self, image: np.ndarray, classes: List[str], **kw: Any) -> List[Proposal]:
        import torch

        inputs = self.processor(
            text=classes, images=image, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.score_threshold,
            target_sizes=[(image.shape[0], image.shape[1])],
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        proposals = [
            Proposal(
                image_id=kw.get("image_id", "img"),
                bbox=boxes[i],
                score=float(scores[i]),
                class_name=classes[int(labels[i])],
            )
            for i in range(len(boxes))
        ]
        return limit_top_k(proposals, self.top_k)

    def __repr__(self) -> str:  # pragma: no cover
        return f"OWL-ViT({self.checkpoint})"


class _YOLOEBackbone:
    """YOLOE26 via ultralytics (cross-backbone ablation, proposal §7.3).

    YOLOE supports text, visual, and prompt-free inference modes. The
    checkpoint and AGPL/Apache status are verified in Milestone 1 (issue #1);
    loading mirrors _YOLOBackbone until the text/visual prompt modes are wired.
    """

    def __init__(self, config: Dict[str, Any], device: str = "cuda") -> None:
        from ultralytics import YOLO

        self.name = config["name"]
        self.checkpoint = config["checkpoint"]
        if not self.checkpoint or self.checkpoint == "TBD":
            raise ValueError(
                "YOLOE26 checkpoint is TBD (verify in Milestone 1, issue #1); "
                "update configs/models/yolo_e26.yaml before use."
            )
        self.model = YOLO(self.checkpoint)
        self.conf = config["inference"].get("conf", 0.05)
        self.top_k = config["inference"].get("top_k", 100)

    @torch_no_grad()
    def predict(self, image: np.ndarray, classes: List[str], **kw: Any) -> List[Proposal]:
        results = self.model.predict(
            source=image, conf=self.conf, classes=None, verbose=False
        )[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        proposals = [
            Proposal(
                image_id=kw.get("image_id", "img"),
                bbox=boxes[i],
                score=float(scores[i]),
                class_name="object",  # class-agnostic until prompt modes are wired
            )
            for i in range(len(boxes))
        ]
        return limit_top_k(proposals, self.top_k)

    def __repr__(self) -> str:  # pragma: no cover
        return f"YOLOE26({self.checkpoint})"


class _YOLOBackbone:
    """YOLO-World-small / YOLO11-small via ultralytics (AGPL-3.0)."""

    def __init__(self, config: Dict[str, Any], device: str = "cuda") -> None:
        from ultralytics import YOLO

        self.name = config["name"]
        self.checkpoint = config["checkpoint"]
        self.model = YOLO(self.checkpoint)
        self.conf = config["inference"].get("conf", 0.05)
        self.top_k = config["inference"].get("top_k", 100)

    @torch_no_grad()
    def predict(self, image: np.ndarray, classes: List[str], **kw: Any) -> List[Proposal]:
        results = self.model.predict(
            source=image, conf=self.conf, classes=None, verbose=False
        )[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        proposals = [
            Proposal(
                image_id=kw.get("image_id", "img"),
                bbox=boxes[i],
                score=float(scores[i]),
                class_name="object",  # YOLO variants report class-agnostic here
            )
            for i in range(len(boxes))
        ]
        return limit_top_k(proposals, self.top_k)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.name}({self.checkpoint})"


def limit_top_k(proposals: List[Proposal], k: int) -> List[Proposal]:
    """Pre-registered top-k proposal limiting (k=100 primary, k=300 ablation).

    Sorts by detector confidence (descending) and keeps the top ``k``. This is
    the single choke point for the top-k requirement — the frozen backbone
    runs once per image and only the top-k proposals proceed to feature
    extraction and caching.
    """
    if k is None or k <= 0:
        return proposals
    if len(proposals) <= k:
        return sorted(proposals, key=lambda p: p.score, reverse=True)
    return sorted(proposals, key=lambda p: p.score, reverse=True)[:k]
