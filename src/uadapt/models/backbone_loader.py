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


def resolve_device(device: str = "cuda") -> str:
    """Return a usable torch device string, falling back from CUDA to MPS/CPU.

    The model configs default to ``device: cuda`` (Colab/GPU runs), but the
    pipeline should still run on a Mac (MPS) or CPU-only box without editing
    configs: if CUDA is unavailable, prefer MPS when present, else CPU.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def load_backbone(config: Dict[str, Any], device: str = "cuda") -> Backbone:
    """Load a backbone from a model config dict (see configs/models/*.yaml)."""
    device = resolve_device(device)
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
        import torch

        texts = [[f"a {c}" for c in classes]]
        inputs = self.processor(images=image, text=texts, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Per-class text alignment from the query->text-token logits.
        # ``logits`` is (B, num_queries, max_text_len=256) padded; the prompt
        # is "[CLS] a <c1> . a <c2> . [SEP]", so each class occupies a
        # contiguous token span that we max-pool over to get one similarity
        # per (query, class). Padding positions are -inf (masked), so the max
        # ignores them automatically.
        spans = _class_token_spans(
            self.processor.tokenizer, inputs["input_ids"][0].tolist(), classes
        )
        probs = torch.sigmoid(outputs.logits[0])  # (Q, max_text_len)
        class_sims = torch.stack(
            [probs[:, s:e].max(dim=1).values for s, e in spans], dim=1  # (Q, C)
        )
        # Detector confidence = best alignment over ANY prompt token (matches
        # the official post_process ``scores = max over last dim`` semantics).
        scores = probs.max(dim=1).values  # (Q,)

        # Winning class per query + keep mask (box + text thresholds).
        best_cls = class_sims.argmax(dim=1)
        best_sim = class_sims.gather(1, best_cls.unsqueeze(1)).squeeze(1)
        keep = (scores > self.box_threshold) & (best_sim > self.text_threshold)
        q = torch.nonzero(keep).squeeze(1)

        # Boxes: pred_boxes are center-format normalized -> xyxy pixels.
        cx, cy, w, h = outputs.pred_boxes[0].unbind(-1)  # each (Q,)
        img_h, img_w = float(image.shape[0]), float(image.shape[1])
        x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        x2, y2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h

        # Box visual features: the encoder's vision tokens (B, H*W, D). These
        # are the detector-internal features the class logits are computed from.
        vision = outputs.encoder_last_hidden_state_vision[0]  # (H*W, D)
        # The vision grid covers the PADDED model input (H_in//stride x W_in//stride),
        # whose aspect follows ``pixel_values`` — not the raw image. Derive the
        # exact grid dims so RoI pooling is not misaligned on non-square inputs.
        h_in, w_in = inputs.pixel_values.shape[-2:]
        n_tokens = vision.shape[0]
        grid_w = int(round((n_tokens * w_in / h_in) ** 0.5))
        grid_h = (n_tokens + grid_w - 1) // grid_w

        proposals: List[Proposal] = []
        for i in q.tolist():
            cls_idx = int(best_cls[i].item())
            box = (
                float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])
            )  # floats -> host memory (MPS-safe)
            # RoI-pool the vision feature map: mean of tokens inside the box
            # (coarse but consistent detector-internal features).
            vis = _roi_mean_feature(
                vision.cpu().numpy(), box, img_w, img_h, grid_h, grid_w
            )
            proposals.append(
                Proposal(
                    image_id=kw.get("image_id", "img"),
                    bbox=np.asarray(box, dtype=np.float64),
                    score=float(best_sim[i].item()),
                    class_name=classes[cls_idx],
                    visual_feature=vis,
                    text_similarities=class_sims[i].cpu().numpy().astype(np.float64),
                    classes=list(classes),
                )
            )
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


def _class_token_spans(tokenizer, input_ids, classes: List[str]):
    """Return the [start, end) token span of each class prompt.

    The Grounding DINO prompt is ``"[CLS] a <c1> . a <c2> . [SEP]"``. For each
    class we locate its tokenized name preceded by the ``a`` article and
    return the span covering ``a <name>`` (the ``.`` separators and special
    tokens are excluded). Padding positions are ignored (masked to -inf).
    """
    toks = tokenizer.convert_ids_to_tokens(input_ids)
    spans = []
    for c in classes:
        name_toks = tokenizer.tokenize(c)
        found = None
        for i in range(len(toks) - len(name_toks) + 1):
            if toks[i : i + len(name_toks)] == name_toks:
                found = i
                break
        if found is None and len(name_toks) == 1:
            # fallback: single-token search on lowercase (only valid for
            # single-word classes; multiword names cannot match one token)
            for i, t in enumerate(toks):
                if t == c.lower():
                    found = i
                    break
        if found is None:
            raise ValueError(f"could not locate class {c!r} in prompt tokens {toks}")
        # include the preceding 'a' article (tokens[found-1] == 'a')
        start = found - 1 if found > 0 and toks[found - 1] == "a" else found
        spans.append((start, found + len(name_toks)))
    return spans


def _roi_mean_feature(
    vision: np.ndarray,
    box,
    img_w: float,
    img_h: float,
    grid_h: Optional[int] = None,
    grid_w: Optional[int] = None,
) -> np.ndarray:
    """Mean of encoder vision tokens inside an (x1,y1,x2,y2) box.

    ``vision`` is (H*W, D) where H*W is the vision encoder's spatial grid
    (sequence of patch tokens in row-major order). The box is in image-pixel
    coordinates and is normalized by ``img_w``/``img_h``; ``grid_h``/``grid_w``
    are the exact grid dims of the vision tokens (derived from the padded
    model input). When omitted, the grid aspect is approximated from the
    token count alone (square-root heuristic — used only by callers without
    access to the input dims). Returns the mean token feature (D,) or a zero
    vector when the box is degenerate.
    """
    vision = np.asarray(vision, dtype=np.float64)
    if vision.ndim != 2 or vision.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    n_tokens = vision.shape[0]
    if grid_h is not None and grid_w is not None:
        H, W = int(grid_h), int(grid_w)
    else:
        side = int(round(n_tokens ** 0.5))
        if side * side != n_tokens:
            side += 1
        H, W = side, (n_tokens + side - 1) // side

    x1, y1, x2, y2 = box
    x1, x2 = min(max(x1, 0.0), img_w), min(max(x2, 0.0), img_w)
    y1, y2 = min(max(y1, 0.0), img_h), min(max(y2, 0.0), img_h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((vision.shape[1],), dtype=np.float64)

    c1 = int(x1 / img_w * W)
    c2 = max(int(x2 / img_w * W) - 1, c1)
    r1 = int(y1 / img_h * H)
    r2 = max(int(y2 / img_h * H) - 1, r1)
    c2 = min(c2, W - 1)
    r2 = min(r2, H - 1)

    idx = []
    for r in range(r1, r2 + 1):
        for cc in range(c1, c2 + 1):
            tok = r * W + cc
            if tok < n_tokens:
                idx.append(tok)
    if not idx:
        return np.zeros((vision.shape[1],), dtype=np.float64)
    return vision[idx].mean(axis=0)


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
