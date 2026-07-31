"""Backbone loading for U-ADAPT (Grounding DINO Swin-T primary, OWL-ViT /
YOLO-World-small / YOLO11-small fallbacks)."""

from .backbone_loader import Backbone, load_backbone

__all__ = ["Backbone", "load_backbone"]
