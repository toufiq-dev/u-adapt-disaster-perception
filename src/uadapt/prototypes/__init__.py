"""Prototype construction (text + visual) from cached features."""

from .prototype_builder import (
    TextPrototype,
    VisualPrototype,
    build_text_prototypes,
    build_visual_prototypes,
    reject_outliers,
)

__all__ = [
    "TextPrototype",
    "VisualPrototype",
    "build_text_prototypes",
    "build_visual_prototypes",
    "reject_outliers",
]
