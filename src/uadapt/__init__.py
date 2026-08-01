"""U-ADAPT: Uncertainty-Aware Post-Hoc Adaptation of Open-Vocabulary Detectors.

Post-hoc, uncertainty-gated fusion for open-vocabulary object detectors in
aerial disaster imagery. Frozen backbone, cached features, training-free
analytic gating (Mode A) with a lightweight calibrated variant (Mode B),
optionally initialized from COCO/LVIS-pretrained gate weights (ablation; the
former Mode C, per proposal §5.4.3).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
