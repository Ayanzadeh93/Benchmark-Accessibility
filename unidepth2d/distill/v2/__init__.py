"""UniDepth2D V2 (CoDeQ) training package. See docs/V2_PLAN.md and its binding amendments.

Modules (import them directly; this package import stays light and does not pull in timm):

* ``distill.v2.encoder``: plain MobileNetV4-Conv encoder (BN/ReLU/Add explicit, block deletion,
  timm key map).
* ``distill.v2.model``: ``build_v2_model(cfg)``, ``count_params``, ``count_macs``, ``V2_VARIANTS``
  (CoDeQ-P/N/S with decoder modes s4/s2/ps/fpn, and the legacy EfficientViT rungs R0-R3).
* ``distill.v2.losses``: ``V2Loss(cfg)`` (corrected L_SI, prox SILog, gradient matching, the R0
  legacy objective, the A7 R1a/R1b split and the Tier-2 optional terms).

The legacy code in ``distill/`` (train.py, losses.py, dataset.py, student.py) is reused, never
modified, by these modules.
"""

__all__ = ["encoder", "model", "losses"]
