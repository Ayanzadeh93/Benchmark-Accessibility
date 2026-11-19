"""Benchmark-replication harness for surpassing UniDepthV2 on its own zero-shot MMDE benchmark.

Ports UniDepthV2's EXACT metrics (unidepth/utils/evaluation_depth.py) with a pure-PyTorch chamfer,
adds per-dataset loaders in the paper's protocol, a reference wrapper on the official V2 checkpoints,
and the eval driver. See docs/EXPERIMENTS.md and the plan for the surpass campaign (T1/T2/T3).
"""
