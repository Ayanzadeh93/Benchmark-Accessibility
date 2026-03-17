#!/usr/bin/env python3
"""Paper-ready benchmark runner for the VQA pipeline.

This script focuses on:
- Video-level MOS correlation + 3-class metrics (if labels provided)
- Runtime/throughput stats
- Ablations (baseline vs VLM routing, calibration toggles)
- Exporting prediction tables (video/segment/frame level)

Input requirements:
- A MOS mapping file (CSV or JSON) with at least: video_id, rel_path, mos
  Optionally: quality_class (poor/acceptable/good)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiments.metrics import (
    accuracy_from_confusion,
    correlation_metrics,
    confusion_matrix,
    f1_macro_from_confusion,
)
from experiments.schemas import VideoLabel


logger = logging.getLogger(__name__)


def _read_labels(path: Path) -> List[VideoLabel]:
    """Load labels from CSV or JSON."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "videos" in data:
            data = data["videos"]
        if not isinstance(data, list):
            raise ValueError("JSON mos-file must be a list of objects or {videos:[...]}")
        return [VideoLabel.model_validate(x) for x in data]

    if path.suffix.lower() == ".csv":
        out: List[VideoLabel] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # allow common column aliases
                normalized = dict(row)
                if "rel_path" not in normalized:
                    normalized["rel_path"] = normalized.get("path") or normalized.get("video_path") or ""
                if "video_id" not in normalized:
                    normalized["video_id"] = normalized.get("id") or Path(normalized["rel_path"]).stem
                if "mos" in normalized:
                    normalized["mos"] = float(normalized["mos"])
                out.append(VideoLabel.model_validate(normalized))
        return out

    raise ValueError("mos-file must be .csv or .json")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_scatter(out_path: Path, y_true: List[float], y_pred: List[float], title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, s=25, alpha=0.75)
    plt.xlabel("MOS (ground truth)")
    plt.ylabel("Predicted quality score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_confusion(out_path: Path, cm: np.ndarray, classes: List[str], title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _mode_configs(mode_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Map mode name -> overrides applied to QualityConfig / pipeline."""
    cfgs: Dict[str, Dict[str, Any]] = {}
    for m in mode_names:
        m2 = m.strip().lower()
        if not m2:
            continue
        if m2 == "baseline":
            cfgs[m] = {"enable_vlm": False}
        elif m2 in {"vlm_no_routing", "vlm-norouting"}:
            cfgs[m] = {"enable_vlm": True, "router_alpha": 0.0}
        elif m2 in {"vlm_routing", "vlm-routing"}:
            cfgs[m] = {"enable_vlm": True}  # use default router_alpha
        elif m2 in {"calib_off", "calibration_off"}:
            cfgs[m] = {"enable_vlm": False, "score_stretch": 1.0, "temporal_penalty": 0.0}
        else:
            # user-defined mode name (no overrides)
            cfgs[m] = {}
    return cfgs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark VQA pipeline on a labeled dataset")
    p.add_argument("--dataset-root", type=str, required=True, help="Root directory containing videos")
    p.add_argument("--mos-file", type=str, required=True, help="CSV/JSON mapping video_id, rel_path, mos, [quality_class]")
    p.add_argument("--coco-json", type=str, default=None, help="Optional COCO-style JSON (frames/videos). Used for dataset bookkeeping and future frame-level eval.")
    p.add_argument("--output-dir", type=str, default="./benchmark_results", help="Output directory")
    p.add_argument("--modes", type=str, default="baseline,vlm_routing", help="Comma-separated modes (baseline,vlm_no_routing,vlm_routing,calib_off)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of videos (debug)")
    p.add_argument("--resume", action="store_true", help="Skip videos that already have a saved report in the output folder")

    # Pipeline knobs
    p.add_argument("--segment-duration", type=int, default=6)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--vlm-model", type=str, default="qwen", choices=["qwen", "gpt4o"])
    p.add_argument("--vlm-device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    # Efficiency defaults for benchmarking (override keyfram_analysis defaults)
    p.add_argument("--no-visualizations", action="store_true", help="Disable dashboards/overview")
    p.add_argument("--no-heatmaps", action="store_true", help="Disable heatmaps")
    p.add_argument("--no-all-frames", action="store_true", help="Save keyframes only")
    p.add_argument("--nr-max-side", type=int, default=512, help="Downsample size for NR-IQA proxy computation")
    p.add_argument("--clip-max-frames", type=int, default=32, help="Limit frames embedded by CLIP per segment")
    p.add_argument("--save-frame-level", action="store_true", help="Export frame_level.csv (can be large)")
    p.add_argument("--max-frame-rows", type=int, default=200000, help="Cap total frame rows written (safety)")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dataset_root = Path(args.dataset_root)
    labels = _read_labels(Path(args.mos_file))
    if args.limit and args.limit > 0:
        labels = labels[: args.limit]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    coco_summary: Optional[Dict[str, Any]] = None
    if args.coco_json:
        coco_path = Path(args.coco_json)
        try:
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
            coco_summary = {
                "path": str(coco_path),
                "num_images": len(coco.get("images", []) or []),
                "num_annotations": len(coco.get("annotations", []) or []),
                "num_categories": len(coco.get("categories", []) or []),
                "has_videos_field": bool(coco.get("videos")),
                "num_videos_field": len(coco.get("videos", []) or []) if isinstance(coco.get("videos"), list) else 0,
            }
            (out_root / "coco_summary.json").write_text(json.dumps(coco_summary, indent=2), encoding="utf-8")
        except Exception as e:
            coco_summary = {"path": str(coco_path), "error": str(e)}
            (out_root / "coco_summary.json").write_text(json.dumps(coco_summary, indent=2), encoding="utf-8")

    mode_names = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    modes = _mode_configs(mode_names)

    # Import pipeline lazily (heavy imports)
    from keyfram_analysis import QualityConfig, VideoQualityPipeline

    all_mode_metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": str(dataset_root),
        "mos_file": str(Path(args.mos_file)),
        "coco_json": coco_summary,
        "num_videos": len(labels),
        "modes": {},
    }

    for mode, overrides in modes.items():
        logger.info(f"=== MODE: {mode} ===")
        mode_dir = out_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        outputs_base = mode_dir / "pipeline_outputs"
        outputs_base.mkdir(parents=True, exist_ok=True)

        # Build config (apply benchmarking defaults)
        cfg = QualityConfig(segment_duration=int(args.segment_duration), fps_target=int(args.fps))
        cfg.save_visualizations = not bool(args.no_visualizations)
        cfg.save_heatmaps = not bool(args.no_heatmaps)
        cfg.save_all_frames = not bool(args.no_all_frames)
        cfg.nr_quality_max_side = int(args.nr_max_side)
        cfg.clip_max_frames = int(args.clip_max_frames)

        # Apply ablation overrides
        if "score_stretch" in overrides:
            cfg.score_stretch = float(overrides["score_stretch"])
        if "temporal_penalty" in overrides:
            cfg.temporal_penalty = float(overrides["temporal_penalty"])
        if "router_alpha" in overrides:
            cfg.router_alpha = float(overrides["router_alpha"])

        enable_vlm = bool(overrides.get("enable_vlm", False))

        pipeline = VideoQualityPipeline(
            cfg,
            enable_vlm=enable_vlm,
            vlm_model=str(args.vlm_model),
            vlm_api_key=None,
            vlm_device=str(args.vlm_device),
        )

        video_rows: List[Dict[str, Any]] = []
        segment_rows: List[Dict[str, Any]] = []
        frame_rows: List[Dict[str, Any]] = []
        frame_row_budget = int(args.max_frame_rows)

        y_true: List[float] = []
        y_pred: List[float] = []
        gt_class: List[str] = []
        pred_class: List[str] = []
        runtime_s: List[float] = []

        for lab in labels:
            video_path = dataset_root / lab.rel_path
            if not video_path.exists():
                video_rows.append(
                    {
                        "mode": mode,
                        "video_id": lab.video_id,
                        "rel_path": lab.rel_path,
                        "error": "video_not_found",
                    }
                )
                continue

            # Resume: if a report JSON exists in mode_dir, skip processing
            # We look for a matching quality report under pipeline_outputs/**/Quality_reports/*_quality_report.json
            if args.resume:
                existing = list(outputs_base.rglob(f"*{lab.video_id}*quality_report.json"))
                if existing:
                    logger.info(f"Skipping (resume): {lab.video_id}")
                    continue

            try:
                video_report, segment_reports, out_dir = pipeline.process_video(str(video_path), str(outputs_base))
            except Exception as e:
                video_rows.append(
                    {
                        "mode": mode,
                        "video_id": lab.video_id,
                        "rel_path": lab.rel_path,
                        "error": str(e),
                    }
                )
                continue

            pred_score = float(video_report.get("overall_quality_score", 0.0))
            pred_qclass = str(video_report.get("quality_class", "acceptable"))
            y_true.append(float(lab.mos))
            y_pred.append(pred_score)
            runtime_s.append(float(video_report.get("processing_time", 0.0)))

            if lab.quality_class is not None:
                gt_class.append(str(lab.quality_class))
                pred_class.append(pred_qclass)

            video_rows.append(
                {
                    "mode": mode,
                    "video_id": lab.video_id,
                    "rel_path": lab.rel_path,
                    "mos": float(lab.mos),
                    "gt_quality_class": str(lab.quality_class) if lab.quality_class else "",
                    "pred_quality_score": pred_score,
                    "pred_quality_class": pred_qclass,
                    "processing_time_s": float(video_report.get("processing_time", 0.0)),
                    "output_dir": str(out_dir),
                    "enable_vlm": enable_vlm,
                    "vlm_model": str(args.vlm_model) if enable_vlm else "",
                    "score_stretch": float(cfg.score_stretch),
                    "temporal_penalty": float(cfg.temporal_penalty),
                    "router_alpha": float(cfg.router_alpha),
                }
            )

            # Segment-level exports
            for s in (video_report.get("segments") or []):
                segment_rows.append(
                    {
                        "mode": mode,
                        "video_id": lab.video_id,
                        "segment_id": s.get("segment_id"),
                        "start_time": s.get("start_time"),
                        "end_time": s.get("end_time"),
                        "mean_quality": s.get("mean_quality"),
                        "quality_class": s.get("quality_class"),
                        "is_flagged": s.get("is_flagged"),
                        "flag_reasons": "|".join(s.get("flag_reasons") or []),
                        "vlm_enabled": s.get("vlm_enabled"),
                        "vlm_model": s.get("vlm_model"),
                        "vlm_has_artifacts": s.get("vlm_has_artifacts"),
                        "vlm_artifact_type": s.get("vlm_artifact_type"),
                        "routing_rationale": s.get("routing_rationale"),
                    }
                )

                # Frame-level exports (optional, capped)
                if args.save_frame_level and frame_row_budget > 0:
                    fm = s.get("frame_metrics") or []
                    for m in fm:
                        if frame_row_budget <= 0:
                            break
                        frame_rows.append(
                            {
                                "mode": mode,
                                "video_id": lab.video_id,
                                "segment_id": s.get("segment_id"),
                                "frame_idx": m.get("frame_idx"),
                                "timestamp": m.get("timestamp"),
                                "overall_quality": m.get("overall_quality"),
                                "artifact_probability": m.get("artifact_probability"),
                                "quality_class": m.get("quality_class"),
                            }
                        )
                        frame_row_budget -= 1

        # Save tables
        _write_csv(mode_dir / "video_level.csv", video_rows)
        _write_csv(mode_dir / "segment_level.csv", segment_rows)
        if args.save_frame_level:
            _write_csv(mode_dir / "frame_level.csv", frame_rows)

        # Metrics + plots
        corr = correlation_metrics(y_true, y_pred)
        mode_metrics: Dict[str, Any] = {
            "mode": mode,
            "num_videos_evaluated": len(y_pred),
            "runtime": {
                "mean_s": float(np.mean(runtime_s)) if runtime_s else 0.0,
                "median_s": float(np.median(runtime_s)) if runtime_s else 0.0,
                "total_s": float(np.sum(runtime_s)) if runtime_s else 0.0,
            },
            "correlation": asdict(corr) if corr else None,
        }

        if gt_class:
            classes = ["poor", "acceptable", "good"]
            cm = confusion_matrix(gt_class, pred_class, classes)
            mode_metrics["classification"] = {
                "classes": classes,
                "confusion_matrix": cm.tolist(),
                "accuracy": accuracy_from_confusion(cm),
                "macro_f1": f1_macro_from_confusion(cm),
            }
            _plot_confusion(mode_dir / "plots" / "confusion_matrix.png", cm, classes, f"{mode}: quality class")

        _plot_scatter(mode_dir / "plots" / "mos_scatter.png", y_true, y_pred, f"{mode}: MOS vs predicted score")

        with (mode_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(mode_metrics, f, indent=2)

        all_mode_metrics["modes"][mode] = mode_metrics

    with (out_root / "metrics_all_modes.json").open("w", encoding="utf-8") as f:
        json.dump(all_mode_metrics, f, indent=2)

    print("[OK] Benchmark complete")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()

