#!/usr/bin/env python3
"""
========================================================================
 SCIENTIFIC VIDEO QUALITY ASSESSMENT PIPELINE
========================================================================
Based on Q-Router: Agentic Video Quality Assessment with Expert Model Routing

Features:
- Multi-metric Quality Assessment (Sharpness, Noise, Brightness, Contrast, NR-IQA proxies)
- Uniform frame sampling per segment (deterministic)
- LPIPS Artifact Localization with Heatmaps (optional)
- CLIP-based Keyframe Selection
- VLM-assisted keyframe analysis (Qwen/GPT-4o) and lightweight expert routing (optional)
- Comprehensive Visualizations per segment (optional)
- Automatic Quality Flagging with recommendations
- Stores keyframes by default; can optionally store extracted frames
- Output resolution support for saved images (e.g., 1080p)
- 3-tier quality classification (poor/good/acceptable) with wider spread

Author: Advanced VQA System
For Nature / NeurIPS-style benchmarking and analysis
========================================================================
"""

# =======================================================================
# CELL 2: IMPORTS AND SETUP
# =======================================================================

import os
import re
import cv2
import json
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from tqdm.auto import tqdm
from scipy import stats
from scipy.signal import convolve2d
from scipy.special import gamma
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import lpips
from pytubefix import YouTube
from pytubefix.cli import on_progress
from keyframe_extraction import CLIPKeyframeSelector, VLMKeyframeAnalyzer

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Check device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =======================================================================
# DATA CLASSES
# =======================================================================

@dataclass
class QualityConfig:
    """Configuration for quality assessment pipeline."""
    segment_duration: int = 6
    fps_target: int = 1
    # WIDER thresholds for better spread (0.3-0.7 range)
    quality_threshold_poor: float = 0.40      # Below this = poor
    quality_threshold_good: float = 0.60      # Above this = good
    # Between 0.40-0.60 = acceptable
    # Score calibration (keeps code simple, but fixes "all scores ~0.5-0.6")
    # Final per-frame quality is a contrast-stretch around `score_center`:
    #   q = clip(score_center + (base_quality - score_center) * score_stretch, 0, 1)
    score_center: float = 0.50
    score_stretch: float = 1.60
    # Segment aggregation penalty for unstable segments:
    #   Q_seg = clip(mean(q_t) - temporal_penalty * std(q_t), 0, 1)
    temporal_penalty: float = 0.12
    # If VLM is enabled, blend routed score into segment mean:
    #   Q_final = (1-router_alpha)*Q_seg + router_alpha*Q_routed
    router_alpha: float = 0.35
    # Efficiency knobs (keep pipeline fast for large batches)
    # Downsample for NR-IQA proxy computation (CPU-heavy at high resolutions)
    nr_quality_max_side: int = 512
    # Limit CLIP embeddings per segment (CLIP is GPU-heavy if you embed every frame)
    clip_max_frames: int = 32
    tau_high: float = 0.65
    tau_low: float = 0.50
    min_clip_length: int = 8
    padding_frames: int = 4
    clip_model: str = "ViT-B/32"
    lpips_net: str = "alex"
    jpeg_quality: int = 95
    save_heatmaps: bool = True
    save_visualizations: bool = True
    output_resolution: Tuple[int, int] = (1920, 1080)  # 1080p
    video_types: List[str] = field(default_factory=lambda: ['general', 'walking_tour', 'static', 'dynamic', 'indoor', 'outdoor'])
    # NEW: Save ALL frames regardless of quality
    save_all_frames: bool = True


@dataclass
class FrameQualityMetrics:
    """Quality metrics for a single frame."""
    frame_idx: int
    timestamp: float
    laplacian_variance: float = 0.0
    tenengrad: float = 0.0
    normalized_sharpness: float = 0.0
    noise_estimate: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    edge_density: float = 0.0
    gradient_kurtosis: float = 0.0
    brisque_score: float = 0.0
    niqe_score: float = 0.0
    motion_residual: float = 0.0
    histogram_uniformity: float = 0.0
    technical_quality: float = 0.0
    overall_quality: float = 0.0
    artifact_probability: float = 0.0
    video_type: str = "general"
    quality_class: str = "acceptable"  # Added per-frame quality class


@dataclass
class SegmentQualityReport:
    """Quality report for a video segment."""
    segment_id: int
    segment_path: str
    start_time: float
    end_time: float
    duration: float
    num_frames: int = 0
    num_extracted_frames: int = 0
    num_saved_frames: int = 0  # NEW: Track saved frames
    mean_quality: float = 0.0
    min_quality: float = 0.0
    max_quality: float = 0.0
    std_quality: float = 0.0
    mean_sharpness: float = 0.0
    mean_brightness: float = 0.0
    mean_contrast: float = 0.0
    mean_brisque: float = 0.0
    mean_niqe: float = 0.0
    temporal_consistency: float = 0.0
    artifact_frames: List[int] = field(default_factory=list)
    artifact_severity: float = 0.0
    quality_class: str = "acceptable"
    is_flagged: bool = False
    flag_reasons: List[str] = field(default_factory=list)
    keyframe_idx: int = 0
    keyframe_path: str = ""
    frame_metrics: List[Dict] = field(default_factory=list)
    video_type: str = "general"
    frame_quality_distribution: Dict[str, int] = field(default_factory=dict)  # NEW: Track quality distribution
    # VLM + routing (lightweight Q-Router-style interpretability)
    vlm_enabled: bool = False
    vlm_model: str = ""
    vlm_has_artifacts: bool = False
    vlm_artifact_type: str = "none"
    routed_mean_quality: float = 0.0
    routing_weights: Dict[str, float] = field(default_factory=dict)
    routing_rationale: str = ""


# =======================================================================
# NR-IQA PROXY EXTRACTORS (lightweight, not standard BRISQUE/NIQE)
# =======================================================================

class BRISQUEProxyFeatureExtractor:
    """
    Lightweight BRISQUE-like proxy (NOT the standard BRISQUE implementation).
    Kept intentionally simple for large-batch runs; treat as a proxy feature.
    """
    
    def __init__(self):
        self.scales = [1, 2]
        
    def _compute_mscn_coefficients(self, img: np.ndarray, window_size: int = 7) -> np.ndarray:
        img = img.astype(np.float64)
        kernel_size = window_size
        sigma = window_size / 6.0
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel /= kernel.sum()
        mu = convolve2d(img, kernel, mode='same', boundary='symm')
        mu_sq = convolve2d(img**2, kernel, mode='same', boundary='symm')
        sigma_map = np.sqrt(np.maximum(mu_sq - mu**2, 0))
        mscn = (img - mu) / (sigma_map + 1.0)
        return mscn
    
    def _fit_ggd(self, data: np.ndarray) -> Tuple[float, float]:
        data = data.flatten()
        data = data[np.isfinite(data)]
        if len(data) < 10:
            return 2.0, 1.0
        mean_abs = np.mean(np.abs(data))
        variance = np.var(data)
        if mean_abs < 1e-10 or variance < 1e-10:
            return 2.0, 1.0
        rho = variance / (mean_abs ** 2 + 1e-10)
        alpha_range = np.arange(0.2, 10.0, 0.01)
        def gamma_ratio(alpha):
            return (gamma(3/alpha) * gamma(1/alpha)) / (gamma(2/alpha) ** 2)
        ratios = np.array([gamma_ratio(a) for a in alpha_range])
        idx = np.argmin(np.abs(ratios - rho))
        return float(alpha_range[idx]), float(variance)
    
    def _fit_aggd(self, data: np.ndarray) -> Tuple[float, float, float]:
        data = data.flatten()
        data = data[np.isfinite(data)]
        if len(data) < 10:
            return 2.0, 1.0, 1.0
        left = data[data < 0]
        right = data[data > 0]
        sigma_left = np.sqrt(np.mean(left ** 2)) if len(left) > 0 else 1.0
        sigma_right = np.sqrt(np.mean(right ** 2)) if len(right) > 0 else 1.0
        alpha, _ = self._fit_ggd(data)
        return float(alpha), float(sigma_left), float(sigma_right)
    
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        features = []
        for scale in self.scales:
            if scale > 1:
                h, w = gray.shape[:2]
                gray_scaled = cv2.resize(gray, (w // scale, h // scale))
            else:
                gray_scaled = gray
            mscn = self._compute_mscn_coefficients(gray_scaled)
            alpha, variance = self._fit_ggd(mscn)
            features.extend([alpha, variance])
            products = {
                'h': mscn[:, :-1] * mscn[:, 1:],
                'v': mscn[:-1, :] * mscn[1:, :],
                'd1': mscn[:-1, :-1] * mscn[1:, 1:],
                'd2': mscn[:-1, 1:] * mscn[1:, :-1]
            }
            for prod in products.values():
                alpha, sigma_l, sigma_r = self._fit_aggd(prod)
                eta = (sigma_r - sigma_l) * gamma(2/alpha) / gamma(1/alpha) if alpha > 0 else 0
                features.extend([alpha, eta, sigma_l, sigma_r])
        return np.array(features)
    
    def compute_score(self, img: np.ndarray) -> float:
        features = self.extract_features(img)
        alpha_values = features[::4][:4]
        mean_alpha = np.mean(alpha_values)
        score = abs(mean_alpha - 2.5) * 20
        return float(np.clip(score, 0, 100))


class NIQEProxyFeatureExtractor:
    """
    Lightweight NIQE-like proxy (NOT the standard NIQE implementation).
    Kept intentionally simple for large-batch runs; treat as a proxy feature.
    """
    
    def __init__(self, patch_size: int = 96, stride: int = 48):
        self.patch_size = patch_size
        self.stride = stride
        self.brisque = BRISQUEProxyFeatureExtractor()
        
    def _extract_patches(self, img: np.ndarray) -> List[np.ndarray]:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        h, w = gray.shape
        patches = []
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = gray[i:i+self.patch_size, j:j+self.patch_size]
                if np.std(patch) > 10:
                    patches.append(patch)
        return patches
    
    def compute_score(self, img: np.ndarray) -> float:
        patches = self._extract_patches(img)
        if len(patches) == 0:
            return 50.0
        patch_features = []
        for patch in patches[:20]:
            features = self.brisque.extract_features(patch)
            patch_features.append(features)
        patch_features = np.array(patch_features)
        mean_features = np.mean(patch_features, axis=0)
        score = np.mean(np.abs(mean_features[:4] - 2.5)) * 15
        return float(np.clip(score, 0, 100))


# =======================================================================
# VIDEO TYPE DETECTOR (unchanged)
# =======================================================================

class VideoTypeDetector:
    """Detect video type based on content analysis."""
    
    def __init__(self):
        self.type_weights = {
            'walking_tour': {'motion': 0.7, 'edge_density': 0.3},
            'static': {'motion': 0.1, 'edge_density': 0.9},
            'dynamic': {'motion': 0.9, 'edge_density': 0.5},
            'indoor': {'brightness': 0.4, 'contrast': 0.6},
            'outdoor': {'brightness': 0.7, 'contrast': 0.8},
            'general': {}
        }
    
    def detect_type(self, frame_metrics: List[Dict]) -> str:
        """Detect video type from frame metrics."""
        if not frame_metrics:
            return "general"
        
        avg_motion = np.mean([m.get('motion_residual', 0) for m in frame_metrics])
        avg_edge = np.mean([m.get('edge_density', 0) for m in frame_metrics])
        avg_brightness = np.mean([m.get('brightness', 0.5) for m in frame_metrics])
        avg_contrast = np.mean([m.get('contrast', 0.5) for m in frame_metrics])
        
        scores = {}
        
        if avg_motion > 15 and 0.1 < avg_edge < 0.3:
            scores['walking_tour'] = 0.8
        
        if avg_motion < 5 and avg_edge > 0.2:
            scores['static'] = 0.7
        
        if avg_motion > 25:
            scores['dynamic'] = 0.9
        
        if avg_brightness < 0.4 and 0.3 < avg_contrast < 0.6:
            scores['indoor'] = 0.6
        
        if avg_brightness > 0.5 and avg_contrast > 0.5:
            scores['outdoor'] = 0.7
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "general"


# =======================================================================
# QUALITY METRICS CALCULATOR (UPDATED - WIDER SPREAD)
# =======================================================================

class QualityMetricsCalculator:
    """Calculates various quality metrics for video frames."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.brisque = BRISQUEProxyFeatureExtractor()
        self.niqe = NIQEProxyFeatureExtractor()
        # LPIPS model is large; only initialize if/when heatmaps are requested.
        self.lpips_fn = None
        self.lpips_net = config.lpips_net
        self.sharpness_max = 5000.0
        self.type_detector = VideoTypeDetector()
    
    def compute_laplacian_variance(self, gray: np.ndarray) -> float:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())
    
    def compute_tenengrad(self, gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx**2 + gy**2))
    
    def compute_gradient_kurtosis(self, gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        flat = gradient_mag.flatten()
        if np.std(flat) < 1e-6:
            return 0.0
        return float(stats.kurtosis(flat, fisher=True))
    
    def compute_edge_density(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 100, 200)
        return float(np.sum(edges > 0) / edges.size)
    
    def estimate_noise(self, gray: np.ndarray) -> float:
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
        filtered = convolve2d(gray.astype(np.float32), kernel, mode='same', boundary='symm')
        return float(np.median(np.abs(filtered)) / 0.6745)
    
    def compute_histogram_metrics(self, gray: np.ndarray) -> Tuple[float, float, float]:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        brightness = float(np.mean(gray)) / 255.0
        contrast = float(np.std(gray)) / 128.0
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10))
        uniformity = float(entropy / 8.0)
        return brightness, contrast, uniformity
    
    def compute_motion_residual(self, frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
        if prev_frame is None:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
        diff = np.abs(gray.astype(float) - gray_prev.astype(float))
        return float(np.mean(diff))
    
    def compute_lpips_heatmap(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        # Lazy-load LPIPS model (only if heatmaps are enabled and requested)
        if self.lpips_fn is None:
            try:
                self.lpips_fn = lpips.LPIPS(net=self.lpips_net, spatial=True, verbose=False).to(DEVICE)
                self.lpips_fn.eval()
            except Exception:
                self.lpips_fn = None

        def prep_frame(f: np.ndarray) -> torch.Tensor:
            if len(f.shape) == 2:
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
            else:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (256, 256))
            f = f.astype(np.float32) / 255.0
            f = (f - 0.5) / 0.5
            f = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            return f

        def _normalize_heatmap(h: np.ndarray) -> np.ndarray:
            h = h.astype(np.float32)
            denom = float(h.max() - h.min())
            if denom < 1e-8:
                return np.zeros_like(h)
            return (h - h.min()) / denom

        t1, t2 = prep_frame(frame1), prep_frame(frame2)
        with torch.no_grad():
            if self.lpips_fn is not None:
                try:
                    dist = self.lpips_fn(t1, t2)
                    if dist.ndim == 4 and dist.shape[-1] > 1:
                        heatmap = dist.squeeze().cpu().numpy()
                    else:
                        raise ValueError("LPIPS returned non-spatial output")
                except Exception:
                    diff = torch.abs(t1 - t2).mean(dim=1, keepdim=True)
                    diff = F.interpolate(diff, size=(64, 64), mode='bilinear', align_corners=False)
                    heatmap = diff.squeeze().cpu().numpy()
            else:
                diff = torch.abs(t1 - t2).mean(dim=1, keepdim=True)
                diff = F.interpolate(diff, size=(64, 64), mode='bilinear', align_corners=False)
                heatmap = diff.squeeze().cpu().numpy()
        return _normalize_heatmap(heatmap)
    
    def _classify_frame_quality(self, score: float) -> str:
        """Classify individual frame quality."""
        if score >= self.config.quality_threshold_good:  # >= 0.60
            return 'good'
        elif score >= self.config.quality_threshold_poor:  # >= 0.40
            return 'acceptable'
        else:  # < 0.40
            return 'poor'
    
    def compute_frame_metrics(self, frame: np.ndarray, frame_idx: int,
                             timestamp: float, prev_frame: Optional[np.ndarray] = None,
                             video_type: str = "general") -> FrameQualityMetrics:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        laplacian_var = self.compute_laplacian_variance(gray)
        tenengrad = self.compute_tenengrad(gray)
        grad_kurtosis = self.compute_gradient_kurtosis(gray)
        edge_density = self.compute_edge_density(gray)
        noise = self.estimate_noise(gray)
        brightness, contrast, hist_uniformity = self.compute_histogram_metrics(gray)
        motion = self.compute_motion_residual(frame, prev_frame)
        # NR-IQA proxies are CPU-heavy at high resolutions; downsample for speed while keeping signal.
        frame_nr = frame
        try:
            max_side = int(getattr(self.config, "nr_quality_max_side", 0) or 0)
        except Exception:
            max_side = 0
        if max_side > 0:
            h, w = frame.shape[:2]
            cur_max = max(h, w)
            if cur_max > max_side:
                scale = max_side / float(cur_max)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                frame_nr = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        brisque_score = self.brisque.compute_score(frame_nr)
        niqe_score = self.niqe.compute_score(frame_nr)
        
        norm_sharpness = min(laplacian_var / self.sharpness_max, 1.0)
        
        # ENHANCED: More sensitive quality calculation with WIDER spread
        # Adjusted to spread values across 0.3-0.7 range (wider than before)
        base_quality = (
            0.28 * norm_sharpness +           # Sharpness weight
            0.22 * (1.0 - min(noise / 50.0, 1.0)) +  # Noise (inverse)
            0.18 * (1.0 - abs(brightness - 0.5)) +  # Brightness (prefer mid-range)
            0.16 * min(contrast, 1.0) +       # Contrast
            0.10 * (1.0 - brisque_score / 100.0) +  # BRISQUE-proxy (inverse)
            0.06 * (1.0 - niqe_score / 100.0)       # NIQE-proxy (inverse)
        )
        
        # Apply video type adjustments (more pronounced)
        type_adjustments = {
            'walking_tour': 0.03,  # Boost for motion content
            'static': 0.02,        # Boost for sharp static scenes
            'dynamic': -0.02,      # Penalty for motion blur
            'indoor': 0.01,        # Slight boost
            'outdoor': 0.03,       # Boost for outdoor lighting
            'general': 0.0
        }
        base_quality += type_adjustments.get(video_type, 0.0)
        
        # Score calibration: expand dynamic range (avoids scores collapsing near ~0.5-0.6)
        base_quality = float(np.clip(base_quality, 0.0, 1.0))
        tech_quality = self.config.score_center + (base_quality - self.config.score_center) * self.config.score_stretch
        tech_quality = float(np.clip(tech_quality, 0.0, 1.0))
        
        artifact_prob = 1.0 - tech_quality
        quality_class = self._classify_frame_quality(tech_quality)
        
        return FrameQualityMetrics(
            frame_idx=frame_idx, timestamp=timestamp,
            laplacian_variance=laplacian_var, tenengrad=tenengrad,
            normalized_sharpness=norm_sharpness, noise_estimate=noise,
            brightness=brightness, contrast=contrast,
            edge_density=edge_density, gradient_kurtosis=grad_kurtosis,
            brisque_score=brisque_score, niqe_score=niqe_score,
            motion_residual=motion, histogram_uniformity=hist_uniformity,
            technical_quality=tech_quality, overall_quality=tech_quality,
            artifact_probability=artifact_prob, video_type=video_type,
            quality_class=quality_class
        )


# =======================================================================
# CLIP KEYFRAME SELECTOR - moved to keyframe_extraction/
# =======================================================================
# Imported from keyframe_extraction.clip_selector


# =======================================================================
# QUALITY VISUALIZER (UPDATED for wider thresholds)
# =======================================================================

class QualityVisualizer:
    """Generate comprehensive quality visualizations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "Visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        # 3-tier colormap: poor (red), acceptable (yellow), good (green)
        self.quality_cmap = LinearSegmentedColormap.from_list(
            'quality', ['#e74c3c', '#f39c12', '#2ecc71']
        )
    
    def create_segment_dashboard(self, segment_report: SegmentQualityReport,
                                 frames: List[np.ndarray], keyframe: np.ndarray,
                                 video_title: str) -> str:
        """Create comprehensive dashboard for a segment."""
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle(f'📊 Quality Assessment: {video_title}\nSegment {segment_report.segment_id} '
                    f'({segment_report.start_time:.1f}s - {segment_report.end_time:.1f}s)',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Row 1: Keyframe and Summary
        ax_keyframe = fig.add_subplot(gs[0, :2])
        keyframe_rgb = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)
        ax_keyframe.imshow(keyframe_rgb)
        ax_keyframe.set_title('🔑 Selected Keyframe', fontsize=12, fontweight='bold')
        ax_keyframe.axis('off')
        
        quality_colors = {
            'good': '#2ecc71',
            'acceptable': '#f39c12',
            'poor': '#e74c3c'
        }
        badge_color = quality_colors.get(segment_report.quality_class, '#95a5a6')
        ax_keyframe.add_patch(Rectangle((10, 10), 150, 40, facecolor=badge_color, alpha=0.9))
        ax_keyframe.text(85, 30, segment_report.quality_class.upper(),
                        ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        ax_summary = fig.add_subplot(gs[0, 2:])
        ax_summary.axis('off')
        summary_text = [
            f"📈 Overall Quality: {segment_report.mean_quality:.3f}",
            f"📊 Quality Range: {segment_report.min_quality:.3f} - {segment_report.max_quality:.3f}",
            f"🔍 Sharpness: {segment_report.mean_sharpness:.2f}",
            f"☀️ Brightness: {segment_report.mean_brightness:.2f}",
            f"🎨 Contrast: {segment_report.mean_contrast:.2f}",
            f"📉 BRISQUE-proxy: {segment_report.mean_brisque:.1f}",
            f"📉 NIQE-proxy: {segment_report.mean_niqe:.1f}",
            f"⏱️ Temporal: {segment_report.temporal_consistency:.2f}",
            f"⚠️ Artifacts: {len(segment_report.artifact_frames)}/{segment_report.num_extracted_frames}",
            f"🎬 Video Type: {segment_report.video_type}",
            f"💾 Frames Saved: {segment_report.num_saved_frames}/{segment_report.num_extracted_frames}",
        ]
        if segment_report.frame_quality_distribution:
            dist_text = " | ".join([f"{k}: {v}" for k, v in segment_report.frame_quality_distribution.items()])
            summary_text.append(f"📊 Quality Dist: {dist_text}")
        if segment_report.is_flagged:
            summary_text.append(f"\n🚩 FLAGGED: {', '.join(segment_report.flag_reasons)}")
        ax_summary.text(0.1, 0.95, '\n'.join(summary_text), transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_summary.set_title('📋 Quality Summary', fontsize=12, fontweight='bold')
        
        # Row 2: Frame samples
        num_samples = min(4, len(frames))
        sample_indices = np.linspace(0, len(frames)-1, num_samples, dtype=int) if num_samples > 1 else [0]
        for i, idx in enumerate(sample_indices):
            ax = fig.add_subplot(gs[1, i])
            frame_rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)
            if idx < len(segment_report.frame_metrics):
                q = segment_report.frame_metrics[idx].get('overall_quality', 0)
                q_class = segment_report.frame_metrics[idx].get('quality_class', 'acceptable')
                color = quality_colors.get(q_class, '#95a5a6')
                ax.set_title(f'Frame {idx}: Q={q:.3f} ({q_class})', fontsize=10, color=color)
            ax.axis('off')
        
        # Row 3: Quality timeline
        ax_timeline = fig.add_subplot(gs[2, :])
        if segment_report.frame_metrics:
            frame_indices = [m.get('frame_idx', i) for i, m in enumerate(segment_report.frame_metrics)]
            qualities = [m.get('overall_quality', 0) for m in segment_report.frame_metrics]
            sharpness = [m.get('normalized_sharpness', 0) for m in segment_report.frame_metrics]
            brisque = [1 - m.get('brisque_score', 50)/100 for m in segment_report.frame_metrics]
            
            ax_timeline.plot(frame_indices, qualities, 'g-', linewidth=2, label='Overall Quality', marker='o', markersize=4)
            ax_timeline.plot(frame_indices, sharpness, 'b--', linewidth=1.5, label='Sharpness', alpha=0.7)
            ax_timeline.plot(frame_indices, brisque, 'r:', linewidth=1.5, label='BRISQUE-proxy (inv)', alpha=0.7)
            for af in segment_report.artifact_frames:
                if af in frame_indices:
                    ax_timeline.axvline(x=af, color='red', alpha=0.3, linestyle='--')
            ax_timeline.axhline(y=segment_report.mean_quality, color='green', linestyle=':', alpha=0.5)
            # Updated thresholds: 0.40 and 0.60
            ax_timeline.axhline(y=0.40, color='red', linestyle='--', alpha=0.3, label='Poor threshold (0.40)')
            ax_timeline.axhline(y=0.60, color='green', linestyle='--', alpha=0.3, label='Good threshold (0.60)')
            ax_timeline.fill_between(frame_indices, 0, 0.40, alpha=0.1, color='red')
            ax_timeline.set_xlabel('Frame Index')
            ax_timeline.set_ylabel('Score')
            ax_timeline.set_ylim(0.25, 0.75)  # Wider range: 0.3-0.7
            ax_timeline.legend(loc='upper right', fontsize=9)
            ax_timeline.set_title('📈 Quality Over Time', fontsize=12, fontweight='bold')
            ax_timeline.grid(True, alpha=0.3)
        
        # Row 4: Distributions
        if segment_report.frame_metrics:
            qualities = [m.get('overall_quality', 0) for m in segment_report.frame_metrics]
            
            ax_dist1 = fig.add_subplot(gs[3, 0])
            ax_dist1.hist(qualities, bins=15, color='green', alpha=0.7, edgecolor='black')
            ax_dist1.axvline(x=np.mean(qualities), color='red', linestyle='--')
            ax_dist1.axvline(x=0.40, color='red', linestyle=':', alpha=0.5, label='Poor')
            ax_dist1.axvline(x=0.60, color='green', linestyle=':', alpha=0.5, label='Good')
            ax_dist1.set_xlabel('Quality')
            ax_dist1.set_title('Quality Distribution', fontsize=10)
            ax_dist1.legend(fontsize=8)
            
            ax_dist2 = fig.add_subplot(gs[3, 1])
            sharpness_vals = [m.get('normalized_sharpness', 0) for m in segment_report.frame_metrics]
            ax_dist2.hist(sharpness_vals, bins=15, color='blue', alpha=0.7, edgecolor='black')
            ax_dist2.set_xlabel('Sharpness')
            ax_dist2.set_title('Sharpness Distribution', fontsize=10)
            
            ax_scatter = fig.add_subplot(gs[3, 2])
            brightness_vals = [m.get('brightness', 0.5) for m in segment_report.frame_metrics]
            contrast_vals = [m.get('contrast', 0.5) for m in segment_report.frame_metrics]
            colors = [self.quality_cmap(q) for q in qualities]
            ax_scatter.scatter(brightness_vals, contrast_vals, c=colors, s=50, alpha=0.7, edgecolors='black')
            ax_scatter.set_xlabel('Brightness')
            ax_scatter.set_ylabel('Contrast')
            ax_scatter.set_title('Brightness vs Contrast', fontsize=10)
            
            ax_gauge = fig.add_subplot(gs[3, 3], projection='polar')
            mean_q = segment_report.mean_quality
            theta = np.linspace(0, np.pi, 100)
            # Updated for wider thresholds
            ax_gauge.fill_between(theta, 0, 1, where=(theta < np.pi/3), alpha=0.3, color='red', label='Poor')
            ax_gauge.fill_between(theta, 0, 1, where=(theta >= np.pi/3) & (theta < 2*np.pi/3), alpha=0.3, color='yellow', label='Acceptable')
            ax_gauge.fill_between(theta, 0, 1, where=(theta >= 2*np.pi/3), alpha=0.3, color='green', label='Good')
            # Map 0.3-0.7 to 0-π
            needle_angle = np.pi * ((mean_q - 0.3) / 0.4)
            ax_gauge.annotate('', xy=(needle_angle, 0.85), xytext=(needle_angle, 0),
                            arrowprops=dict(arrowstyle='->', color='black', lw=3))
            ax_gauge.set_ylim(0, 1)
            ax_gauge.set_theta_zero_location('W')
            ax_gauge.set_theta_direction(-1)
            ax_gauge.set_thetamin(0)
            ax_gauge.set_thetamax(180)
            ax_gauge.set_title(f'Quality: {mean_q:.3f}', fontsize=10, fontweight='bold')
            ax_gauge.set_yticklabels([])
            ax_gauge.set_xticklabels(['Poor', '', 'Acceptable', '', 'Good'])
        
        output_path = self.viz_dir / f"segment_{segment_report.segment_id:03d}_dashboard.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return str(output_path)
    
    def create_video_overview(self, video_report: Dict, segment_reports: List[SegmentQualityReport],
                             keyframes: List[np.ndarray]) -> str:
        """Create overview visualization for entire video."""
        num_segments = len(segment_reports)
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle(f'🎬 Video Overview: {video_report.get("video_title", "Unknown")}\n'
                    f'Duration: {video_report.get("duration", 0):.1f}s | '
                    f'Segments: {num_segments} | '
                    f'Quality: {video_report.get("overall_quality_score", 0):.3f}',
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Top keyframes
        num_show = min(4, len(keyframes))
        sorted_indices = sorted(range(num_segments), key=lambda i: segment_reports[i].mean_quality, reverse=True)[:num_show]
        for i, idx in enumerate(sorted_indices):
            ax = fig.add_subplot(gs[0, i])
            if idx < len(keyframes):
                kf_rgb = cv2.cvtColor(keyframes[idx], cv2.COLOR_BGR2RGB)
                ax.imshow(kf_rgb)
            ax.set_title(f'Seg {idx}: Q={segment_reports[idx].mean_quality:.3f}', fontsize=10)
            ax.axis('off')
        
        # Quality timeline
        ax_timeline = fig.add_subplot(gs[1, :])
        segment_ids = [s.segment_id for s in segment_reports]
        qualities = [s.mean_quality for s in segment_reports]
        colors = [self.quality_cmap(q) for q in qualities]
        ax_timeline.bar(segment_ids, qualities, color=colors, edgecolor='black', alpha=0.8)
        for i, sr in enumerate(segment_reports):
            if sr.is_flagged:
                ax_timeline.annotate('🚩', (i, qualities[i] + 0.02), ha='center', fontsize=12)
        # Updated thresholds
        ax_timeline.axhline(y=0.40, color='red', linestyle='--', alpha=0.5, label='Poor (0.40)')
        ax_timeline.axhline(y=0.60, color='green', linestyle='--', alpha=0.5, label='Good (0.60)')
        ax_timeline.axhline(y=np.mean(qualities), color='blue', linestyle=':', alpha=0.7)
        ax_timeline.set_xlabel('Segment ID')
        ax_timeline.set_ylabel('Quality')
        ax_timeline.set_ylim(0.25, 0.75)
        ax_timeline.legend(loc='upper right')
        ax_timeline.set_title('📊 Segment Quality Scores', fontsize=12, fontweight='bold')
        ax_timeline.grid(True, alpha=0.3, axis='y')
        
        # Summary stats
        ax_pie = fig.add_subplot(gs[2, 0])
        class_counts = defaultdict(int)
        for sr in segment_reports:
            class_counts[sr.quality_class] += 1
        class_colors = {
            'good': '#2ecc71',
            'acceptable': '#f39c12',
            'poor': '#e74c3c'
        }
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        colors_pie = [class_colors.get(l, '#95a5a6') for l in labels]
        ax_pie.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%', startangle=90)
        ax_pie.set_title('Quality Classes', fontsize=10, fontweight='bold')
        
        # Metrics bar
        ax_metrics = fig.add_subplot(gs[2, 1])
        metrics = ['Sharpness', 'Brightness', 'Contrast', 'BRISQUE-proxy', 'NIQE-proxy', 'Temporal']
        values = [
            np.mean([s.mean_sharpness for s in segment_reports]),
            np.mean([s.mean_brightness for s in segment_reports]),
            np.mean([s.mean_contrast for s in segment_reports]),
            1 - np.mean([s.mean_brisque for s in segment_reports]) / 100,
            1 - np.mean([s.mean_niqe for s in segment_reports]) / 100,
            np.mean([s.temporal_consistency for s in segment_reports])
        ]
        ax_metrics.barh(metrics, values, color='steelblue', alpha=0.7, edgecolor='black')
        ax_metrics.set_xlim(0, 1)
        ax_metrics.set_title('Average Metrics', fontsize=10, fontweight='bold')
        ax_metrics.grid(True, alpha=0.3, axis='x')
        
        # Flagged list
        ax_flags = fig.add_subplot(gs[2, 2:])
        ax_flags.axis('off')
        flagged = [s for s in segment_reports if s.is_flagged]
        if flagged:
            flag_text = "🚩 Flagged Segments (ALL frames/keyframes stored):\n\n"
            for s in flagged[:5]:
                flag_text += f"  Segment {s.segment_id}: {', '.join(s.flag_reasons[:2])}\n"
            if len(flagged) > 5:
                flag_text += f"  ... and {len(flagged)-5} more\n"
        else:
            flag_text = "✅ No segments flagged!\n\nAll segments meet quality standards."
        ax_flags.text(0.1, 0.9, flag_text, transform=ax_flags.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax_flags.set_title('📋 Flagged Segments', fontsize=10, fontweight='bold')
        
        output_path = self.viz_dir / "video_overview.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return str(output_path)
    
    def create_artifact_heatmap(self, frame: np.ndarray, heatmap: np.ndarray,
                               segment_id: int, frame_idx: int) -> str:
        """Create artifact heatmap overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[0].imshow(frame_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        im = axes[1].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Artifact Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(frame_rgb, 0.6, heatmap_colored, 0.4, 0)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        fig.suptitle(f'Segment {segment_id} | Frame {frame_idx} | Severity: {np.mean(heatmap):.3f}',
                    fontsize=12, fontweight='bold')
        
        output_path = self.viz_dir / f"artifact_seg{segment_id:03d}_frame{frame_idx:04d}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return str(output_path)


# =======================================================================
# MAIN PIPELINE (UPDATED - SAVE ALL FRAMES)
# =======================================================================

class VideoQualityPipeline:
    """Complete Video Quality Assessment Pipeline."""
    
    def __init__(
        self,
        config: QualityConfig = None,
        enable_vlm: bool = False,
        vlm_model: str = "qwen",
        vlm_api_key: str = None,
        vlm_device: str = "auto"
    ):
        self.config = config or QualityConfig()
        print(f"Using device: {DEVICE}")
        if DEVICE.type == "cuda":
            try:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
        self.metrics_calculator = QualityMetricsCalculator(self.config)
        self.keyframe_selector = CLIPKeyframeSelector(self.config.clip_model, device=DEVICE)
        self.visualizer = None
        self.type_detector = VideoTypeDetector()
        
        # VLM integration (optional)
        self.vlm_analyzer = None
        if enable_vlm:
            try:
                from config import get_openai_api_key
                api_key = vlm_api_key or get_openai_api_key()
                # Determine device for VLM (Qwen) separately from global DEVICE.
                if vlm_device == "auto":
                    chosen_vlm_device = "cuda" if DEVICE.type == "cuda" else "cpu"
                else:
                    chosen_vlm_device = vlm_device
                    if chosen_vlm_device == "cuda" and DEVICE.type != "cuda":
                        chosen_vlm_device = "cpu"
                self.vlm_analyzer = VLMKeyframeAnalyzer(
                    model_type=vlm_model,
                    api_key=api_key,
                    device=chosen_vlm_device
                )
                if self.vlm_analyzer.enabled:
                    print("VLM analyzer enabled")
            except Exception as e:
                print(f"VLM initialization skipped: {e}")
        
        print("Pipeline initialized!")
        print(f"   Quality thresholds: Poor < {self.config.quality_threshold_poor}, "
              f"Good >= {self.config.quality_threshold_good}")
        print(f"   Save all frames: {self.config.save_all_frames}")
    
    def _sanitize_filename(self, name: str) -> str:
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name[:100]
    
    def _resize_to_output_resolution(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to `config.output_resolution` while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        target_w, target_h = self.config.output_resolution
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        if new_w != target_w or new_h != target_h:
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            resized = cv2.copyMakeBorder(resized, pad_h, target_h - new_h - pad_h,
                                        pad_w, target_w - new_w - pad_w,
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        return resized
    
    def _download_youtube(self, url: str, output_dir: Path) -> Tuple[str, str, str]:
        print("Downloading from YouTube...")
        yt = YouTube(url, on_progress_callback=on_progress)
        video_id = yt.video_id
        title = self._sanitize_filename(yt.title)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
        output_path = output_dir / f"{video_id}_{title}.mp4"
        stream.download(output_path=str(output_dir), filename=output_path.name)
        print(f"Downloaded: {title}")
        return str(output_path), video_id, yt.title
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _segment_video_opencv(self, video_path: str, output_dir: Path, video_id: str) -> List[str]:
        """Segment video using OpenCV (fallback when ffmpeg is not available)."""
        segments_dir = output_dir / "Segmented_videos"
        segments_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_per_segment = int(fps * self.config.segment_duration)
        segment_count = 0
        current_frame = 0
        segments = []
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        
        print(f"   Using OpenCV for segmentation (ffmpeg not available)")
        
        while current_frame < total_frames:
            if current_frame % frames_per_segment == 0:
                if out is not None:
                    out.release()
                
                segment_path = segments_dir / f"{video_id}_segment_{segment_count:06d}.mp4"
                out = cv2.VideoWriter(str(segment_path), fourcc, fps, (width, height))
                segments.append(str(segment_path))
                segment_count += 1
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if out is not None:
                out.write(frame)
            
            current_frame += 1
        
        if out is not None:
            out.release()
        cap.release()
        
        print(f"Created {len(segments)} segments using OpenCV")
        return segments
    
    def _segment_video(self, video_path: str, output_dir: Path, video_id: str) -> List[str]:
        """Segment video using ffmpeg if available, otherwise use OpenCV."""
        if self._check_ffmpeg():
            # Use ffmpeg (faster, better quality)
            segments_dir = output_dir / "Segmented_videos"
            segments_dir.mkdir(exist_ok=True)
            segment_pattern = str(segments_dir / f"{video_id}_segment_%06d.mp4")
            cmd = ['ffmpeg', '-i', video_path, '-c', 'copy', '-map', '0',
                   '-segment_time', str(self.config.segment_duration), '-f', 'segment',
                   '-reset_timestamps', '1', segment_pattern, '-y', '-loglevel', 'error']
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                segments = sorted(segments_dir.glob(f"{video_id}_segment_*.mp4"))
                print(f"Created {len(segments)} segments using ffmpeg")
                return [str(s) for s in segments]
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to OpenCV if ffmpeg fails
                print("ffmpeg failed, falling back to OpenCV...")
                return self._segment_video_opencv(video_path, output_dir, video_id)
        else:
            # Use OpenCV fallback
            print("ffmpeg not found, using OpenCV for segmentation...")
            return self._segment_video_opencv(video_path, output_dir, video_id)
    
    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(fps / self.config.fps_target))
        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                # Keep native resolution for metrics; resize only when saving outputs.
                frames.append(frame)
            frame_idx += 1
        cap.release()
        return frames, fps
    
    def _classify_quality(self, score: float) -> str:
        """3-tier classification: poor, acceptable, good"""
        if score >= self.config.quality_threshold_good:  # >= 0.60
            return 'good'
        elif score >= self.config.quality_threshold_poor:  # >= 0.40
            return 'acceptable'
        else:  # < 0.40
            return 'poor'
    
    def _check_flags(self, report: SegmentQualityReport) -> Tuple[bool, List[str]]:
        reasons = []
        if report.mean_quality < self.config.quality_threshold_poor: reasons.append("Low quality")
        if report.mean_sharpness < 0.3: reasons.append("Blurry")
        if len(report.artifact_frames) > 0.3 * report.num_extracted_frames: reasons.append("High artifacts")
        if report.temporal_consistency < 0.5: reasons.append("Temporal issues")
        if report.mean_brightness < 0.2: reasons.append("Too dark")
        if report.mean_brightness > 0.8: reasons.append("Overexposed")
        if report.mean_brisque > 60: reasons.append("High distortion")
        return len(reasons) > 0, reasons

    def _hysteresis_select_frames(self, probs: List[float]) -> List[int]:
        """
        Q-Router-style hysteresis selection (Algorithm 2 idea), simplified to frame indices.
        Starts a clip when p >= tau_high and ends when p < tau_low.
        """
        tau_high = self.config.tau_high
        tau_low = self.config.tau_low
        selected: List[int] = []
        active = False
        for t, p in enumerate(probs):
            if (not active) and (p >= tau_high):
                active = True
            if active:
                selected.append(t)
                if p < tau_low:
                    active = False
        return sorted(set(selected))

    def _compute_vlm_routed_quality(
        self,
        frame_metrics: List[Dict[str, Any]],
        vlm_result: Dict[str, Any],
        video_type: str
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Lightweight "expert routing" inspired by Q-Router Tier-1:
        treat existing no-ref experts (sharpness/noise/brightness/contrast/BRISQUE-proxy/NIQE-proxy/temporal)
        as a routing pool and let VLM outputs adjust their weights.
        """
        # Baseline expert weights (sum ~ 1.0)
        w = {
            "sharpness": 0.28,
            "noise": 0.22,
            "brightness": 0.18,
            "contrast": 0.16,
            "brisque": 0.10,
            "niqe": 0.06,
            "temporal": 0.00,
        }

        rationale: List[str] = []
        artifacts = vlm_result.get("artifacts", {}) if isinstance(vlm_result, dict) else {}
        objects = vlm_result.get("objects", {}) if isinstance(vlm_result, dict) else {}

        has_artifacts = bool(artifacts.get("has_artifacts", False))
        artifact_type = str(artifacts.get("artifact_type", "none")).strip().lower()

        # Content cues (cheap heuristics over VLM object categories)
        cats = objects.get("categories", {}) if isinstance(objects, dict) else {}
        has_signs = bool(cats.get("signs"))
        has_tech = bool(cats.get("technology"))
        has_people = bool(cats.get("people"))

        if has_signs or has_tech:
            w["sharpness"] += 0.05
            w["contrast"] += 0.03
            rationale.append("text/screens -> upweight sharpness/contrast")

        if has_people:
            w["sharpness"] += 0.03
            w["noise"] += 0.02
            rationale.append("people/faces -> upweight sharpness/noise")

        # Distortion cues from VLM artifact type
        if has_artifacts and artifact_type == "2":
            w["brisque"] += 0.06
            w["niqe"] += 0.04
            rationale.append("VLM=image artifacts -> upweight BRISQUE-proxy/NIQE-proxy")
        elif has_artifacts and artifact_type in ("1", "3"):
            w["temporal"] += 0.10
            rationale.append("VLM=hallucination/AI inconsistency -> add temporal stability term")

        # Video-type cue
        if video_type in ("dynamic", "walking_tour"):
            w["temporal"] = max(w["temporal"], 0.08)
            rationale.append("dynamic content -> include temporal stability term")

        # Make all weights non-negative, then renormalize
        for k in list(w.keys()):
            w[k] = max(float(w[k]), 0.0)
        s = sum(w.values()) or 1.0
        w = {k: v / s for k, v in w.items()}

        # Compute routed per-frame quality (fast: uses already-computed metrics)
        def _clip01(x: float) -> float:
            return float(np.clip(x, 0.0, 1.0))

        routed_scores: List[float] = []
        for m in frame_metrics:
            sharp = _clip01(m.get("normalized_sharpness", 0.0))
            noise = _clip01(1.0 - min(float(m.get("noise_estimate", 0.0)) / 50.0, 1.0))
            bright = _clip01(1.0 - abs(float(m.get("brightness", 0.0)) - 0.5))
            contr = _clip01(min(float(m.get("contrast", 0.0)), 1.0))
            bris = _clip01(1.0 - float(m.get("brisque_score", 100.0)) / 100.0)
            niqe = _clip01(1.0 - float(m.get("niqe_score", 100.0)) / 100.0)
            temp = _clip01(1.0 - min(float(m.get("motion_residual", 0.0)) / 50.0, 1.0))

            base = (
                w["sharpness"] * sharp
                + w["noise"] * noise
                + w["brightness"] * bright
                + w["contrast"] * contr
                + w["brisque"] * bris
                + w["niqe"] * niqe
                + w["temporal"] * temp
            )
            # Apply the same score calibration used elsewhere
            q = self.config.score_center + (base - self.config.score_center) * self.config.score_stretch
            routed_scores.append(_clip01(q))

        routed_mean = float(np.mean(routed_scores)) if routed_scores else 0.0
        return routed_mean, w, "; ".join(rationale) if rationale else "baseline weights"
    
    def process_segment(self, segment_path: str, segment_id: int,
                       output_dir: Path, video_id: str) -> Tuple[SegmentQualityReport, np.ndarray, List[np.ndarray]]:
        frames, fps = self._extract_frames(segment_path)
        if len(frames) == 0:
            return None, None, []
        
        cap = cv2.VideoCapture(segment_path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
        cap.release()
        
        start_time = segment_id * self.config.segment_duration
        end_time = start_time + duration
        
        # -------------------------------------------------------------------
        # Efficiency: compute metrics ONCE (NR-IQA proxies are expensive).
        # We detect video_type from the first pass, then apply only the small
        # type-adjustment without re-running NR-IQA proxies again.
        # -------------------------------------------------------------------
        frame_metrics: List[Dict[str, Any]] = []
        prev_frame = None
        for i, frame in enumerate(frames):
            timestamp = start_time + (i / self.config.fps_target)
            metrics = self.metrics_calculator.compute_frame_metrics(frame, i, timestamp, prev_frame, "general")
            frame_metrics.append(asdict(metrics))
            prev_frame = frame

        # Detect video type from cheap stats (no second pass)
        video_type = self.type_detector.detect_type(frame_metrics)

        # Apply the same type adjustments used in compute_frame_metrics()
        type_adjustments = {
            'walking_tour': 0.03,
            'static': 0.02,
            'dynamic': -0.02,
            'indoor': 0.01,
            'outdoor': 0.03,
            'general': 0.0
        }
        type_adj = float(type_adjustments.get(video_type, 0.0))

        quality_distribution = {'poor': 0, 'acceptable': 0, 'good': 0}
        artifact_probs: List[float] = []
        for m in frame_metrics:
            q = float(m.get("technical_quality", 0.0)) + type_adj
            q = float(np.clip(q, 0.0, 1.0))
            m["video_type"] = video_type
            m["technical_quality"] = q
            m["overall_quality"] = q
            m["artifact_probability"] = 1.0 - q
            q_class = self.metrics_calculator._classify_frame_quality(q)
            m["quality_class"] = q_class
            quality_distribution[q_class] = quality_distribution.get(q_class, 0) + 1
            artifact_probs.append(float(m["artifact_probability"]))

        # Q-Router-inspired hysteresis selection (stable vs single threshold)
        artifact_frames = self._hysteresis_select_frames(artifact_probs)

        qualities = [float(m.get('overall_quality', 0.0)) for m in frame_metrics]
        mean_quality_raw = float(np.mean(qualities)) if qualities else 0.0
        std_quality = float(np.std(qualities)) if qualities else 0.0
        mean_quality = float(np.clip(mean_quality_raw - self.config.temporal_penalty * std_quality, 0.0, 1.0))
        
        # Efficiency: CLIP embedding for every frame is expensive.
        # Limit to a uniform subset for keyframe selection when there are many frames.
        clip_max = int(getattr(self.config, "clip_max_frames", 0) or 0)
        if clip_max > 0 and len(frames) > clip_max:
            idxs = np.linspace(0, len(frames) - 1, clip_max, dtype=int)
            frames_sub = [frames[i] for i in idxs]
            qualities_sub = [qualities[i] for i in idxs]
            sub_idx = self.keyframe_selector.select_keyframe(frames_sub, qualities_sub)
            keyframe_idx = int(idxs[sub_idx])
        else:
            keyframe_idx = self.keyframe_selector.select_keyframe(frames, qualities)
        keyframe = frames[keyframe_idx]
        keyframe_out = self._resize_to_output_resolution(keyframe)
        
        # Save keyframe (always)
        keyframes_dir = output_dir / "Keyframes"
        keyframes_dir.mkdir(exist_ok=True)
        keyframe_path = keyframes_dir / f"{video_id}_segment_{segment_id:06d}_keyframe.jpg"
        cv2.imwrite(str(keyframe_path), keyframe_out, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        
        # VLM analysis (optional)
        vlm_result = None
        if self.vlm_analyzer and self.vlm_analyzer.enabled:
            try:
                import json
                vlm_dir = output_dir / "VLM_analysis"
                vlm_dir.mkdir(exist_ok=True)
                vlm_json = vlm_dir / f"{video_id}_segment_{segment_id:06d}_vlm.json"
                
                # Check if JSON already exists - skip to avoid reprocessing and save time/costs
                # Works for all models: Qwen, Florence, GPT-4O, GPT-5mini, etc.
                if vlm_json.exists():
                    try:
                        with open(vlm_json, 'r', encoding='utf-8') as f:
                            existing = json.load(f)
                        # Check if it's a valid VLM result
                        if isinstance(existing, dict):
                            # Check for simplified format (from main.py phase1 images mode)
                            # Has: objects (list) or objects_list (list), num_objects, blur
                            # Works for all models: Qwen, Florence, GPT-4O, GPT-5mini, etc.
                            objects_data = existing.get('objects') or existing.get('objects_list')
                            if objects_data is not None and isinstance(objects_data, list) and 'num_objects' in existing:
                                # Convert simplified format to full format expected by keyfram_analysis
                                # Wrap the objects list in a dict structure matching extract_objects_from_keyframe output
                                vlm_result = {
                                    'enabled': True,
                                    'model': self.vlm_analyzer.vlm.model_type if (hasattr(self.vlm_analyzer, 'vlm') and hasattr(self.vlm_analyzer.vlm, 'model_type')) else 'unknown',
                                    'objects': {
                                        'objects': objects_data,
                                        'num_objects': existing.get('num_objects', len(objects_data)),
                                        'categories': {},  # Simplified format doesn't have categories
                                        'error': False
                                    },
                                    'keyframe_path': str(keyframe_path)
                                }
                                logger.debug(f"Skipping VLM analysis for segment {segment_id}: JSON already exists (simplified format)")
                            # Check for full format (from keyfram_analysis.py video mode)
                            # Has: enabled, objects (dict), model, artifacts, etc.
                            elif existing.get('enabled') or ('objects' in existing and isinstance(existing.get('objects'), dict)):
                                vlm_result = existing
                                logger.debug(f"Skipping VLM analysis for segment {segment_id}: JSON already exists (full format)")
                            else:
                                # Invalid format, reprocess
                                logger.debug(f"Existing JSON for segment {segment_id} has invalid format, reprocessing...")
                                vlm_result = None
                        else:
                            # Invalid format, reprocess
                            vlm_result = None
                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        # If JSON is invalid or corrupted, reprocess it
                        logger.debug(f"Existing JSON for segment {segment_id} is invalid/corrupted ({e}), reprocessing...")
                        vlm_result = None
                
                # Only analyze if we don't have a valid existing result
                if vlm_result is None:
                    vlm_result = self.vlm_analyzer.analyze_keyframe(str(keyframe_path))
                    if vlm_result.get('enabled'):
                        with open(vlm_json, 'w', encoding='utf-8') as f:
                            json.dump(vlm_result, f, indent=2)
                    elif vlm_result.get('skipped'):
                        # CUDA error - skip silently, continue processing
                        pass
            except Exception as e:
                logger.warning(f"VLM analysis failed for segment {segment_id}: {e}")
                # Continue without VLM analysis

        # Lightweight Q-Router-style routing: use VLM output to adjust expert weights (Tier-1 idea)
        routed_mean_quality = 0.0
        routing_weights: Dict[str, float] = {}
        routing_rationale = ""
        vlm_enabled = bool(vlm_result and vlm_result.get("enabled"))
        vlm_model = str(vlm_result.get("model", "")) if vlm_enabled else ""
        vlm_has_artifacts = bool(vlm_result.get("artifacts", {}).get("has_artifacts", False)) if vlm_enabled else False
        vlm_artifact_type = str(vlm_result.get("artifacts", {}).get("artifact_type", "none")) if vlm_enabled else "none"

        if vlm_enabled:
            try:
                routed_mean_quality, routing_weights, routing_rationale = self._compute_vlm_routed_quality(
                    frame_metrics, vlm_result, video_type
                )
                mean_quality = float(
                    np.clip(
                        (1.0 - self.config.router_alpha) * mean_quality + self.config.router_alpha * routed_mean_quality,
                        0.0,
                        1.0,
                    )
                )
                # If VLM says "no artifacts", reduce false positives for heatmaps/flagging
                if not vlm_has_artifacts:
                    artifact_frames = []
            except Exception as e:
                logger.debug(f"VLM routing skipped (segment {segment_id}): {e}")
        
        # Save frames (optional, controlled by config.save_all_frames / CLI --no-all-frames)
        saved_count = 1  # keyframe
        if self.config.save_all_frames:
            frames_dir = output_dir / "Segmented_frames" / f"{video_id}_segment_{segment_id:06d}"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                frame_path = frames_dir / f"frame_{i:06d}.jpg"
                frame_out = self._resize_to_output_resolution(frame)
                cv2.imwrite(str(frame_path), frame_out, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
            saved_count = len(frames)
            print(f"Segment {segment_id}: Saved {saved_count} frames (all frames stored)")
        else:
            print(f"Segment {segment_id}: Saved keyframe only")
        
        report = SegmentQualityReport(
            segment_id=segment_id, segment_path=segment_path,
            start_time=start_time, end_time=end_time, duration=duration,
            num_frames=int(duration * fps), num_extracted_frames=len(frames),
            num_saved_frames=saved_count,  # Track saved frames
            mean_quality=float(mean_quality),
            min_quality=float(np.min(qualities)) if qualities else 0.0,
            max_quality=float(np.max(qualities)) if qualities else 0.0,
            std_quality=std_quality,
            mean_sharpness=float(np.mean([float(m.get('normalized_sharpness', 0.0)) for m in frame_metrics])) if frame_metrics else 0.0,
            mean_brightness=float(np.mean([float(m.get('brightness', 0.0)) for m in frame_metrics])) if frame_metrics else 0.0,
            mean_contrast=float(np.mean([float(m.get('contrast', 0.0)) for m in frame_metrics])) if frame_metrics else 0.0,
            mean_brisque=float(np.mean([float(m.get('brisque_score', 0.0)) for m in frame_metrics])) if frame_metrics else 0.0,
            mean_niqe=float(np.mean([float(m.get('niqe_score', 0.0)) for m in frame_metrics])) if frame_metrics else 0.0,
            temporal_consistency=float(1.0 - min(float(np.std([float(m.get('motion_residual', 0.0)) for m in frame_metrics])) / 50.0, 1.0)) if frame_metrics else 1.0,
            artifact_frames=artifact_frames,
            artifact_severity=len(artifact_frames) / len(frames) if frames else 0,
            quality_class=self._classify_quality(mean_quality),
            keyframe_idx=keyframe_idx, keyframe_path=str(keyframe_path),
            frame_metrics=frame_metrics, video_type=video_type,
            frame_quality_distribution=quality_distribution,
            vlm_enabled=vlm_enabled,
            vlm_model=vlm_model,
            vlm_has_artifacts=vlm_has_artifacts,
            vlm_artifact_type=vlm_artifact_type,
            routed_mean_quality=float(routed_mean_quality),
            routing_weights=routing_weights,
            routing_rationale=routing_rationale
        )
        
        is_flagged, reasons = self._check_flags(report)
        report.is_flagged = is_flagged
        report.flag_reasons = reasons
        
        return report, keyframe_out, frames
    
    def process_video(self, input_source: str, output_base: str = "./output") -> Dict:
        """Process complete video from file or YouTube URL."""
        start_time = datetime.now()
        is_youtube = 'youtube.com' in input_source or 'youtu.be' in input_source
        
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        
        if is_youtube:
            video_path, video_id, title = self._download_youtube(input_source, output_base)
        else:
            video_path = input_source
            video_id = Path(video_path).stem[:20]
            title = Path(video_path).stem
        
        title_safe = self._sanitize_filename(title)
        # Avoid duplicate folder names like "Airport_BWI_Airport_BWI" for local files.
        if is_youtube:
            output_dir_name = f"{video_id}_{title_safe}"
        else:
            output_dir_name = title_safe or video_id
        output_dir = output_base / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not is_youtube:
            # Preserve the source extension (don't rename .MOV -> .mp4)
            dest = output_dir / Path(video_path).name
            if not dest.exists():
                shutil.copy2(video_path, dest)
            video_path = str(dest)
        
        self.visualizer = QualityVisualizer(str(output_dir))
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        cap.release()
        
        print(f"\nVideo: {title}")
        print(f"   Duration: {duration:.1f}s | Resolution: {width}x{height} | FPS: {fps:.1f}")
        print(f"   Output images: {self.config.output_resolution[0]}x{self.config.output_resolution[1]}")
        
        print("\nSegmenting video...")
        segments = self._segment_video(video_path, output_dir, video_id)
        
        print(f"\nAnalyzing {len(segments)} segments...")
        if self.config.save_all_frames:
            print("   Saving ALL frames (5-6 per segment) regardless of quality")
        else:
            print("   Saving keyframes only (frames are not exported)")
        segment_reports = []
        all_keyframes = []
        
        for i, segment_path in enumerate(tqdm(segments, desc="Processing")):
            report, keyframe, frames = self.process_segment(segment_path, i, output_dir, video_id)
            if report is not None:
                segment_reports.append(report)
                all_keyframes.append(keyframe)
                
                # Always save visualizations
                if self.config.save_visualizations and frames:
                    self.visualizer.create_segment_dashboard(report, frames, keyframe, title)
                
                if self.config.save_heatmaps and len(report.artifact_frames) > 0 and len(frames) > 1:
                    for af_idx in report.artifact_frames[:2]:
                        if af_idx > 0 and af_idx < len(frames):
                            try:
                                heatmap = self.metrics_calculator.compute_lpips_heatmap(frames[af_idx-1], frames[af_idx])
                                self.visualizer.create_artifact_heatmap(frames[af_idx], heatmap, i, af_idx)
                            except:
                                pass
        
        if segment_reports:
            best_idx = max(range(len(segment_reports)), key=lambda i: segment_reports[i].mean_quality)
            best_keyframe_path = output_dir / "Keyframes" / f"{video_id}_BEST_KEYFRAME.jpg"
            cv2.imwrite(str(best_keyframe_path), all_keyframes[best_idx], [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        else:
            best_idx = 0
            best_keyframe_path = ""
        
        all_qualities = [float(r.mean_quality) for r in segment_reports]
        overall_quality = float(np.mean(all_qualities)) if all_qualities else 0.0
        flagged = [r for r in segment_reports if r.is_flagged]
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate total frames saved
        total_frames_saved = sum(r.num_saved_frames for r in segment_reports)
        
        video_report = {
            "video_id": video_id, "video_path": video_path, "video_title": title,
            "duration": duration, "resolution": (width, height), "fps": fps, "total_frames": total_frames,
            "output_resolution": self.config.output_resolution,
            "num_segments": len(segment_reports), "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "score_calibration": {
                "score_center": self.config.score_center,
                "score_stretch": self.config.score_stretch,
                "temporal_penalty": self.config.temporal_penalty,
                "router_alpha": self.config.router_alpha,
            },
            "nr_iqa": {
                "backend": "proxy",
                "note": "NR-IQA values are lightweight proxies (not the standard BRISQUE/NIQE implementations).",
            },
            "overall_quality_score": overall_quality,
            "quality_class": self._classify_quality(overall_quality),
            "mean_segment_quality": float(np.mean(all_qualities)) if all_qualities else 0.0,
            "min_segment_quality": float(np.min(all_qualities)) if all_qualities else 0.0,
            "max_segment_quality": float(np.max(all_qualities)) if all_qualities else 0.0,
            "num_flagged_segments": len(flagged),
            "flagged_segment_ids": [r.segment_id for r in flagged],
            "best_keyframe_path": str(best_keyframe_path),
            "best_keyframe_segment": best_idx,
            "total_frames_saved": total_frames_saved,
            "segments": [asdict(r) for r in segment_reports]
        }
        
        # Save reports
        reports_dir = output_dir / "Quality_reports"
        reports_dir.mkdir(exist_ok=True)
        with open(reports_dir / f"{video_id}_quality_report.json", 'w') as f:
            json.dump(video_report, f, indent=2, default=str)
        
        with open(reports_dir / f"{video_id}_summary.txt", 'w') as f:
            f.write(f"VIDEO QUALITY REPORT\n{'='*50}\n\n")
            f.write(f"Video: {title}\nDuration: {duration:.1f}s\nResolution: {width}x{height}\n")
            f.write(f"Output Resolution: {self.config.output_resolution[0]}x{self.config.output_resolution[1]} (1080p)\n\n")
            f.write(f"OVERALL QUALITY: {overall_quality:.3f} ({self._classify_quality(overall_quality).upper()})\n\n")
            f.write(f"Segments: {len(segment_reports)} | Flagged: {len(flagged)}\n")
            if self.config.save_all_frames:
                f.write(f"Total Frames Saved: {total_frames_saved} (ALL frames stored, 5-6 per segment)\n\n")
            else:
                f.write(f"Total Frames Saved: {total_frames_saved} (keyframes only)\n\n")
            f.write(f"Quality Thresholds:\n")
            f.write(f"  - Poor: < {self.config.quality_threshold_poor}\n")
            f.write(f"  - Acceptable: {self.config.quality_threshold_poor} - {self.config.quality_threshold_good}\n")
            f.write(f"  - Good: >= {self.config.quality_threshold_good}\n\n")
            if flagged:
                f.write("FLAGGED:\n")
                for r in flagged:
                    f.write(f"  - Segment {r.segment_id}: {', '.join(r.flag_reasons)}\n")
        
        # Generate overview
        if self.config.save_visualizations and segment_reports:
            print("\nGenerating overview...")
            self.visualizer.create_video_overview(video_report, segment_reports, all_keyframes)
        
        # Save flagged info
        flagged_dir = output_dir / "Flagged_segments"
        flagged_dir.mkdir(exist_ok=True)
        for r in flagged:
            with open(flagged_dir / f"segment_{r.segment_id:06d}_FLAGGED.txt", 'w') as f:
                f.write(f"Segment {r.segment_id} - FLAGGED\n")
                f.write(f"Reasons: {', '.join(r.flag_reasons)}\n")
                f.write(f"Quality: {r.mean_quality:.3f} ({r.quality_class})\n")
                f.write(f"Video Type: {r.video_type}\n")
                f.write(f"Frames Saved: {r.num_saved_frames}/{r.num_extracted_frames}\n\n")
                if self.config.save_all_frames:
                    f.write("NOTE: All frames and keyframes for this segment are stored.\n")
                    f.write("They are available in Segmented_frames/ and Keyframes/ directories.\n")
                else:
                    f.write("NOTE: Only the keyframe is stored for this segment.\n")
                    f.write("It is available in the Keyframes/ directory.\n")
        
        print("\nComplete!")
        print(f"   Output: {output_dir}")
        print(f"   Quality: {overall_quality:.3f} ({self._classify_quality(overall_quality).upper()})")
        if self.config.save_all_frames:
            print(f"   Total Frames Saved: {total_frames_saved} (all frames, 5-6 per segment)")
        else:
            print(f"   Total Frames Saved: {total_frames_saved} (keyframes only)")
        print(f"   Flagged: {len(flagged)}/{len(segment_reports)} (frames/keyframes still stored)")
        print(f"   Time: {processing_time:.1f}s")
        
        return video_report, segment_reports, output_dir


# =======================================================================
# EASY RUN FUNCTION
# =======================================================================

def analyze_video(source: str, output_dir: str = "./output", segment_duration: int = 6,
                  fps_target: int = 1, save_visualizations: bool = True,
                  enable_vlm: bool = False, vlm_model: str = "qwen", vlm_api_key: str = None, vlm_device: str = "auto"):
    """
    🎬 Easy function to analyze any video.
    
    Args:
        source: Path to video file OR YouTube URL
        output_dir: Where to save results
        segment_duration: Segment length in seconds
        fps_target: Frames to extract per second
        save_visualizations: Generate dashboard images
        enable_vlm: Enable VLM analysis of keyframes
        vlm_model: "qwen" or "gpt4o"
        vlm_api_key: OpenAI API key (for GPT-4o)
    
    Returns:
        video_report, segment_reports, output_path
    """
    config = QualityConfig(
        segment_duration=segment_duration,
        fps_target=fps_target,
        save_visualizations=save_visualizations,
        output_resolution=(1920, 1080),  # 1080p
        save_all_frames=True  # Save ALL frames
    )
    pipeline = VideoQualityPipeline(
        config,
        enable_vlm=enable_vlm,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
        vlm_device=vlm_device
    )
    return pipeline.process_video(source, output_dir)


# =======================================================================
# DISPLAY RESULTS FUNCTION
# =======================================================================

def display_results(output_dir):
    """Display results in Colab."""
    from IPython.display import display, Image as IPImage
    import glob
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / "Visualizations"
    
    # Show overview
    overview = viz_dir / "video_overview.png"
    if overview.exists():
        print("VIDEO OVERVIEW:")
        display(IPImage(filename=str(overview), width=1000))
    
    # Show segment dashboards
    dashboards = sorted(viz_dir.glob("segment_*_dashboard.png"))
    if dashboards:
        print(f"\nSEGMENT DASHBOARDS ({len(dashboards)} segments):")
        for dash in dashboards[:5]:
            display(IPImage(filename=str(dash), width=1000))
    
    # Show artifact heatmaps
    heatmaps = sorted(viz_dir.glob("artifact_*.png"))
    if heatmaps:
        print(f"\nARTIFACT HEATMAPS ({len(heatmaps)} found):")
        for hm in heatmaps[:3]:
            display(IPImage(filename=str(hm), width=800))


if __name__ == "__main__":
    import sys
    
    # Check if running from command line with arguments
    if len(sys.argv) > 1:
        # Command line mode - use argument parser
        try:
            from arg import parse_args, create_config_from_args
            
            args = parse_args()
            
            print("=" * 60)
            print(" VIDEO QUALITY ASSESSMENT PIPELINE")
            print("=" * 60)
            print(f"\nSource: {args.source}")
            print(f"Output: {args.output}")
            print(f"Settings: {args.segment_duration}s segments, {args.fps_target} fps")
            print(f"Quality: Poor < {args.poor_threshold}, Good >= {args.good_threshold}")
            print(f"Save all frames: {not args.no_save_all_frames}")
            print()
            
            # Create config from arguments
            config = create_config_from_args(args)
            
            # Create pipeline and process
            pipeline = VideoQualityPipeline(
                config,
                enable_vlm=args.enable_vlm,
                vlm_model=args.vlm_model,
                vlm_api_key=args.vlm_api_key,
                vlm_device=getattr(args, "vlm_device", "auto")
            )
            video_report, segment_reports, output_dir = pipeline.process_video(
                args.source,
                args.output
            )
            
            # Print summary
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE!")
            print("=" * 60)
            print(f"\nResults saved to: {output_dir}")
            print(f"Overall Quality: {video_report['overall_quality_score']:.3f}")
            print(f"Quality Class: {video_report['quality_class'].upper()}")
            print(f"Segments Analyzed: {video_report['num_segments']}")
            print(f"Flagged Segments: {video_report['num_flagged_segments']}")
            print(f"Total Frames Saved: {video_report['total_frames_saved']}")
            print(f"Processing Time: {video_report['processing_time']:.1f}s")
            
        except ImportError as e:
            print(f"Error importing argument parser: {e}")
            print("   Make sure arg.py is in the same directory.")
            sys.exit(1)
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Interactive/library mode - show usage
        print("=" * 60)
        print(" VIDEO QUALITY ASSESSMENT PIPELINE")
        print(" Ready for analysis!")
        print("=" * 60)
        print("\nUsage:")
        print("  # Command line:")
        print("  python keyfram_analysis.py video.mp4")
        print("  python keyfram_analysis.py 'https://youtube.com/watch?v=...'")
        print("  python keyfram_analysis.py video.mp4 --output ./results --segment-duration 10")
        print("\n  # Python API:")
        print("  from keyfram_analysis import analyze_video")
        print("  report, segments, output = analyze_video('video.mp4')")
        print("  report, segments, output = analyze_video('https://youtube.com/watch?v=...')")
        print("\n  # Display results (Colab only):")
        print("  from keyfram_analysis import display_results")
        print("  display_results(output)")
        print("\nFor help: python keyfram_analysis.py --help")