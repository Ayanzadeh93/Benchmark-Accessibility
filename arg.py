#!/usr/bin/env python3
"""
Argument parser for Video Quality Assessment Pipeline
"""
import argparse
from typing import Tuple


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for video quality assessment."""
    
    parser = argparse.ArgumentParser(
        description='Video Quality Assessment Pipeline - Scientific analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with local video
  python keyfram_analysis.py video.mp4
  
  # YouTube URL
  python keyfram_analysis.py "https://youtube.com/watch?v=..."
  
  # Custom output directory and settings
  python keyfram_analysis.py video.mp4 --output ./results --segment-duration 10 --fps 2
  
  # Advanced quality thresholds
  python keyfram_analysis.py video.mp4 --poor-threshold 0.35 --good-threshold 0.65
  
  # Disable visualizations for faster processing
  python keyfram_analysis.py video.mp4 --no-visualizations --no-heatmaps
        """
    )
    
    # =======================================================================
    # REQUIRED ARGUMENTS
    # =======================================================================
    
    parser.add_argument(
        'source',
        type=str,
        help='Video file path OR YouTube URL'
    )
    
    # =======================================================================
    # BASIC OPTIONS
    # =======================================================================
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    parser.add_argument(
        '-s', '--segment-duration',
        type=int,
        default=6,
        metavar='SECONDS',
        help='Segment duration in seconds (default: 6)'
    )
    
    parser.add_argument(
        '-f', '--fps',
        type=int,
        default=1,
        dest='fps_target',
        metavar='FPS',
        help='Frames per second to extract (default: 1)'
    )
    
    # =======================================================================
    # QUALITY THRESHOLDS
    # =======================================================================
    
    quality_group = parser.add_argument_group('Quality Thresholds')
    
    quality_group.add_argument(
        '--poor-threshold',
        type=float,
        default=0.40,
        metavar='SCORE',
        help='Quality score below this is "poor" (default: 0.40)'
    )
    
    quality_group.add_argument(
        '--good-threshold',
        type=float,
        default=0.60,
        metavar='SCORE',
        help='Quality score above this is "good" (default: 0.60)'
    )
    
    # =======================================================================
    # OUTPUT SETTINGS
    # =======================================================================
    
    output_group = parser.add_argument_group('Output Settings')
    
    output_group.add_argument(
        '--resolution',
        type=str,
        default='1920x1080',
        metavar='WxH',
        help='Output resolution as WIDTHxHEIGHT (default: 1920x1080)'
    )
    
    output_group.add_argument(
        '--jpeg-quality',
        type=int,
        default=95,
        choices=range(1, 101),
        metavar='1-100',
        help='JPEG quality for saved frames (default: 95)'
    )
    
    output_group.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Disable dashboard and overview visualizations'
    )
    
    output_group.add_argument(
        '--no-heatmaps',
        action='store_true',
        help='Disable artifact heatmap generation'
    )
    
    output_group.add_argument(
        '--no-all-frames',
        action='store_true',
        dest='no_save_all_frames',
        help='Only save keyframes (not all frames)'
    )
    
    # =======================================================================
    # MODEL SETTINGS
    # =======================================================================
    
    model_group = parser.add_argument_group('Model Settings')
    
    model_group.add_argument(
        '--clip-model',
        type=str,
        default='ViT-B/32',
        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'],
        help='CLIP model for keyframe selection (default: ViT-B/32)'
    )
    
    model_group.add_argument(
        '--lpips-net',
        type=str,
        default='alex',
        choices=['alex', 'vgg', 'squeeze'],
        help='LPIPS network for artifact detection (default: alex)'
    )
    
    # =======================================================================
    # ADVANCED OPTIONS (Optional)
    # =======================================================================
    
    advanced_group = parser.add_argument_group('Advanced Options (Optional)')

    # =======================================================================
    # SCORE CALIBRATION (Optional - fixes narrow score band)
    # =======================================================================

    advanced_group.add_argument(
        '--score-stretch',
        type=float,
        default=None,
        metavar='FLOAT',
        help='Score stretch factor around 0.5 (e.g., 1.6 widens dynamic range)'
    )

    advanced_group.add_argument(
        '--temporal-penalty',
        type=float,
        default=None,
        metavar='FLOAT',
        help='Penalty on segment score based on std over frames (e.g., 0.12)'
    )

    advanced_group.add_argument(
        '--router-alpha',
        type=float,
        default=None,
        metavar='FLOAT',
        help='Blend factor for VLM-routed score into segment mean (e.g., 0.35)'
    )

    advanced_group.add_argument(
        '--nr-max-side',
        type=int,
        default=None,
        metavar='PX',
        help='Max side length for BRISQUE/NIQE computation (downsample for speed, e.g., 512)'
    )

    advanced_group.add_argument(
        '--clip-max-frames',
        type=int,
        default=None,
        metavar='N',
        help='Max frames per segment to embed with CLIP for keyframe selection (e.g., 32)'
    )
    
    # =======================================================================
    # VLM OPTIONS
    # =======================================================================
    
    vlm_group = parser.add_argument_group('VLM Analysis (Optional)')
    
    vlm_group.add_argument(
        '--enable-vlm',
        action='store_true',
        help='Enable VLM analysis of keyframes'
    )
    
    vlm_group.add_argument(
        '--vlm-model',
        type=str,
        default='qwen',
        choices=['qwen', 'gpt4o'],
        help='VLM model: qwen (free, GPU) or gpt4o (API, paid)'
    )
    
    vlm_group.add_argument(
        '--vlm-api-key',
        type=str,
        default=None,
        help='OpenAI API key (for GPT-4o, or use .env file)'
    )

    vlm_group.add_argument(
        '--vlm-device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device for Qwen VLM: auto/cuda/cpu (default: auto)'
    )
    
    # =======================================================================
    # DETECTION OPTIONS (Phase 2)
    # =======================================================================
    
    detection_group = parser.add_argument_group('Detection Options (Phase 2)')
    
    detection_group.add_argument(
        '--detection-conf',
        type=float,
        default=0.15,
        metavar='FLOAT',
        help='Detection confidence threshold (default: 0.15, lower = more detections)'
    )
    
    detection_group.add_argument(
        '--detection-text-threshold',
        type=float,
        default=0.20,
        metavar='FLOAT',
        help='Text matching threshold for GroundingDINO (default: 0.20)'
    )
    
    detection_group.add_argument(
        '--detection-nms-iou',
        type=float,
        default=0.50,
        metavar='FLOAT',
        help='NMS IoU threshold for removing duplicate detections (default: 0.50)'
    )
    
    # =======================================================================
    # UTILITY FLAGS
    # =======================================================================
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Video Quality Assessment Pipeline v1.0'
    )
    
    return parser.parse_args()


def parse_resolution(resolution_str: str) -> Tuple[int, int]:
    """Parse resolution string like '1920x1080' into (width, height)."""
    try:
        parts = resolution_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError
        return (width, height)
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid resolution format: {resolution_str}. Expected format: WIDTHxHEIGHT (e.g., 1920x1080)"
        )


def create_config_from_args(args: argparse.Namespace):
    """Create QualityConfig from parsed arguments."""
    # Avoid double-importing `keyfram_analysis.py` when it is executed as a script.
    # If we're being called from `keyfram_analysis.py` (as __main__), prefer that class.
    QualityConfig = None
    try:
        import __main__ as main  # type: ignore
        QualityConfig = getattr(main, "QualityConfig", None)
    except Exception:
        QualityConfig = None
    if QualityConfig is None:
        from keyfram_analysis import QualityConfig
    
    # Parse resolution
    output_resolution = parse_resolution(args.resolution)
    
    # Build config with provided arguments
    config_kwargs = {
        'segment_duration': args.segment_duration,
        'fps_target': args.fps_target,
        'quality_threshold_poor': args.poor_threshold,
        'quality_threshold_good': args.good_threshold,
        'clip_model': args.clip_model,
        'lpips_net': args.lpips_net,
        'jpeg_quality': args.jpeg_quality,
        'save_heatmaps': not args.no_heatmaps,
        'save_visualizations': not args.no_visualizations,
        'output_resolution': output_resolution,
        'save_all_frames': not args.no_save_all_frames,
    }
    
    # Score calibration (optional)
    if getattr(args, "score_stretch", None) is not None:
        config_kwargs['score_stretch'] = args.score_stretch
    if getattr(args, "temporal_penalty", None) is not None:
        config_kwargs['temporal_penalty'] = args.temporal_penalty
    if getattr(args, "router_alpha", None) is not None:
        config_kwargs['router_alpha'] = args.router_alpha
    if getattr(args, "nr_max_side", None) is not None:
        config_kwargs['nr_quality_max_side'] = args.nr_max_side
    if getattr(args, "clip_max_frames", None) is not None:
        config_kwargs['clip_max_frames'] = args.clip_max_frames
    
    return QualityConfig(**config_kwargs)


if __name__ == '__main__':
    # Test argument parsing
    args = parse_args()
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
