"""Visualization utilities for LLM-as-judge evaluation results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def generate_all_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate all visualization plots for evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    
    try:
        plot_score_distribution(results_df, output_dir)
        plot_criteria_comparison(results_df, output_dir)
        plot_model_comparison(results_df, output_dir)
        plot_score_heatmap(results_df, output_dir)
        plot_overall_score_histogram(results_df, output_dir)
        plot_criteria_correlation(results_df, output_dir)
        
        logger.info(f"Generated all plots in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)


def plot_score_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot distribution of scores for each criterion."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Score Distribution by Criterion", fontsize=16, fontweight="bold")
    
    criteria = ["clarity", "completeness", "robustness", "user_friendliness", "accuracy", "overall_score"]
    axes = axes.flatten()
    
    for idx, criterion in enumerate(criteria):
        if criterion in df.columns:
            ax = axes[idx]
            ax.hist(df[criterion], bins=20, edgecolor="black", alpha=0.7, color="skyblue")
            ax.set_xlabel("Score", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(criterion.replace("_", " ").title(), fontsize=12, fontweight="bold")
            ax.set_xlim(0, 11)
            ax.axvline(df[criterion].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df[criterion].mean():.2f}")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated score distribution plot")


def plot_criteria_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot comparison of average scores across criteria."""
    criteria = ["clarity", "completeness", "robustness", "user_friendliness", "accuracy"]
    
    # Calculate means and std
    means = [df[c].mean() for c in criteria if c in df.columns]
    stds = [df[c].std() for c in criteria if c in df.columns]
    labels = [c.replace("_", " ").title() for c in criteria if c in df.columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="coral", edgecolor="black")
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_xlabel("Criteria", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Score", fontsize=12, fontweight="bold")
    ax.set_title("Average Scores by Evaluation Criteria", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 11)
    ax.axhline(5, color="gray", linestyle="--", alpha=0.5, label="Midpoint (5.0)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "criteria_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated criteria comparison plot")


def plot_model_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot comparison of scores across different judge models."""
    if "model" not in df.columns or df["model"].nunique() <= 1:
        logger.info("Skipping model comparison (single model or no model column)")
        return
    
    criteria = ["clarity", "completeness", "robustness", "user_friendliness", "accuracy", "overall_score"]
    available_criteria = [c for c in criteria if c in df.columns]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Group by model and calculate means
    model_means = df.groupby("model")[available_criteria].mean()
    
    x = np.arange(len(available_criteria))
    width = 0.8 / len(model_means)
    
    for idx, (model_name, row) in enumerate(model_means.iterrows()):
        offset = width * idx - (width * len(model_means) / 2) + width / 2
        ax.bar(x + offset, row.values, width, label=model_name, alpha=0.8, edgecolor="black")
    
    ax.set_xlabel("Criteria", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison Across Criteria", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in available_criteria], rotation=15, ha="right")
    ax.set_ylim(0, 11)
    ax.legend(title="Judge Model", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated model comparison plot")


def plot_score_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot heatmap of scores across images and criteria."""
    criteria = ["clarity", "completeness", "robustness", "user_friendliness", "accuracy", "overall_score"]
    available_criteria = [c for c in criteria if c in df.columns]
    
    # Limit to first 30 images for readability
    plot_df = df.head(30)[["image_name"] + available_criteria].copy()
    plot_df["image_name"] = plot_df["image_name"].apply(lambda x: Path(x).stem[:20])  # Truncate long names
    
    # Set image_name as index
    plot_df = plot_df.set_index("image_name")
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(plot_df) * 0.3)))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=1,
        vmax=10,
        cbar_kws={"label": "Score"},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_xlabel("Criteria", fontsize=12, fontweight="bold")
    ax.set_ylabel("Image", fontsize=12, fontweight="bold")
    ax.set_title("Score Heatmap (First 30 Images)", fontsize=14, fontweight="bold")
    ax.set_xticklabels([c.replace("_", " ").title() for c in available_criteria], rotation=15, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated score heatmap")


def plot_overall_score_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot histogram of overall scores with statistics."""
    if "overall_score" not in df.columns:
        logger.warning("No overall_score column found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scores = df["overall_score"]
    ax.hist(scores, bins=30, edgecolor="black", alpha=0.7, color="mediumseagreen")
    
    # Add statistics
    mean_score = scores.mean()
    median_score = scores.median()
    std_score = scores.std()
    
    ax.axvline(mean_score, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_score:.2f}")
    ax.axvline(median_score, color="blue", linestyle="--", linewidth=2, label=f"Median: {median_score:.2f}")
    
    ax.set_xlabel("Overall Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(f"Overall Score Distribution (σ={std_score:.2f})", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "overall_score_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated overall score histogram")


def plot_criteria_correlation(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot correlation matrix between different criteria."""
    criteria = ["clarity", "completeness", "robustness", "user_friendliness", "accuracy", "overall_score"]
    available_criteria = [c for c in criteria if c in df.columns]
    
    if len(available_criteria) < 2:
        logger.warning("Not enough criteria for correlation plot")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[available_criteria].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"label": "Correlation"},
        ax=ax
    )
    
    ax.set_title("Criteria Correlation Matrix", fontsize=14, fontweight="bold")
    ax.set_xticklabels([c.replace("_", " ").title() for c in available_criteria], rotation=45, ha="right")
    ax.set_yticklabels([c.replace("_", " ").title() for c in available_criteria], rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / "criteria_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Generated criteria correlation plot")
