"""
Visual Insights Generator for Template-Niche Analysis
Generates charts and visualizations for template distribution across niches
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Tuple

# ================= CONFIG =================
ANALYSIS_DIR = Path("data/output/analysis_reports")
OUTPUT_DIR = Path("data/output/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ================= DATA LOADING =================
def load_data():
    """Load all analysis data from CSV files."""
    dominant = pd.read_csv(ANALYSIS_DIR / "dominant_templates_by_niche.csv")
    template_summary = pd.read_csv(ANALYSIS_DIR / "template_niche_summary.csv")
    diversity = pd.read_csv(ANALYSIS_DIR / "niche_diversity_metrics.csv")
    cross_niche = pd.read_csv(ANALYSIS_DIR / "templates_cross_niche.csv")
    primary_format = pd.read_csv(ANALYSIS_DIR / "primary_format_by_niche.csv")
    storytelling = pd.read_csv(ANALYSIS_DIR / "storytelling_structure_by_niche.csv")
    versatility_comp = pd.read_csv(ANALYSIS_DIR / "versatility_complexity_matrix.csv")
    personality = pd.read_csv(ANALYSIS_DIR / "personality_by_niche.csv")
    versatile_templates = pd.read_csv(ANALYSIS_DIR / "versatile_templates_cross_niche.csv")
    
    return {
        "dominant": dominant,
        "summary": template_summary,
        "diversity": diversity,
        "cross_niche": cross_niche,
        "primary_format": primary_format,
        "storytelling": storytelling,
        "versatility_complexity": versatility_comp,
        "personality": personality,
        "versatile_templates": versatile_templates
    }

# ================= VISUALIZATION 1: TOP TEMPLATES BY NICHE =================
def plot_top_templates_by_niche(data: dict) -> None:
    """
    Create a bar chart showing top templates for each niche.
    """
    df = data["summary"].copy()
    
    # Get top 5 niches by video count
    top_niches = df.groupby("niche")["video_count"].sum().nlargest(5).index.tolist()
    df_top = df[df["niche"].isin(top_niches)].copy()
    
    # Get top 3 templates per niche
    df_top = df_top.sort_values(["niche", "video_count"], ascending=[True, False])
    df_top = df_top.groupby("niche").head(3)
    
    fig, axes = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create grouped bar chart
    niches = df_top["niche"].unique()
    x = np.arange(len(niches))
    width = 0.25
    
    templates_per_niche = {}
    for niche in niches:
        templates_per_niche[niche] = df_top[df_top["niche"] == niche].sort_values("video_count", ascending=False)
    
    # Plot stacked approach for clarity
    bottom = np.zeros(len(niches))
    colors = sns.color_palette("husl", 8)
    
    all_templates = set()
    for niche in niches:
        for _, row in templates_per_niche[niche].iterrows():
            all_templates.add(row["template_category"])
    
    for idx, template in enumerate(sorted(all_templates)[:6]):
        values = []
        for niche in niches:
            count = df_top[(df_top["niche"] == niche) & (df_top["template_category"] == template)]["video_count"].sum()
            values.append(count)
        axes.bar(x, values, width=0.6, label=template, bottom=bottom, color=colors[idx % len(colors)])
        bottom += np.array(values)
    
    axes.set_xlabel("Niche", fontsize=12, fontweight="bold")
    axes.set_ylabel("Video Count", fontsize=12, fontweight="bold")
    axes.set_title("Top Template Categories by Niche (Top 5 Niches)", fontsize=14, fontweight="bold")
    axes.set_xticks(x)
    axes.set_xticklabels([niche.replace(" > ", "\n> ") for niche in niches], rotation=45, ha="right")
    axes.legend(loc="upper right", fontsize=9)
    axes.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_top_templates_by_niche.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 01_top_templates_by_niche.png")
    plt.close()

# ================= VISUALIZATION 2: DOMINANT TEMPLATE PER NICHE =================
def plot_dominant_templates(data: dict) -> None:
    """
    Horizontal bar chart showing the dominant (most common) template for each niche.
    """
    df = data["dominant"].copy()
    df = df.sort_values("video_count", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.3)))
    
    bars = ax.barh(range(len(df)), df["video_count"].values, color=sns.color_palette("viridis", len(df)))
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row["video_count"] + 0.1, i, f"{int(row['video_count'])} videos", 
                va="center", fontsize=9)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{niche.split('>')[-1].strip()}\n({template})" 
                         for niche, template in zip(df["niche"], df["template_category"])], 
                        fontsize=9)
    ax.set_xlabel("Number of Videos", fontsize=11, fontweight="bold")
    ax.set_title("Dominant Template Type per Niche", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_dominant_templates.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 02_dominant_templates.png")
    plt.close()

# ================= VISUALIZATION 3: TEMPLATE DIVERSITY HEATMAP =================
def plot_template_diversity_heatmap(data: dict) -> None:
    """
    Create a heatmap showing template popularity across niches.
    """
    df = data["summary"].copy()
    
    # Filter to top niches and templates
    top_niches = df.groupby("niche")["video_count"].sum().nlargest(6).index.tolist()
    top_templates = df.groupby("template_category")["video_count"].sum().nlargest(12).index.tolist()
    
    df_filtered = df[(df["niche"].isin(top_niches)) & (df["template_category"].isin(top_templates))]
    
    # Create pivot table
    pivot = df_filtered.pivot_table(index="template_category", columns="niche", 
                                     values="video_count", fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(pivot, annot=True, fmt="g", cmap="YlOrRd", cbar_kws={"label": "Video Count"},
                ax=ax, linewidths=0.5)
    
    ax.set_xlabel("Niche", fontsize=11, fontweight="bold")
    ax.set_ylabel("Template Category", fontsize=11, fontweight="bold")
    ax.set_title("Template Category Popularity Across Niches (Heatmap)", fontsize=13, fontweight="bold")
    ax.set_xticklabels([label.get_text().replace(" > ", "\n") for label in ax.get_xticklabels()], 
                        rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_template_diversity_heatmap.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 03_template_diversity_heatmap.png")
    plt.close()

# ================= VISUALIZATION 4: CROSS-NICHE TEMPLATES =================
def plot_cross_niche_templates(data: dict) -> None:
    """
    Bar chart showing which templates are used across multiple niches.
    """
    df = data["cross_niche"].copy()
    df = df.nlargest(10, "niche_count").sort_values("total_videos", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.barh(range(len(df)), df["total_videos"].values, color=sns.color_palette("coolwarm", len(df)))
    
    # Color bars by niche count for additional insight
    for i, bar in enumerate(bars):
        niche_count = df.iloc[i]["niche_count"]
        opacity = 0.6 + (niche_count / df["niche_count"].max()) * 0.4
        bar.set_alpha(opacity)
    
    # Add labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row["total_videos"] + 0.1, i, 
                f"{int(row['total_videos'])} videos ({int(row['niche_count'])} niches)", 
                va="center", fontsize=9)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["template_category"].values, fontsize=10)
    ax.set_xlabel("Number of Videos", fontsize=11, fontweight="bold")
    ax.set_title("Top Templates Used Across Multiple Niches", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_cross_niche_templates.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 04_cross_niche_templates.png")
    plt.close()

# ================= VISUALIZATION 5: NICHE VIDEO DISTRIBUTION =================
def plot_niche_distribution(data: dict) -> None:
    """
    Pie chart and bar chart showing video distribution across niches.
    """
    df = data["summary"].copy()
    niche_counts = df.groupby("niche")["video_count"].sum().sort_values(ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pie chart
    colors = sns.color_palette("husl", len(niche_counts))
    wedges, texts, autotexts = ax1.pie(niche_counts.values, labels=None, autopct="%1.1f%%",
                                         colors=colors, startangle=90)
    
    # Create legend with better formatting
    legend_labels = [f"{niche.split('>')[-1].strip()} ({count})" 
                     for niche, count in niche_counts.items()]
    ax1.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax1.set_title("Video Distribution Across Niches", fontsize=12, fontweight="bold")
    
    # Bar chart
    niche_counts.plot(kind="barh", ax=ax2, color=colors)
    ax2.set_xlabel("Number of Videos", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Niche", fontsize=11, fontweight="bold")
    ax2.set_title("Video Count by Niche", fontsize=12, fontweight="bold")
    ax2.set_yticklabels([label.get_text().split(">")[-1].strip() for label in ax2.get_yticklabels()], 
                         fontsize=9)
    
    for i, v in enumerate(niche_counts.values):
        ax2.text(v + 2, i, str(int(v)), va="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_niche_distribution.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 05_niche_distribution.png")
    plt.close()

# ================= VISUALIZATION 6: CONFIDENCE BY TEMPLATE =================
def plot_confidence_analysis(data: dict) -> None:
    """
    Box plot showing confidence distribution by template category.
    """
    df = data["summary"].copy()
    
    # Get top 10 templates
    top_templates = df.groupby("template_category")["video_count"].sum().nlargest(10).index.tolist()
    df_top = df[df["template_category"].isin(top_templates)].copy()
    df_top = df_top.sort_values("avg_confidence", ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar plot with error representation
    x_pos = np.arange(len(df_top))
    colors = sns.color_palette("RdYlGn", len(df_top))
    
    bars = ax.barh(x_pos, df_top["avg_confidence"].values, color=colors)
    
    # Add value labels
    for i, (idx, row) in enumerate(df_top.iterrows()):
        ax.text(row["avg_confidence"] - 0.02, i, f"{row['avg_confidence']:.2f}", 
                va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(df_top["template_category"].values, fontsize=10)
    ax.set_xlabel("Average Confidence Score", fontsize=11, fontweight="bold")
    ax.set_title("Template Confidence Analysis (Top 10 Templates)", fontsize=13, fontweight="bold")
    ax.set_xlim(0.7, 1.0)
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_confidence_analysis.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 06_confidence_analysis.png")
    plt.close()

# ================= VISUALIZATION 7: TEMPLATE USAGE TRENDS =================
def plot_template_usage_summary(data: dict) -> None:
    """
    Summary statistics visualization showing key template metrics.
    """
    df = data["summary"].copy()
    
    # Calculate statistics
    total_videos = df["video_count"].sum()
    total_niches = df["niche"].nunique()
    total_templates = df["template_category"].nunique()
    avg_confidence = df["avg_confidence"].mean()
    
    # Top 5 templates
    top_5_templates = df.groupby("template_category")["video_count"].sum().nlargest(5)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Summary metrics (top left)
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis("off")
    
    summary_text = f"""
    OVERALL STATISTICS
    ─────────────────────────────────
    Total Videos Analyzed:  {total_videos:,}
    Total Niches:           {total_niches}
    Total Template Types:   {total_templates}
    Average Confidence:     {avg_confidence:.3f}
    """
    
    ax_summary.text(0.05, 0.5, summary_text, fontsize=11, family="monospace",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                   verticalalignment="center")
    
    # Top 5 templates
    ax_top5 = fig.add_subplot(gs[1, 0])
    top_5_templates_sorted = top_5_templates.sort_values(ascending=True)
    colors = sns.color_palette("viridis", len(top_5_templates_sorted))
    ax_top5.barh(range(len(top_5_templates_sorted)), top_5_templates_sorted.values, color=colors)
    ax_top5.set_yticks(range(len(top_5_templates_sorted)))
    ax_top5.set_yticklabels(top_5_templates_sorted.index, fontsize=10)
    ax_top5.set_xlabel("Video Count", fontsize=10, fontweight="bold")
    ax_top5.set_title("Top 5 Most Used Templates", fontsize=11, fontweight="bold")
    ax_top5.grid(axis="x", alpha=0.3)
    
    for i, v in enumerate(top_5_templates_sorted.values):
        ax_top5.text(v + 0.3, i, str(int(v)), va="center", fontsize=9)
    
    # Templates per niche
    ax_templates_per_niche = fig.add_subplot(gs[1, 1])
    templates_per_niche = df.groupby("niche")["template_category"].nunique().sort_values(ascending=True)
    colors = sns.color_palette("muted", len(templates_per_niche))
    ax_templates_per_niche.barh(range(len(templates_per_niche)), templates_per_niche.values, color=colors)
    ax_templates_per_niche.set_yticks(range(len(templates_per_niche)))
    ax_templates_per_niche.set_yticklabels([n.split(">")[-1].strip() for n in templates_per_niche.index], fontsize=9)
    ax_templates_per_niche.set_xlabel("Number of Unique Templates", fontsize=10, fontweight="bold")
    ax_templates_per_niche.set_title("Template Variety by Niche", fontsize=11, fontweight="bold")
    ax_templates_per_niche.grid(axis="x", alpha=0.3)
    
    # Confidence distribution
    ax_conf = fig.add_subplot(gs[2, 0])
    ax_conf.hist(df["avg_confidence"].values, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax_conf.axvline(avg_confidence, color="red", linestyle="--", linewidth=2, label=f"Mean: {avg_confidence:.3f}")
    ax_conf.set_xlabel("Confidence Score", fontsize=10, fontweight="bold")
    ax_conf.set_ylabel("Frequency", fontsize=10, fontweight="bold")
    ax_conf.set_title("Distribution of Confidence Scores", fontsize=11, fontweight="bold")
    ax_conf.legend()
    ax_conf.grid(axis="y", alpha=0.3)
    
    # Batch count distribution
    ax_batch = fig.add_subplot(gs[2, 1])
    batch_stats = df.groupby("template_category")["video_count"].sum().nlargest(8).sort_values(ascending=True)
    colors = sns.color_palette("coolwarm", len(batch_stats))
    ax_batch.barh(range(len(batch_stats)), batch_stats.values, color=colors)
    ax_batch.set_yticks(range(len(batch_stats)))
    ax_batch.set_yticklabels(batch_stats.index, fontsize=9)
    ax_batch.set_xlabel("Total Videos Analyzed", fontsize=10, fontweight="bold")
    ax_batch.set_title("Most Analyzed Templates (by Video Count)", fontsize=11, fontweight="bold")
    ax_batch.grid(axis="x", alpha=0.3)
    
    plt.savefig(OUTPUT_DIR / "07_template_summary_dashboard.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: 07_template_summary_dashboard.png")
    plt.close()

# ================= MAIN EXECUTION =================
def main():
    print("\n" + "="*80)
    print("TEMPLATE-NICHE VISUAL INSIGHTS GENERATOR")
    print("="*80 + "\n")
    
    print("Loading analysis data...")
    data = load_data()
    print(f"✓ Loaded data from {ANALYSIS_DIR}")
    
    print("\nGenerating visualizations...\n")
    
    plot_top_templates_by_niche(data)
    plot_dominant_templates(data)
    plot_template_diversity_heatmap(data)
    plot_cross_niche_templates(data)
    plot_niche_distribution(data)
    plot_confidence_analysis(data)
    plot_template_usage_summary(data)
    
    print("\n" + "="*80)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")
    print("Generated files:")
    print("  1. 01_top_templates_by_niche.png       - Stacked templates for top 5 niches")
    print("  2. 02_dominant_templates.png           - Most common template per niche")
    print("  3. 03_template_diversity_heatmap.png   - Cross-niche template popularity")
    print("  4. 04_cross_niche_templates.png        - Templates used across multiple niches")
    print("  5. 05_niche_distribution.png           - Video distribution across niches")
    print("  6. 06_confidence_analysis.png          - Model confidence by template type")
    print("  7. 07_template_summary_dashboard.png   - Overall statistics & metrics")
    print("\n")

if __name__ == "__main__":
    main()
