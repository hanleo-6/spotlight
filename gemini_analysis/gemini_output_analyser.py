import json
from pathlib import Path
import pandas as pd
import logging
from collections import Counter
from niche_deduplicator import deduplicate_niches, apply_niche_mapping, get_niche_stats

# ================= CONFIG =================
ANALYSIS_DIR = Path("data/output/gemini_analysis")
OUTPUT_DIR = Path("data/output/analysis_reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NICHE_CACHE_FILE = Path("data/output/niche_deduplication_cache.json")  # Cache for niche deduplication
USERNAME_NICHE_MAPPING_FILE = Path("data/username_niche_mapping.json")  # Optional mapping file

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_username_niche_mapping() -> dict:
    """Load username-to-niche mapping from file if it exists."""
    if USERNAME_NICHE_MAPPING_FILE.exists():
        logger.info(f"Loading username-niche mapping from {USERNAME_NICHE_MAPPING_FILE}")
        with open(USERNAME_NICHE_MAPPING_FILE, "r") as f:
            return json.load(f)
    else:
        logger.warning(f"No username-niche mapping file found at {USERNAME_NICHE_MAPPING_FILE}")
        logger.warning("Using usernames as niches. Create a mapping file to use actual niches.")
        return {}

# ================= FUNCTIONS =================
def parse_gemini_output(gemini_output: dict) -> dict:
    """Parse Gemini output - handles both old and new template formats."""
    # New format: check for template_identity (new structure)
    if isinstance(gemini_output, dict) and "template_identity" in gemini_output:
        return gemini_output
    
    # Old format: check for template_type (legacy structure)
    if isinstance(gemini_output, dict) and "template_type" in gemini_output:
        return gemini_output
    
    # If there's raw_text, parse it
    raw_text = gemini_output.get("raw_text", "")
    if raw_text:
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Gemini output: {clean_text[:100]}...")
    
    return {}

def extract_analysis_from_file(json_path: Path) -> list:
    """Extract per-VIDEO analysis from new template format."""
    with open(json_path, "r") as f:
        data = json.load(f)

    video_id = data.get("video_id")
    gcs_path = data.get("gcs_input_path", "")
    username = data.get("username", "unknown")
    
    # NEW FORMAT: Single template_data object per video
    if "template_data" in data:
        template = data["template_data"]
        
        # Skip parse errors
        if "parse_error" in template:
            logger.warning(f"Skipping {video_id} - parse error in template data")
            return []
        
        # Extract from new nested structure
        identity = template.get("template_identity", {})
        niche_targeting = template.get("niche_targeting", {})
        parametric = template.get("parametric_design", {})
        blueprint = template.get("replication_blueprint", {})
        metadata = template.get("template_metadata", {})
        structure = template.get("structure", {})
        visual = template.get("visual_system", {})
        content = template.get("content_formula", {})
        
        result = {
            "video_id": video_id,
            "gcs_path": gcs_path,
            "username": username,
            
            # Core identification
            "template_name": identity.get("template_name", "Unknown"),
            "template_category": identity.get("template_category", "Unknown"),
            "visual_fingerprint": identity.get("visual_fingerprint", ""),
            "personality": identity.get("personality", "Unknown"),
            
            # Niche targeting
            "niche": niche_targeting.get("primary_niche", "Unknown"),
            "secondary_niches": ", ".join(niche_targeting.get("secondary_niches", [])),
            "niche_fit_reasoning": niche_targeting.get("template_niche_fit_reasoning", ""),
            "confidence": niche_targeting.get("confidence_score", 0.0),
            "audience_psychology": niche_targeting.get("audience_psychology", ""),
            "trust_signals": ", ".join(niche_targeting.get("trust_signals", [])),
            
            # Parametric design
            "fixed_elements": ", ".join(parametric.get("fixed_elements", [])),
            "variable_elements": ", ".join(parametric.get("variable_elements", [])),
            "customization_boundaries": parametric.get("customization_boundaries", ""),
            
            # Replication
            "complexity_level": blueprint.get("complexity_level", "Unknown"),
            "automation_potential": blueprint.get("automation_potential", "Unknown"),
            "software_category": blueprint.get("software_category", "Unknown"),
            "required_skills": ", ".join(blueprint.get("key_skills_needed", [])),
            
            # Metadata
            "versatility": metadata.get("versatility", "Unknown"),
            "scalability": metadata.get("scalability", "Unknown"),
            "search_keywords": ", ".join(metadata.get("search_keywords", [])),
            
            # Structure & Style
            "hook_strategy": structure.get("hook_strategy", ""),
            "pacing_rhythm": structure.get("pacing_rhythm", "Unknown"),
            "primary_format": content.get("primary_format", "Unknown"),
            "storytelling_structure": content.get("storytelling_structure", "Unknown"),
            "information_density": content.get("information_density", "Unknown"),
            
            # Visual system basics
            "grid_system": visual.get("layout", {}).get("grid_system", "Unknown"),
            "graphic_style": visual.get("graphic_style", "Unknown")
        }
        
        return [result]
    
    # OLD FORMAT: analysis_batches (legacy support)
    batches = data.get("analysis_batches", [])
    if not batches:
        return []
    
    # Legacy aggregation logic
    niches = []
    templates = []
    confidences = []
    
    for batch in batches:
        gemini_output = batch.get("gemini_output", {})
        parsed = parse_gemini_output(gemini_output)
        
        if parsed:
            niches.append(parsed.get("niche", "Unknown"))
            templates.append(parsed.get("template_type", "Unknown"))
            confidences.append(parsed.get("confidence", 0.0))
    
    if not niches:
        return []
    
    def most_common_or_first(items):
        if not items:
            return "Unknown"
        counter = Counter(items)
        most_common = counter.most_common(1)[0]
        if most_common[1] == 1 and len(counter) == len(items):
            return items[0]
        return most_common[0]
    
    result = {
        "video_id": video_id,
        "gcs_path": gcs_path,
        "username": username,
        "template_name": "Legacy Format",
        "template_category": most_common_or_first(templates),
        "niche": most_common_or_first(niches),
        "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "complexity_level": "Unknown",
        "versatility": "Unknown",
        "automation_potential": "Unknown"
    }
    
    return [result]

def map_niche_from_path(gcs_path: str, username_niche_map: dict) -> str:
    """Extract username from GCS path and map to niche."""
    try:
        parts = gcs_path.split("/")
        username = parts[1]  # username is after gemini_inputs/
        
        # Use mapping if available, otherwise use username as niche
        return username_niche_map.get(username, username)
    except IndexError:
        return "unknown"

# ================= MAIN SCRIPT =================
def main():
    # Load username-to-niche mapping (optional, for fallback only)
    username_niche_map = load_username_niche_mapping()
    
    all_json_files = list(ANALYSIS_DIR.rglob("*.json"))
    if not all_json_files:
        logger.error(f"No analysis JSON files found in {ANALYSIS_DIR}")
        return

    logger.info(f"Found {len(all_json_files)} analysis files. Processing...")

    rows = []
    for file_path in all_json_files:
        rows.extend(extract_analysis_from_file(file_path))

    if not rows:
        logger.error("No valid analysis data extracted.")
        return

    df = pd.DataFrame(rows)
    
    # No longer need to extract username from path - it's in the data
    if "username" not in df.columns:
        df["username"] = df["gcs_path"].apply(lambda p: p.split("/")[1] if len(p.split("/")) > 1 else "unknown")

    logger.info(f"Extracted {len(df)} analysis records from {df['video_id'].nunique()} videos")
    logger.info(f"Found {df['niche'].nunique()} unique niches before deduplication")

    # ========== NICHE DEDUPLICATION ==========
    logger.info("\n" + "="*80)
    logger.info("STARTING NICHE DEDUPLICATION WITH GEMINI")
    logger.info("="*80)
    
    unique_niches = df["niche"].unique().tolist()
    niche_groups = deduplicate_niches(unique_niches, batch_size=50, cache_file=str(NICHE_CACHE_FILE))
    df = apply_niche_mapping(df, niche_groups, niche_column="niche")
    
    niche_stats = get_niche_stats(df, canonical_niche_column="niche_canonical")
    logger.info(f"\n✓ Deduplication Results:")
    logger.info(f"  - Before: {niche_stats['total_videos']} videos across {len(unique_niches)} niches")
    logger.info(f"  - After: {niche_stats['total_videos']} videos across {niche_stats['total_canonical_niches']} canonical niches")
    logger.info(f"  - Videos per niche: min={niche_stats['videos_per_niche']['min']}, "
                f"max={niche_stats['videos_per_niche']['max']}, "
                f"mean={niche_stats['videos_per_niche']['mean']:.1f}")
    
    # Use canonical niches for all subsequent analysis
    niche_col = "niche_canonical"
    
    logger.info("\n" + "="*80)
    logger.info("GENERATING ANALYSIS REPORTS")
    logger.info("="*80 + "\n")

    # ========== 1. TEMPLATE-NICHE SUMMARY ==========
    template_niche = df.groupby([niche_col, "template_category"]).agg(
        video_count=("video_id", "nunique"),
        avg_confidence=("confidence", "mean")
    ).reset_index().sort_values([niche_col, "video_count"], ascending=[True, False])
    
    template_niche.rename(columns={niche_col: "niche"}, inplace=True)
    template_niche.to_csv(OUTPUT_DIR / "template_niche_summary.csv", index=False)
    logger.info(f"✓ Template-niche summary saved ({len(template_niche)} rows)")

    # ========== 2. DOMINANT TEMPLATES PER NICHE ==========
    dominant_templates = template_niche.loc[template_niche.groupby("niche")["video_count"].idxmax()]
    dominant_templates = dominant_templates.sort_values("video_count", ascending=False)
    dominant_templates.to_csv(OUTPUT_DIR / "dominant_templates_by_niche.csv", index=False)
    logger.info(f"✓ Dominant templates per niche saved ({len(dominant_templates)} niches)")

    # ========== 3. CROSS-NICHE TEMPLATES ==========
    template_popularity = df.groupby("template_category").agg(
        niche_count=(niche_col, "nunique"),
        total_videos=("video_id", "nunique"),
        avg_confidence=("confidence", "mean")
    ).reset_index().sort_values("niche_count", ascending=False)
    
    template_popularity.to_csv(OUTPUT_DIR / "templates_cross_niche.csv", index=False)
    logger.info(f"✓ Cross-niche template analysis saved")

    # ========== 4. PRIMARY FORMAT ANALYSIS ==========
    format_niche = df.groupby([niche_col, "primary_format"]).agg(
        video_count=("video_id", "nunique"),
        avg_confidence=("confidence", "mean")
    ).reset_index().sort_values([niche_col, "video_count"], ascending=[True, False])
    
    format_niche.rename(columns={niche_col: "niche"}, inplace=True)
    format_niche.to_csv(OUTPUT_DIR / "primary_format_by_niche.csv", index=False)
    logger.info(f"✓ Primary format analysis saved")

    # ========== 5. STORYTELLING STRUCTURE PATTERNS ==========
    storytelling_niche = df.groupby([niche_col, "storytelling_structure"]).agg(
        video_count=("video_id", "nunique"),
        avg_confidence=("confidence", "mean")
    ).reset_index().sort_values([niche_col, "video_count"], ascending=[True, False])
    
    storytelling_niche.rename(columns={niche_col: "niche"}, inplace=True)
    storytelling_niche.to_csv(OUTPUT_DIR / "storytelling_structure_by_niche.csv", index=False)
    logger.info(f"✓ Storytelling structure analysis saved")

    # ========== 6. TEMPLATE DIVERSITY PER NICHE ==========
    diversity = df.groupby(niche_col).agg(
        total_videos=("video_id", "nunique"),
        unique_templates=("template_category", "nunique"),
        avg_confidence=("confidence", "mean")
    ).reset_index()
    
    diversity["template_diversity_ratio"] = diversity["unique_templates"] / diversity["total_videos"]
    diversity = diversity.sort_values("template_diversity_ratio", ascending=False)
    diversity.rename(columns={niche_col: "niche"}, inplace=True)
    diversity.to_csv(OUTPUT_DIR / "niche_diversity_metrics.csv", index=False)
    logger.info(f"✓ Template diversity metrics saved")

    # ========== 7. VERSATILITY & COMPLEXITY ANALYSIS (NEW) ==========
    versatility_complexity = df.groupby(["versatility", "complexity_level"]).agg(
        video_count=("video_id", "nunique"),
        avg_confidence=("confidence", "mean")
    ).reset_index().sort_values("video_count", ascending=False)
    
    versatility_complexity.to_csv(OUTPUT_DIR / "versatility_complexity_matrix.csv", index=False)
    logger.info(f"✓ Versatility-complexity matrix saved")

    # ========== 8. AUTOMATION POTENTIAL BY NICHE (NEW) ==========
    automation_by_niche = df.groupby([niche_col, "automation_potential"]).agg(
        video_count=("video_id", "nunique"),
        avg_complexity=("complexity_level", lambda x: x.mode()[0] if len(x) > 0 else "Unknown")
    ).reset_index().sort_values([niche_col, "video_count"], ascending=[True, False])
    
    automation_by_niche.rename(columns={niche_col: "niche"}, inplace=True)
    automation_by_niche.to_csv(OUTPUT_DIR / "automation_potential_by_niche.csv", index=False)
    logger.info(f"✓ Automation potential by niche saved")

    # ========== 9. NICHE FIT REASONING ANALYSIS (NEW) ==========
    niche_fit = df[[niche_col, "template_name", "niche_fit_reasoning", "confidence"]].copy()
    niche_fit = niche_fit[niche_fit["niche_fit_reasoning"] != ""]
    niche_fit.rename(columns={niche_col: "niche"}, inplace=True)
    niche_fit = niche_fit.sort_values("confidence", ascending=False)
    niche_fit.to_csv(OUTPUT_DIR / "niche_template_fit_reasoning.csv", index=False)
    logger.info(f"✓ Niche-template fit reasoning saved ({len(niche_fit)} entries)")

    # ========== 10. TEMPLATE PERSONALITY BY NICHE (NEW) ==========
    personality_niche = df.groupby([niche_col, "personality"]).agg(
        video_count=("video_id", "nunique"),
        avg_confidence=("confidence", "mean")
    ).reset_index().sort_values([niche_col, "video_count"], ascending=[True, False])
    
    personality_niche.rename(columns={niche_col: "niche"}, inplace=True)
    personality_niche.to_csv(OUTPUT_DIR / "personality_by_niche.csv", index=False)
    logger.info(f"✓ Template personality by niche saved")

    # ========== 11. CROSS-NICHE VERSATILITY (NEW) ==========
    # Templates marked as highly-versatile that appear in multiple niches
    versatile_templates = df[df["versatility"].isin(["highly-versatile", "moderately-versatile"])].groupby("template_name").agg(
        niche_count=(niche_col, "nunique"),
        total_videos=("video_id", "nunique"),
        niches_list=(niche_col, lambda x: ", ".join(x.unique()[:5])),
        versatility_level=("versatility", lambda x: x.mode()[0] if len(x) > 0 else "Unknown"),
        avg_confidence=("confidence", "mean")
    ).reset_index().sort_values("niche_count", ascending=False)
    
    versatile_templates.to_csv(OUTPUT_DIR / "versatile_templates_cross_niche.csv", index=False)
    logger.info(f"✓ Versatile templates analysis saved")

    # ========== 12. CONFIDENCE ANALYSIS ==========
    confidence_stats = df.groupby("template_category").agg(
        video_count=("video_id", "nunique"),
        avg_confidence=("confidence", "mean"),
        min_confidence=("confidence", "min"),
        max_confidence=("confidence", "max"),
        std_confidence=("confidence", "std")
    ).reset_index().sort_values("avg_confidence", ascending=False)
    
    confidence_stats.to_csv(OUTPUT_DIR / "template_confidence_analysis.csv", index=False)
    logger.info(f"✓ Confidence analysis saved")

    # ========== 13. DETAILED SUMMARY REPORT ==========
    # ========== 13. DETAILED SUMMARY REPORT ==========
    report = []
    report.append("=" * 80)
    report.append("ADVANCED TEMPLATE-NICHE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal Videos Analyzed: {df['video_id'].nunique()}")
    report.append(f"Total Canonical Niches: {df[niche_col].nunique()}")
    report.append(f"Total Template Categories: {df['template_category'].nunique()}")
    report.append(f"Unique Template Names: {df['template_name'].nunique()}")
    
    report.append("\n" + "=" * 80)
    report.append("TOP 10 MOST POPULAR TEMPLATE CATEGORIES (Cross-Niche)")
    report.append("=" * 80)
    for _, row in template_popularity.head(10).iterrows():
        report.append(f"{row['template_category']}: {row['total_videos']} videos across {row['niche_count']} niches (conf: {row['avg_confidence']:.2f})")
    
    report.append("\n" + "=" * 80)
    report.append("TEMPLATE VERSATILITY & COMPLEXITY")
    report.append("=" * 80)
    for _, row in versatility_complexity.head(10).iterrows():
        report.append(f"{row['versatility']} + {row['complexity_level']}: {row['video_count']} videos (conf: {row['avg_confidence']:.2f})")
    
    report.append("\n" + "=" * 80)
    report.append("TOP VERSATILE TEMPLATES ACROSS NICHES")
    report.append("=" * 80)
    for _, row in versatile_templates.head(10).iterrows():
        report.append(f"{row['template_name']}: {row['niche_count']} niches, {row['total_videos']} videos - {row['niches_list']}")
    
    report.append("\n" + "=" * 80)
    report.append("TOP 10 NICHES WITH HIGHEST TEMPLATE DIVERSITY")
    report.append("=" * 80)
    for _, row in diversity.head(10).iterrows():
        report.append(f"{row['niche']}: {row['unique_templates']} templates / {row['total_videos']} videos (ratio: {row['template_diversity_ratio']:.2f})")
    
    report.append("\n" + "=" * 80)
    report.append("TOP 15 NICHES BY VIDEO COUNT")
    report.append("=" * 80)
    top_niches = df.groupby(niche_col)["video_id"].nunique().sort_values(ascending=False).head(15)
    for niche, count in top_niches.items():
        report.append(f"{niche}: {count} videos")
    
    report.append("\n" + "=" * 80)
    report.append("TEMPLATE PERSONALITY DISTRIBUTION")
    report.append("=" * 80)
    personality_dist = df["personality"].value_counts().head(10)
    for personality, count in personality_dist.items():
        report.append(f"{personality}: {count} videos")
    
    report_text = "\n".join(report)
    with open(OUTPUT_DIR / "analysis_report.txt", "w") as f:
        f.write(report_text)
    
    logger.info("\n" + report_text)
    logger.info(f"\n✓ All analysis reports saved to {OUTPUT_DIR}")
    logger.info(f"\nGenerated files:")
    logger.info(f"  1. template_niche_summary.csv - Template usage by niche")
    logger.info(f"  2. dominant_templates_by_niche.csv - Most popular template per niche")
    logger.info(f"  3. templates_cross_niche.csv - Templates used across multiple niches")
    logger.info(f"  4. primary_format_by_niche.csv - Primary formats per niche")
    logger.info(f"  5. storytelling_structure_by_niche.csv - Storytelling structures per niche")
    logger.info(f"  6. niche_diversity_metrics.csv - Template diversity metrics")
    logger.info(f"  7. versatility_complexity_matrix.csv - Versatility vs complexity")
    logger.info(f"  8. automation_potential_by_niche.csv - Automation potential analysis")
    logger.info(f"  9. niche_template_fit_reasoning.csv - Why templates fit specific niches")
    logger.info(f"  10. personality_by_niche.csv - Template personalities by niche")
    logger.info(f"  11. versatile_templates_cross_niche.csv - Versatile templates analysis")
    logger.info(f"  12. template_confidence_analysis.csv - Confidence statistics")
    logger.info(f"  13. analysis_report.txt - Human-readable summary report")
    logger.info(f"  14. niche_deduplication_cache.json - Cached niche groupings")

if __name__ == "__main__":
    main()
