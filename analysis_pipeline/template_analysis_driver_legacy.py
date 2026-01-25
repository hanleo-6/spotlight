from pathlib import Path
import json
from template_analyser import VideoTemplateAnalyser


def load_templates(base_dir: Path) -> list:
    templates_path = base_dir / "data/output/templates"
    templates = []
    for file in templates_path.glob("*.json"):
        with open(file) as f:
            templates.append(json.load(f))
    if not templates:
        raise FileNotFoundError(f"No template JSON found in {templates_path}")
    return templates

def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data/output"
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = load_templates(project_root)
    analyser = VideoTemplateAnalyser(templates)

    analyser.analyze_by_niche()
    analyser.identify_template_clusters(n_clusters=3)
    analyser.plot_niche_comparison(output_dir / "niche_comparison.png")

    # Use first available niche for fingerprint plot
    default_niche = analyser.df['niche'].iloc[0]
    analyser.plot_template_fingerprint(default_niche, output_dir / f"fingerprint_{default_niche}.png")

    analyser.generate_recommendation('tech_saas')
    analyser.export_insights(output_dir / "insights.json")


if __name__ == "__main__":
    main()
