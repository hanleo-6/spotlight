#!/usr/bin/env python3
"""
Video Analysis Pipeline Driver

A unified driver for running the complete video analysis pipeline on any video dataset.
This pipeline consists of three ordered modules:
  1. Classification - Classify videos by niche
  2. Template Extraction - Extract video template profiles
  3. Template Analysis - Analyze patterns and generate insights

Usage:
    python pipeline_driver.py --video-dir <path> --features-dir <path> --output-dir <path>
    
Example:
    python pipeline_driver.py --video-dir best_media --features-dir features
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Import pipeline modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from classification import VideoClassifier
    from template_extraction import TemplateExtractor
    from template_analysis import TemplateAnalyser
except ImportError:
    # Fallback for direct numbered module names
    import importlib.util
    
    def load_module(file_name, module_name):
        spec = importlib.util.spec_from_file_location(module_name, Path(__file__).parent / file_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    classification_mod = load_module("01_classification.py", "classification")
    extraction_mod = load_module("02_template_extraction.py", "template_extraction")
    analysis_mod = load_module("03_template_analysis.py", "template_analysis")
    
    VideoClassifier = classification_mod.VideoClassifier
    TemplateExtractor = extraction_mod.TemplateExtractor
    TemplateAnalyser = analysis_mod.TemplateAnalyser


class VideoPipelineDriver:
    """Main driver for the video analysis pipeline."""
    
    def __init__(self, workspace_root: Path = None):
        """Initialize pipeline driver."""
        if workspace_root is None:
            workspace_root = Path(__file__).resolve().parent.parent
        
        self.workspace_root = workspace_root
        self.start_time = None
        
    def run_pipeline(self, 
                     video_dir: Path, 
                     features_dir: Path = None,
                     output_dir: Path = None,
                     skip_classification: bool = False,
                     skip_extraction: bool = False,
                     skip_analysis: bool = False):
        """
        Run the complete analysis pipeline.
        
        Args:
            video_dir: Directory containing video files (required)
            features_dir: Directory containing feature JSON files (optional)
            output_dir: Directory for output files (default: data/output)
            skip_classification: Skip classification step
            skip_extraction: Skip template extraction step
            skip_analysis: Skip template analysis step
        """
        self.start_time = datetime.now()
        
        # Set defaults
        if output_dir is None:
            output_dir = self.workspace_root / "data" / "output"
        
        if features_dir is None:
            features_dir = self.workspace_root / "features"
        
        # Ensure paths are absolute
        video_dir = Path(video_dir).resolve()
        features_dir = Path(features_dir).resolve()
        output_dir = Path(output_dir).resolve()
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("VIDEO ANALYSIS PIPELINE")
        print("=" * 80)
        print(f"Workspace: {self.workspace_root}")
        print(f"Video Directory: {video_dir}")
        print(f"Features Directory: {features_dir}")
        print(f"Output Directory: {output_dir}")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Step 1: Classification
        if not skip_classification:
            success = self._run_step_1_classification(video_dir, features_dir, output_dir)
            if not success:
                print("\n❌ Pipeline failed at Step 1: Classification")
                return False
        else:
            print("\n⏭️  Skipping Step 1: Classification")
        
        # Step 2: Template Extraction
        if not skip_extraction:
            success = self._run_step_2_extraction(video_dir, output_dir)
            if not success:
                print("\n❌ Pipeline failed at Step 2: Template Extraction")
                return False
        else:
            print("\n⏭️  Skipping Step 2: Template Extraction")
        
        # Step 3: Template Analysis
        if not skip_analysis:
            success = self._run_step_3_analysis(output_dir)
            if not success:
                print("\n❌ Pipeline failed at Step 3: Template Analysis")
                return False
        else:
            print("\n⏭️  Skipping Step 3: Template Analysis")
        
        # Pipeline complete
        elapsed = datetime.now() - self.start_time
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Total Time: {elapsed}")
        print(f"Output Location: {output_dir}")
        print("\nGenerated Files:")
        print(f"  • video_classifications.csv - Video niche classifications")
        print(f"  • templates/*.json - Individual video template profiles")
        print(f"  • template_insights.json - Aggregated insights by niche")
        print(f"  • niche_comparison.png - Visual comparison of niches")
        print("=" * 80)
        
        return True
    
    def _run_step_1_classification(self, video_dir: Path, features_dir: Path, output_dir: Path) -> bool:
        """Run Step 1: Video Classification."""
        print("\n" + "=" * 80)
        print("STEP 1: VIDEO CLASSIFICATION")
        print("=" * 80)
        
        try:
            classifier = VideoClassifier(self.workspace_root)
            classifications_csv = output_dir / "video_classifications.csv"
            
            classifier.classify_videos(
                video_dir=video_dir,
                features_dir=features_dir,
                output_csv=classifications_csv
            )
            
            print(f"\n✅ Step 1 Complete")
            return True
            
        except Exception as e:
            print(f"\n❌ Step 1 Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_step_2_extraction(self, video_dir: Path, output_dir: Path) -> bool:
        """Run Step 2: Template Extraction."""
        print("\n" + "=" * 80)
        print("STEP 2: TEMPLATE EXTRACTION")
        print("=" * 80)
        
        try:
            extractor = TemplateExtractor(self.workspace_root)
            classifications_csv = output_dir / "video_classifications.csv"
            templates_dir = output_dir / "templates"
            
            extractor.extract_templates_batch(
                video_dir=video_dir,
                classifications_csv=classifications_csv,
                output_dir=templates_dir
            )
            
            print(f"\n✅ Step 2 Complete")
            return True
            
        except Exception as e:
            print(f"\n❌ Step 2 Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_step_3_analysis(self, output_dir: Path) -> bool:
        """Run Step 3: Template Analysis."""
        print("\n" + "=" * 80)
        print("STEP 3: TEMPLATE ANALYSIS")
        print("=" * 80)
        
        try:
            templates_dir = output_dir / "templates"
            
            analyser = TemplateAnalyser(templates_dir=templates_dir, workspace_root=self.workspace_root)
            analyser.load_templates()
            
            if len(analyser.templates) == 0:
                print("⚠️  No templates found. Skipping analysis.")
                return True
            
            # Run all analyses
            analyser.analyze_by_niche()
            
            if len(analyser.df) > 3:  # Only cluster if we have enough data
                analyser.identify_template_clusters(n_clusters=min(3, len(analyser.df)))
            
            analyser.plot_niche_comparison(output_dir / "niche_comparison.png")
            analyser.export_insights(output_dir / "template_insights.json")
            
            # Generate recommendation for most common niche
            if len(analyser.df) > 0:
                from collections import Counter
                most_common_niche = Counter(analyser.df['niche']).most_common(1)[0][0]
                analyser.generate_recommendation(most_common_niche)
            
            print(f"\n✅ Step 3 Complete")
            return True
            
        except Exception as e:
            print(f"\n❌ Step 3 Failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the video analysis pipeline on any video dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on best_media folder
  python pipeline_driver.py --video-dir best_media

  # Run with custom output directory
  python pipeline_driver.py --video-dir my_videos --output-dir results

  # Skip classification (if already done)
  python pipeline_driver.py --video-dir best_media --skip-classification
  
  # Run only analysis step
  python pipeline_driver.py --video-dir best_media --skip-classification --skip-extraction
        """
    )
    
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing video files (required)"
    )
    
    parser.add_argument(
        "--features-dir",
        type=str,
        help="Directory containing feature JSON files (default: features/)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for output files (default: data/output/)"
    )
    
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip the classification step"
    )
    
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip the template extraction step"
    )
    
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the template analysis step"
    )
    
    args = parser.parse_args()
    
    # Create driver and run pipeline
    driver = VideoPipelineDriver()
    
    success = driver.run_pipeline(
        video_dir=Path(args.video_dir),
        features_dir=Path(args.features_dir) if args.features_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        skip_classification=args.skip_classification,
        skip_extraction=args.skip_extraction,
        skip_analysis=args.skip_analysis
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
