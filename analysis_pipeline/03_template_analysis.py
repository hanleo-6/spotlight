"""
Module 3: Template Analysis
Analyzes template profiles to identify patterns, clusters, and generate recommendations.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path


class TemplateAnalyser:
    """Analyzes video template profiles to identify patterns by niche."""
    
    def __init__(self, templates_dir: Path = None, workspace_root: Path = None):
        """
        Initialize with templates directory.
        
        Args:
            templates_dir: Directory containing template JSON files
            workspace_root: Workspace root directory
        """
        if workspace_root is None:
            workspace_root = Path(__file__).resolve().parent.parent
        
        if templates_dir is None:
            templates_dir = workspace_root / "data" / "output" / "templates"
        
        self.workspace_root = workspace_root
        self.templates_dir = templates_dir
        self.templates = []
        self.df = None
        
    def load_templates(self):
        """Load all template JSON files from directory."""
        print(f"Loading templates from: {self.templates_dir}")
        
        for json_file in sorted(self.templates_dir.glob("*_template.json")):
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    self.templates.append(json.load(f))
            except Exception as e:
                print(f"  Error loading {json_file.name}: {e}")
        
        print(f"Loaded {len(self.templates)} templates")
        
        if self.templates:
            self.df = self._flatten_templates()
            print(f"Created DataFrame with {len(self.df)} rows")
        
        return self.templates
    
    def _flatten_templates(self):
        """Flatten nested JSON structure into pandas DataFrame."""
        flattened = []
        
        for template in self.templates:
            scenes = template.get('scenes', {})
            text_overlays = template.get('text_overlays', {})
            audio = template.get('audio', {})
            visual = template.get('visual', {})
            
            flat = {
                'video_id': template.get('video_id'),
                'filename': template.get('filename'),
                'duration': template.get('duration', 0),
                'niche': template.get('niche', 'unknown'),
                
                # Scene features
                'total_scenes': scenes.get('total_scenes', 0),
                'avg_scene_length': scenes.get('avg_scene_length', 0),
                'median_scene_length': scenes.get('median_scene_length', 0),
                'cuts_per_minute': (scenes.get('total_scenes', 0) / template.get('duration', 1)) * 60,
                
                # Text overlay features
                'total_text_detections': text_overlays.get('total_detections', 0),
                'text_coverage_percent': text_overlays.get('text_coverage_percent', 0),
                'overlay_timing_pattern': text_overlays.get('overlay_timing_pattern', 'unknown'),
                'common_zones': ','.join(text_overlays.get('common_zones', [])),
                'primary_text_zone': text_overlays.get('common_zones', ['none'])[0] if text_overlays.get('common_zones') else 'none',
                'overlay_density': text_overlays.get('total_detections', 0) / template.get('duration', 1) if template.get('duration', 0) > 0 else 0,
                
                # Audio features
                'word_count': audio.get('word_count', 0),
                'speaking_time_seconds': audio.get('speaking_time_seconds', 0),
                'speech_pace_wpm': (audio.get('word_count', 0) / audio.get('speaking_time_seconds', 1)) * 60 if audio.get('speaking_time_seconds', 0) > 0 else 0,
                
                # Visual features
                'avg_brightness': visual.get('avg_brightness', 0),
                'brightness_variance': visual.get('brightness_variance', 0),
            }
            
            flattened.append(flat)
        
        return pd.DataFrame(flattened)
    
    def analyze_by_niche(self):
        """Generate summary statistics grouped by niche."""
        if self.df is None or len(self.df) == 0:
            print("No data to analyze. Load templates first.")
            return
        
        print("\n" + "=" * 80)
        print("NICHE-LEVEL ANALYSIS")
        print("=" * 80)
        
        for niche, niche_data in self.df.groupby('niche'):
            mean_vals = niche_data.mean(numeric_only=True)
            
            print(f"\n📊 NICHE: {niche.upper()} (n={len(niche_data)})")
            print("-" * 80)
            
            print("\n⚡ Pacing:")
            print(f"  • Avg Cuts/Min: {mean_vals['cuts_per_minute']:.1f}")
            print(f"  • Avg Scene Duration: {mean_vals['avg_scene_length']:.1f}s")
            print(f"  • Total Scenes: {mean_vals['total_scenes']:.1f}")
            
            print("\n📝 Text Overlays:")
            print(f"  • Avg Detections: {mean_vals['total_text_detections']:.1f}")
            print(f"  • Coverage: {mean_vals['text_coverage_percent']:.1f}%")
            print(f"  • Density: {mean_vals['overlay_density']:.1f} overlays/sec")
            
            primary_zones = Counter(niche_data['primary_text_zone'])
            if primary_zones:
                zone, count = primary_zones.most_common(1)[0]
                print(f"  • Primary Zone: {zone} ({count}/{len(niche_data)})")
            
            print("\n🎤 Audio:")
            print(f"  • Avg Word Count: {mean_vals['word_count']:.0f}")
            print(f"  • Speaking Time: {mean_vals['speaking_time_seconds']:.1f}s")
            print(f"  • Speech Pace: {mean_vals['speech_pace_wpm']:.0f} WPM")
            
            print("\n🎨 Visual:")
            print(f"  • Avg Brightness: {mean_vals['avg_brightness']:.1f}")
            print(f"  • Brightness Variance: {mean_vals['brightness_variance']:.1f}")
            
            print("\n⏱️  Duration:")
            print(f"  • Avg: {mean_vals['duration']:.1f}s")
            print(f"  • Range: {niche_data['duration'].min():.1f}s - {niche_data['duration'].max():.1f}s")
    
    def identify_template_clusters(self, n_clusters=5):
        """Identify common template patterns using clustering."""
        if self.df is None or len(self.df) == 0:
            print("No data to cluster. Load templates first.")
            return
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        # Select numeric features for clustering
        features = [
            'cuts_per_minute', 'avg_scene_length', 'overlay_density',
            'text_coverage_percent', 'speech_pace_wpm', 'word_count'
        ]
        
        X = self.df[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42, n_init='auto')
        self.df['cluster'] = kmeans.fit_predict(X_scaled)
        
        print("\n" + "=" * 80)
        print("TEMPLATE CLUSTERS IDENTIFIED")
        print("=" * 80)
        
        for cluster_id in range(min(n_clusters, len(X))):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            print(f"\n🎯 CLUSTER {cluster_id + 1} (n={len(cluster_data)})")
            print("-" * 80)
            
            print(f"Avg Cuts/Min: {cluster_data['cuts_per_minute'].mean():.1f}")
            print(f"Avg Overlay Density: {cluster_data['overlay_density'].mean():.1f}")
            print(f"Avg Text Coverage: {cluster_data['text_coverage_percent'].mean():.1f}%")
            print(f"Avg Speech Pace: {cluster_data['speech_pace_wpm'].mean():.0f} WPM")
            
            # Show which niches use this template
            niche_dist = Counter(cluster_data['niche'])
            print(f"Niches using this template:")
            for niche, count in niche_dist.most_common(3):
                print(f"  • {niche}: {count}")
    
    def plot_niche_comparison(self, save_path: Path):
        """Create comprehensive visualization comparing niches."""
        if self.df is None or len(self.df) == 0:
            print("No data to plot. Load templates first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Template Patterns by Niche', fontsize=16, fontweight='bold')
        
        # 1. Pacing: Cuts per Minute
        ax = axes[0, 0]
        self.df.boxplot(column='cuts_per_minute', by='niche', ax=ax)
        ax.set_title('Pacing (Cuts per Minute) by Niche', fontweight='bold')
        ax.set_xlabel('Niche')
        ax.set_ylabel('Cuts per Minute')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.suptitle('')
        
        # 2. Text Overlay Density
        ax = axes[0, 1]
        self.df.boxplot(column='overlay_density', by='niche', ax=ax)
        ax.set_title('Text Overlay Density by Niche', fontweight='bold')
        ax.set_xlabel('Niche')
        ax.set_ylabel('Overlays per Second')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.suptitle('')
        
        # 3. Speech Pace
        ax = axes[0, 2]
        self.df.boxplot(column='speech_pace_wpm', by='niche', ax=ax)
        ax.set_title('Speech Pace (WPM) by Niche', fontweight='bold')
        ax.set_xlabel('Niche')
        ax.set_ylabel('Words per Minute')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.suptitle('')
        
        # 4. Text Coverage
        ax = axes[1, 0]
        self.df.boxplot(column='text_coverage_percent', by='niche', ax=ax)
        ax.set_title('Text Coverage % by Niche', fontweight='bold')
        ax.set_xlabel('Niche')
        ax.set_ylabel('Coverage (%)')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.suptitle('')
        
        # 5. Duration Distribution
        ax = axes[1, 1]
        for niche in self.df['niche'].unique():
            niche_data = self.df[self.df['niche'] == niche]['duration']
            ax.hist(niche_data, alpha=0.6, label=niche, bins=10)
        ax.set_title('Video Duration Distribution', fontweight='bold')
        ax.set_xlabel('Duration (seconds)')
        ax.set_ylabel('Count')
        ax.legend()
        
        # 6. Scene Length vs Cuts
        ax = axes[1, 2]
        for niche in self.df['niche'].unique():
            niche_data = self.df[self.df['niche'] == niche]
            ax.scatter(niche_data['avg_scene_length'], niche_data['cuts_per_minute'], 
                      alpha=0.6, label=niche, s=50)
        ax.set_title('Scene Length vs Cuts per Minute', fontweight='bold')
        ax.set_xlabel('Avg Scene Length (s)')
        ax.set_ylabel('Cuts per Minute')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {save_path}")
        plt.close()
    
    def generate_recommendation(self, user_niche: str):
        """Generate template recommendations for a user's niche."""
        if self.df is None or len(self.df) == 0:
            print("No data for recommendations. Load templates first.")
            return None
        
        print("\n" + "=" * 80)
        print(f"TEMPLATE RECOMMENDATIONS FOR: {user_niche.upper()}")
        print("=" * 80)
        
        # Find data for this niche
        if user_niche in self.df['niche'].values:
            niche_data = self.df[self.df['niche'] == user_niche]
        else:
            print(f"⚠️  No exact match for '{user_niche}'. Using all data.")
            niche_data = self.df
        
        if len(niche_data) == 0:
            print("No data available for recommendations.")
            return None
        
        # Calculate template signature
        template = {
            'niche': user_niche,
            'sample_size': len(niche_data),
            'avg_cuts_per_minute': niche_data['cuts_per_minute'].mean(),
            'avg_scene_length': niche_data['avg_scene_length'].mean(),
            'avg_overlay_density': niche_data['overlay_density'].mean(),
            'avg_text_coverage': niche_data['text_coverage_percent'].mean(),
            'primary_text_zone': niche_data['primary_text_zone'].mode()[0] if len(niche_data) > 0 else 'middle',
            'avg_duration': niche_data['duration'].mean(),
            'speech_pace': niche_data['speech_pace_wpm'].mean(),
            'avg_word_count': niche_data['word_count'].mean(),
        }
        
        print(f"\n📋 RECOMMENDED TEMPLATE:")
        print("-" * 80)
        print(f"Based on analysis of {template['sample_size']} videos")
        
        print(f"\n⚡ PACING:")
        print(f"  • Target: {template['avg_cuts_per_minute']:.1f} cuts per minute")
        print(f"  • Avg Scene Length: {template['avg_scene_length']:.1f}s")
        if template['avg_cuts_per_minute'] < 10:
            print(f"  → This is SLOW pacing (good for tutorials/demos)")
        elif template['avg_cuts_per_minute'] < 20:
            print(f"  → This is MODERATE pacing (good for explainers)")
        else:
            print(f"  → This is FAST pacing (good for social media)")
        
        print(f"\n📝 TEXT OVERLAYS:")
        print(f"  • Density: {template['avg_overlay_density']:.1f} overlays per second")
        print(f"  • Coverage: {template['avg_text_coverage']:.1f}% of video")
        print(f"  • Primary position: {template['primary_text_zone']}")
        
        print(f"\n🎤 AUDIO/NARRATION:")
        print(f"  • Speech Pace: {template['speech_pace']:.0f} words per minute")
        print(f"  • Target Word Count: ~{template['avg_word_count']:.0f} words")
        
        print(f"\n⏱️  VIDEO LENGTH:")
        print(f"  • Target Duration: {template['avg_duration']:.0f}s")
        
        return template
    
    def export_insights(self, output_file: Path):
        """Export all insights to JSON for use in recommendation engine."""
        if self.df is None or len(self.df) == 0:
            print("No data to export. Load templates first.")
            return
        
        insights = {}
        
        for niche in self.df['niche'].unique():
            niche_data = self.df[self.df['niche'] == niche]
            
            insights[niche] = {
                'sample_size': len(niche_data),
                'template_profile': {
                    'pacing': {
                        'avg_cuts_per_minute': float(niche_data['cuts_per_minute'].mean()),
                        'avg_scene_length': float(niche_data['avg_scene_length'].mean()),
                        'median_scene_length': float(niche_data['median_scene_length'].mean()),
                        'category': self._categorize_pacing(niche_data['cuts_per_minute'].mean())
                    },
                    'text_overlays': {
                        'avg_density': float(niche_data['overlay_density'].mean()),
                        'avg_coverage_percent': float(niche_data['text_coverage_percent'].mean()),
                        'primary_zone': niche_data['primary_text_zone'].mode()[0] if len(niche_data) > 0 else 'none',
                    },
                    'audio': {
                        'avg_word_count': float(niche_data['word_count'].mean()),
                        'avg_speech_pace_wpm': float(niche_data['speech_pace_wpm'].mean()),
                        'avg_speaking_time': float(niche_data['speaking_time_seconds'].mean()),
                    },
                    'visual': {
                        'avg_brightness': float(niche_data['avg_brightness'].mean()),
                        'brightness_variance': float(niche_data['brightness_variance'].mean()),
                    },
                    'duration': {
                        'avg': float(niche_data['duration'].mean()),
                        'median': float(niche_data['duration'].median()),
                        'range': [float(niche_data['duration'].min()), 
                                 float(niche_data['duration'].max())]
                    }
                }
            }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2)
        
        print(f"\n✅ Insights exported to: {output_file}")
    
    def _categorize_pacing(self, cpm):
        """Helper to categorize cuts per minute."""
        if cpm < 10:
            return "slow"
        elif cpm < 20:
            return "moderate"
        elif cpm < 30:
            return "fast"
        else:
            return "very_fast"


def main():
    """Standalone execution for template analysis."""
    workspace_root = Path(__file__).resolve().parent.parent
    output_dir = workspace_root / "data" / "output"
    
    analyser = TemplateAnalyser(workspace_root=workspace_root)
    analyser.load_templates()
    
    if len(analyser.templates) == 0:
        print("No templates found. Run template extraction first.")
        return
    
    # Run analyses
    analyser.analyze_by_niche()
    analyser.identify_template_clusters(n_clusters=3)
    analyser.plot_niche_comparison(output_dir / "niche_comparison.png")
    
    # Generate recommendation for first niche found
    if len(analyser.df) > 0:
        first_niche = analyser.df['niche'].iloc[0]
        analyser.generate_recommendation(first_niche)
    
    analyser.export_insights(output_dir / "template_insights.json")


if __name__ == "__main__":
    main()
