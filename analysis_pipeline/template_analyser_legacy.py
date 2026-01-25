import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class VideoTemplateAnalyser:
    """
    Analyzes video template profiles to identify patterns by niche
    """
    
    def __init__(self, json_files_or_data):
        """
        Initialize with list of JSON file paths or list of dicts
        
        Args:
            json_files_or_data: List of file paths or list of template dicts
        """
        self.templates = []
        
        # Load data
        if isinstance(json_files_or_data[0], str):
            # File paths provided
            for file_path in json_files_or_data:
                with open(file_path, 'r') as f:
                    self.templates.append(json.load(f))
        else:
            # Direct data provided
            self.templates = json_files_or_data
        
        # Convert to DataFrame for easier analysis
        self.df = self._flatten_templates()
        
    def _flatten_templates(self):
        """
        Flatten nested JSON structure into pandas DataFrame
        """
        flattened = []

        for template in self.templates:
            text_overlays = template.get('text_overlays', {})
            overlay_pattern = text_overlays.get('overlay_timing_pattern') or {}
            common_zones = text_overlays.get('common_zones') or []

            flat = {
                'video_id': template.get('video_id'),
                'duration': template.get('duration'),
                'niche': template.get('niche', 'unknown'),

                # Hook features
                'hook_type': template['hook'].get('hook_type'),
                'hook_duration': template['hook'].get('hook_duration'),
                'speech_pace_wpm': template['hook'].get('speech_pace_wpm'),
                'word_count': template['hook'].get('word_count'),
                'has_visual_hook': template['hook'].get('has_visual_hook', False),
                'visual_pattern': template['hook'].get('visual_pattern', 'none'),

                # Pacing features
                'cuts_per_minute': template['pacing'].get('cuts_per_minute'),
                'avg_scene_duration': template['pacing'].get('avg_scene_duration'),
                'scene_duration_variance': template['pacing'].get('scene_duration_variance'),
                'total_scenes': template['pacing'].get('total_scenes'),

                # Text overlay features
                'total_overlays': text_overlays.get('total_overlays'),
                'overlay_density': text_overlays.get('overlay_density'),
                'primary_text_zone': common_zones[0]['zone'] if common_zones else 'none',
                'overlay_timing': overlay_pattern.get('category', 'unknown'),

                # Structure features
                'intro_percentage': template['structure']['intro'].get('percentage'),
                'content_percentage': template['structure']['main_content'].get('percentage'),
                'cta_percentage': template['structure']['cta'].get('percentage'),
            }
            flattened.append(flat)

        return pd.DataFrame(flattened)
    
    def analyze_by_niche(self):
        """
        Generate summary statistics grouped by niche
        """
        print("=" * 80)
        print("NICHE-LEVEL ANALYSIS")
        print("=" * 80)

        for niche, niche_data in self.df.groupby('niche'):
            mean_vals = niche_data.mean(numeric_only=True)

            print(f"\n📊 NICHE: {niche.upper()} (n={len(niche_data)})")
            print("-" * 80)

            hook_types = Counter(niche_data['hook_type'])
            print("\n🎣 Hook Types:")
            for hook_type, count in hook_types.most_common():
                pct = (count / len(niche_data)) * 100
                print(f"  • {hook_type}: {count} ({pct:.1f}%)")

            print("\n⚡ Pacing:")
            print(f"  • Avg Cuts/Min: {mean_vals['cuts_per_minute']:.1f}")
            print(f"  • Avg Scene Duration: {mean_vals['avg_scene_duration']:.1f}s")

            print("\n📝 Text Overlays:")
            print(f"  • Avg Density: {mean_vals['overlay_density']:.1f} overlays/sec")
            primary_zone = Counter(niche_data['primary_text_zone']).most_common(1)[0][0]
            print(f"  • Primary Zone: {primary_zone}")

            print("\n🏗️  Video Structure (avg %):")
            print(f"  • Intro: {mean_vals['intro_percentage']:.1f}%")
            print(f"  • Content: {mean_vals['content_percentage']:.1f}%")
            print(f"  • CTA: {mean_vals['cta_percentage']:.1f}%")

            print("\n⏱️  Duration:")
            print(f"  • Avg: {mean_vals['duration']:.1f}s")
            print(f"  • Range: {niche_data['duration'].min():.1f}s - {niche_data['duration'].max():.1f}s")
    
    def identify_template_clusters(self, n_clusters=5):
        """
        Identify common template patterns using clustering
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        # Select numeric features for clustering
        features = [
            'cuts_per_minute', 'avg_scene_duration', 'overlay_density',
            'intro_percentage', 'content_percentage', 'cta_percentage',
            'speech_pace_wpm'
        ]
        
        X = self.df[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.df['cluster'] = kmeans.fit_predict(X_scaled)
        
        print("\n" + "="*80)
        print("TEMPLATE CLUSTERS IDENTIFIED")
        print("="*80)
        
        for cluster_id in range(n_clusters):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            print(f"\n🎯 CLUSTER {cluster_id + 1} (n={len(cluster_data)})")
            print("-" * 80)
            
            # Characterize this cluster
            print(f"Avg Cuts/Min: {cluster_data['cuts_per_minute'].mean():.1f}")
            print(f"Avg Overlay Density: {cluster_data['overlay_density'].mean():.1f}")
            print(f"Dominant Hook Type: {cluster_data['hook_type'].mode()[0]}")
            print(f"Structure: Intro {cluster_data['intro_percentage'].mean():.0f}% | "
                  f"Content {cluster_data['content_percentage'].mean():.0f}% | "
                  f"CTA {cluster_data['cta_percentage'].mean():.0f}%")
            
            # Show which niches use this template
            niche_dist = Counter(cluster_data['niche'])
            print(f"Niches using this template:")
            for niche, count in niche_dist.most_common(3):
                print(f"  • {niche}: {count}")
    
    def plot_niche_comparison(self, save_path='niche_comparison.png'):
        """
        Create comprehensive visualization comparing niches
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Template Patterns by Niche', fontsize=16, fontweight='bold')

        niches = self.df['niche'].unique()
        colors = sns.color_palette('husl', len(niches))

        # 1. Hook Type Distribution
        ax = axes[0, 0]
        hook_data = pd.crosstab(self.df['niche'], self.df['hook_type'], normalize='index') * 100
        hook_data.plot(kind='bar', stacked=True, ax=ax, legend=True)
        ax.set_title('Hook Type Distribution by Niche', fontweight='bold')
        ax.set_xlabel('Niche')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Hook Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. Pacing: Cuts per Minute
        ax = axes[0, 1]
        self.df.boxplot(column='cuts_per_minute', by='niche', ax=ax)
        ax.set_title('Pacing (Cuts per Minute) by Niche', fontweight='bold')
        ax.set_xlabel('Niche')
        ax.set_ylabel('Cuts per Minute')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.suptitle('')

        # 3. Text Overlay Density
        ax = axes[0, 2]
        self.df.boxplot(column='overlay_density', by='niche', ax=ax)
        ax.set_title('Text Overlay Density by Niche', fontweight='bold')
        ax.set_xlabel('Niche')
        ax.set_ylabel('Overlays per Second')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.suptitle('')

        # 4. Video Structure Comparison
        ax = axes[1, 0]
        structure_cols = ['intro_percentage', 'content_percentage', 'cta_percentage']
        structure_means = self.df.groupby('niche')[structure_cols].mean()
        structure_means.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Video Structure by Niche', fontweight='bold')
        ax.set_xlabel('Niche')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['Intro', 'Content', 'CTA'])
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Duration Distribution
        ax = axes[1, 1]
        for i, niche in enumerate(niches):
            niche_data = self.df[self.df['niche'] == niche]['duration']
            ax.hist(niche_data, alpha=0.6, label=niche, color=colors[i], bins=10)
        ax.set_title('Video Duration Distribution', fontweight='bold')
        ax.set_xlabel('Duration (seconds)')
        ax.set_ylabel('Count')
        ax.legend()

        # 6. Speech Pace
        ax = axes[1, 2]
        self.df.boxplot(column='speech_pace_wpm', by='niche', ax=ax)
        ax.set_title('Speech Pace (WPM) by Niche', fontweight='bold')
        ax.set_xlabel('Niche')
        ax.set_ylabel('Words per Minute')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.suptitle('')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {save_path}")
        plt.show()
    
    def plot_template_fingerprint(self, niche, save_path=None):
        """
        Create a radar chart showing the 'fingerprint' of a niche's template
        """
        niche_data = self.df[self.df['niche'] == niche]
        
        if len(niche_data) == 0:
            print(f"No data found for niche: {niche}")
            return
        
        # Calculate averages for key metrics (normalized 0-100)
        metrics = {
            'Fast Pacing': (niche_data['cuts_per_minute'].mean() / 30) * 100,  # Normalize to 30 CPM max
            'Text Heavy': (niche_data['overlay_density'].mean() / 10) * 100,  # Normalize to 10 overlays/sec
            'Long Intro': niche_data['intro_percentage'].mean(),
            'Short Content': 100 - niche_data['content_percentage'].mean(),
            'Strong CTA': niche_data['cta_percentage'].mean() * 2,  # Amplify for visibility
            'Fast Speech': (niche_data['speech_pace_wpm'].mean() / 300) * 100,  # Normalize to 300 WPM
        }
        
        # Prepare data for radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Close the plot
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label=niche)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(f'Template Fingerprint: {niche.upper()}', 
                     fontweight='bold', size=14, pad=20)
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Fingerprint saved to: {save_path}")
        
        plt.show()
    
    def generate_recommendation(self, user_niche, top_n=3):
        """
        Generate template recommendations for a user's niche
        """
        print("\n" + "="*80)
        print(f"TEMPLATE RECOMMENDATIONS FOR: {user_niche.upper()}")
        print("="*80)
        
        # Find similar niches (for demo, using exact match or 'all')
        if user_niche in self.df['niche'].values:
            niche_data = self.df[self.df['niche'] == user_niche]
        else:
            print(f"⚠️  No exact match for '{user_niche}'. Using all data.")
            niche_data = self.df
        
        if len(niche_data) == 0:
            print("No data available for recommendations.")
            return
        
        # Calculate template signature
        template = {
            'hook_type': niche_data['hook_type'].mode()[0],
            'avg_cuts_per_minute': niche_data['cuts_per_minute'].mean(),
            'avg_overlay_density': niche_data['overlay_density'].mean(),
            'primary_text_zone': niche_data['primary_text_zone'].mode()[0],
            'intro_pct': niche_data['intro_percentage'].mean(),
            'content_pct': niche_data['content_percentage'].mean(),
            'cta_pct': niche_data['cta_percentage'].mean(),
            'avg_duration': niche_data['duration'].mean(),
            'speech_pace': niche_data['speech_pace_wpm'].mean(),
        }
        
        print(f"\n📋 RECOMMENDED TEMPLATE:")
        print("-" * 80)
        print(f"\n🎣 HOOK (First 5 seconds):")
        print(f"  • Type: {template['hook_type'].upper()}")
        print(f"  • Include visual text overlay: Yes")
        print(f"  • Speech pace: {template['speech_pace']:.0f} words per minute")
        
        print(f"\n⚡ PACING:")
        print(f"  • Target: {template['avg_cuts_per_minute']:.1f} cuts per minute")
        if template['avg_cuts_per_minute'] < 10:
            print(f"  → This is SLOW pacing (good for tutorials/demos)")
        elif template['avg_cuts_per_minute'] < 20:
            print(f"  → This is MODERATE pacing (good for product videos)")
        else:
            print(f"  → This is FAST pacing (good for social media ads)")
        
        print(f"\n📝 TEXT OVERLAYS:")
        print(f"  • Density: {template['avg_overlay_density']:.1f} overlays per second")
        print(f"  • Primary position: {template['primary_text_zone']}")
        
        print(f"\n🏗️  VIDEO STRUCTURE (for {template['avg_duration']:.0f}s video):")
        intro_sec = (template['intro_pct'] / 100) * template['avg_duration']
        content_sec = (template['content_pct'] / 100) * template['avg_duration']
        cta_sec = (template['cta_pct'] / 100) * template['avg_duration']
        
        print(f"  • Intro: {template['intro_pct']:.0f}% (~{intro_sec:.0f}s)")
        print(f"  • Main Content: {template['content_pct']:.0f}% (~{content_sec:.0f}s)")
        print(f"  • CTA: {template['cta_pct']:.0f}% (~{cta_sec:.0f}s)")
        
        print(f"\n💡 WHY THIS TEMPLATE?")
        print(f"  Based on analysis of {len(niche_data)} videos in {user_niche} niche.")
        print(f"  This represents the most common successful pattern.")
        
        return template
    
    def export_insights(self, output_file='template_insights.json'):
        """
        Export all insights to JSON for use in recommendation engine
        """
        insights = {}
        
        for niche in self.df['niche'].unique():
            niche_data = self.df[self.df['niche'] == niche]
            
            insights[niche] = {
                'sample_size': len(niche_data),
                'template_profile': {
                    'hook': {
                        'dominant_type': niche_data['hook_type'].mode()[0],
                        'type_distribution': dict(Counter(niche_data['hook_type'])),
                        'avg_speech_pace_wpm': float(niche_data['speech_pace_wpm'].mean()),
                    },
                    'pacing': {
                        'avg_cuts_per_minute': float(niche_data['cuts_per_minute'].mean()),
                        'avg_scene_duration': float(niche_data['avg_scene_duration'].mean()),
                        'category': self._categorize_pacing(niche_data['cuts_per_minute'].mean())
                    },
                    'text_overlays': {
                        'avg_density': float(niche_data['overlay_density'].mean()),
                        'primary_zone': niche_data['primary_text_zone'].mode()[0],
                    },
                    'structure': {
                        'intro_percentage': float(niche_data['intro_percentage'].mean()),
                        'content_percentage': float(niche_data['content_percentage'].mean()),
                        'cta_percentage': float(niche_data['cta_percentage'].mean()),
                    },
                    'duration': {
                        'avg': float(niche_data['duration'].mean()),
                        'median': float(niche_data['duration'].median()),
                        'range': [float(niche_data['duration'].min()), 
                                 float(niche_data['duration'].max())]
                    }
                }
            }
        
        with open(output_file, 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"\n✅ Insights exported to: {output_file}")
    
    def _categorize_pacing(self, cpm):
        """Helper to categorize cuts per minute"""
        if cpm < 10:
            return "slow"
        elif cpm < 20:
            return "moderate"
        elif cpm < 30:
            return "fast"
        else:
            return "very_fast"
