"""
Video Analysis Pipeline Package

A modular pipeline for analyzing video content, extracting template patterns,
and generating insights by niche.

Modules:
    - classification: Classify videos by niche
    - template_extraction: Extract video template profiles
    - template_analysis: Analyze patterns and generate insights
"""

__version__ = "1.0.0"

# Import main classes for easy access
try:
    from .classification import VideoClassifier
    from .template_extraction import TemplateExtractor
    from .template_analysis import TemplateAnalyser
    from .niche_taxonomy import NicheTaxonomy
except ImportError:
    # Fallback for direct execution
    pass

__all__ = [
    'VideoClassifier',
    'TemplateExtractor', 
    'TemplateAnalyser',
    'NicheTaxonomy',
]
