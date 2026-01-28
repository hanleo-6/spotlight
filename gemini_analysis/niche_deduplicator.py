"""
Niche Deduplication Module
Uses Gemini to identify and group similar niches into canonical categories
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

# Get API key from environment
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


def deduplicate_niches(niches: List[str], batch_size: int = 50, cache_file: str = None, force_refresh: bool = False) -> Dict[str, List[str]]:
    """
    Deduplicate niches using Gemini by batching similar niches together.
    
    Args:
        niches: List of niche strings to deduplicate
        batch_size: Number of niches to process per API call
        cache_file: Optional path to cache results
        force_refresh: If True, ignore cache and recompute
    
    Returns:
        Dictionary mapping canonical niches to lists of variant niches
        Example: {"Technology > AI Tools": ["Tech > AI tools", "Technology > AI/ML"]}
    """
    
    if not GEMINI_API_KEY:
        logger.error("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        logger.error("To get a free API key, visit: https://aistudio.google.com/app/apikey")
        return {niche: [niche] for niche in set(niches)}
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    unique_niches = list(set(niches))
    logger.info(f"Starting niche deduplication for {len(unique_niches)} unique niches...")
    
    # Check cache first (unless force_refresh)
    if not force_refresh and cache_file and Path(cache_file).exists():
        logger.info(f"Loading cached niche groups from {cache_file}")
        with open(cache_file, "r") as f:
            cached_groups = json.load(f)
        
        # Validate cache - if it has same number of niches as input, it's likely bad
        if len(cached_groups) < len(unique_niches) * 0.5:  # If heavily deduplicated, likely valid
            logger.info(f"✓ Using cache: {len(unique_niches)} niches reduced to {len(cached_groups)} canonical niches")
            return cached_groups
        else:
            logger.warning(f"⚠ Cache appears invalid ({len(cached_groups)} groups for {len(unique_niches)} niches)")
            logger.info("Proceeding with fresh deduplication...")
    
    niche_groups = {}
    processed = set()
    had_errors = False
    
    # Process in batches
    for i in range(0, len(unique_niches), batch_size):
        batch = unique_niches[i:i+batch_size]
        # Filter out already processed niches
        batch = [n for n in batch if n not in processed]
        
        if not batch:
            continue
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(unique_niches)//batch_size) + 1}...")
        
        prompt = f"""You are a niche categorization expert. Group similar niches together.

Input niches:
{json.dumps(batch, indent=2)}

Rules:
1. Group niches that are essentially the same or should be merged
2. Handle capitalization differences (e.g., "AI tools" = "AI Tools")
3. Handle abbreviations (e.g., "Tech" = "Technology")  
4. Keep the hierarchical format (e.g., "Technology > AI Tools")
5. Be conservative - only merge when confident

Return a JSON object ONLY. Example:
{{"Technology > AI Tools": ["Tech > AI tools", "Technology > AI Tools"], "Entertainment > Comedy": ["Entertainment > Comedy"]}}

Output JSON now:"""
        
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip() if response.text else ""
            
            if not response_text:
                logger.warning(f"  ⚠ Empty response from model")
                had_errors = True
                # Fallback: treat each niche as its own canonical group
                for niche in batch:
                    if niche not in niche_groups:
                        niche_groups[niche] = [niche]
                    processed.add(niche)
                continue
            
            # Try to parse JSON
            try:
                groups = json.loads(response_text)
                niche_groups.update(groups)
                
                # Mark all niches in this batch as processed
                for canonical, variants in groups.items():
                    for variant in variants:
                        processed.add(variant)
                
                logger.info(f"  ✓ Batch processed: created {len(groups)} canonical groups")
                
            except json.JSONDecodeError as e:
                logger.warning(f"  ⚠ Failed to parse JSON response: {e}")
                logger.debug(f"    Response text: {response_text[:200]}")
                had_errors = True
                # Fallback: treat each niche as its own canonical group
                for niche in batch:
                    if niche not in niche_groups:
                        niche_groups[niche] = [niche]
                    processed.add(niche)
        
        except Exception as e:
            logger.error(f"  ✗ Error processing batch: {e}")
            had_errors = True
            # Fallback: treat each niche as its own canonical group
            for niche in batch:
                if niche not in niche_groups:
                    niche_groups[niche] = [niche]
                processed.add(niche)
    
    # Ensure all niches are included in the output
    for niche in unique_niches:
        if niche not in processed:
            niche_groups[niche] = [niche]
    
    logger.info(f"✓ Deduplication complete: {len(unique_niches)} niches grouped into {len(niche_groups)} canonical niches")
    
    # Save to cache if specified and no errors occurred
    if cache_file and not had_errors:
        with open(cache_file, "w") as f:
            json.dump(niche_groups, f, indent=2)
        logger.info(f"✓ Cached niche groups to {cache_file}")
    elif cache_file and had_errors:
        logger.warning(f"⚠ Skipped caching due to API errors")
    
    return niche_groups


def apply_niche_mapping(df, niche_groups: Dict[str, List[str]], niche_column: str = "niche_full") -> object:
    """
    Apply niche deduplication mapping to a DataFrame.
    
    Args:
        df: Pandas DataFrame with niche column
        niche_groups: Dictionary from deduplicate_niches()
        niche_column: Column name containing niches
    
    Returns:
        DataFrame with new 'niche_canonical' column
    """
    
    # Create reverse mapping: variant -> canonical
    reverse_map = {}
    for canonical, variants in niche_groups.items():
        for variant in variants:
            reverse_map[variant] = canonical
    
    # Apply mapping
    df["niche_canonical"] = df[niche_column].map(reverse_map)
    
    # Fill any unmapped niches with original value
    df["niche_canonical"] = df["niche_canonical"].fillna(df[niche_column])
    
    canonical_count = df["niche_canonical"].nunique()
    logger.info(f"✓ Applied mapping: {df[niche_column].nunique()} niches mapped to {canonical_count} canonical niches")
    
    return df


def get_niche_stats(df, canonical_niche_column: str = "niche_canonical") -> Dict:
    """
    Generate statistics about niche distribution.
    
    Returns:
        Dictionary with statistics
    """
    niche_counts = df[canonical_niche_column].value_counts()
    
    stats = {
        "total_canonical_niches": len(niche_counts),
        "total_videos": len(df),
        "videos_per_niche": {
            "min": int(niche_counts.min()),
            "max": int(niche_counts.max()),
            "mean": float(niche_counts.mean()),
            "median": float(niche_counts.median())
        },
        "top_10_niches": niche_counts.head(10).to_dict()
    }
    
    return stats
