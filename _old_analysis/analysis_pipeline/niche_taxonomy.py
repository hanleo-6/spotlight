import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class NicheTaxonomy:
    """
    Hierarchical niche classification system for video templates
    Supports multi-level niche categorization for more targeted recommendations
    """
    
    def __init__(self):
        self.taxonomy = self._build_taxonomy()
        self.keyword_index = self._build_keyword_index()
    
    def _build_taxonomy(self) -> Dict:
        """
        Build hierarchical taxonomy: Category > Subcategory > Micro-niche
        """
        return {
            "tech": {
                "name": "Technology",
                "keywords": ["tech", "technology", "digital", "software"],
                "subcategories": {
                    "saas": {
                        "name": "SaaS Products",
                        "keywords": ["saas", "software as a service", "cloud software"],
                        "micro_niches": {
                            "productivity_saas": {
                                "name": "Productivity SaaS",
                                "keywords": ["notion", "monday", "asana", "clickup", "project management", "task management", "calendar app"],
                                "traits": {"typical_video_length": "30-60s", "pacing": "fast", "cta_strength": "high"}
                            },
                            "dev_tools": {
                                "name": "Developer Tools",
                                "keywords": ["api", "sdk", "devtool", "github", "deployment", "hosting", "vercel", "supabase"],
                                "traits": {"typical_video_length": "45-90s", "pacing": "moderate", "cta_strength": "medium"}
                            },
                            "design_tools": {
                                "name": "Design Tools",
                                "keywords": ["figma", "canva", "framer", "webflow", "design tool", "prototyping"],
                                "traits": {"typical_video_length": "30-45s", "pacing": "fast", "cta_strength": "high"}
                            },
                            "automation_tools": {
                                "name": "Automation & AI Tools",
                                "keywords": ["automation", "zapier", "make", "n8n", "ai tool", "chatgpt", "workflow automation"],
                                "traits": {"typical_video_length": "45-60s", "pacing": "moderate-fast", "cta_strength": "high"}
                            },
                            "crm_sales": {
                                "name": "CRM & Sales Tools",
                                "keywords": ["crm", "sales", "hubspot", "salesforce", "pipedrive", "lead generation"],
                                "traits": {"typical_video_length": "60-90s", "pacing": "moderate", "cta_strength": "high"}
                            }
                        }
                    },
                    "coding": {
                        "name": "Coding & Development",
                        "keywords": ["code", "coding", "programming", "developer", "dev"],
                        "micro_niches": {
                            "web_dev": {
                                "name": "Web Development",
                                "keywords": ["react", "nextjs", "vue", "frontend", "backend", "fullstack", "javascript", "typescript", "web dev"],
                                "traits": {"typical_video_length": "60-120s", "pacing": "slow", "cta_strength": "low"}
                            },
                            "mobile_dev": {
                                "name": "Mobile Development",
                                "keywords": ["ios", "android", "react native", "flutter", "swift", "kotlin", "mobile app"],
                                "traits": {"typical_video_length": "60-90s", "pacing": "moderate", "cta_strength": "medium"}
                            },
                            "ai_ml": {
                                "name": "AI & Machine Learning",
                                "keywords": ["ai", "ml", "machine learning", "neural network", "deep learning", "llm", "gpt"],
                                "traits": {"typical_video_length": "90-120s", "pacing": "slow", "cta_strength": "low"}
                            },
                            "gamedev": {
                                "name": "Game Development",
                                "keywords": ["game dev", "unity", "unreal", "godot", "game engine"],
                                "traits": {"typical_video_length": "60-120s", "pacing": "moderate", "cta_strength": "medium"}
                            }
                        }
                    },
                    "indie_hacking": {
                        "name": "Indie Hacking & Startups",
                        "keywords": ["indiehacker", "indiehack", "buildinpublic", "solopreneur", "bootstrapped"],
                        "micro_niches": {
                            "micro_saas": {
                                "name": "Micro SaaS",
                                "keywords": ["micro saas", "side project", "one person saas", "solo founder"],
                                "traits": {"typical_video_length": "30-60s", "pacing": "fast", "cta_strength": "high"}
                            },
                            "revenue_sharing": {
                                "name": "Revenue & Growth Stories",
                                "keywords": ["mrr", "revenue", "$10k", "monthly recurring", "growth story"],
                                "traits": {"typical_video_length": "30-45s", "pacing": "fast", "cta_strength": "medium"}
                            },
                            "maker_journey": {
                                "name": "Maker Journey",
                                "keywords": ["day in the life", "building", "shipping", "24 hour build", "coding challenge"],
                                "traits": {"typical_video_length": "60-90s", "pacing": "fast", "cta_strength": "low"}
                            }
                        }
                    },
                    "no_code": {
                        "name": "No-Code/Low-Code",
                        "keywords": ["nocode", "no code", "low code", "bubble", "airtable"],
                        "micro_niches": {
                            "no_code_apps": {
                                "name": "No-Code App Building",
                                "keywords": ["bubble", "adalo", "glide", "build app without code"],
                                "traits": {"typical_video_length": "45-90s", "pacing": "moderate", "cta_strength": "high"}
                            },
                            "automation_no_code": {
                                "name": "No-Code Automation",
                                "keywords": ["zapier", "make", "integromat", "automation without code"],
                                "traits": {"typical_video_length": "30-60s", "pacing": "fast", "cta_strength": "high"}
                            }
                        }
                    }
                }
            },
            
            "finance": {
                "name": "Finance & Investing",
                "keywords": ["finance", "money", "invest", "wealth"],
                "subcategories": {
                    "investing": {
                        "name": "Investing",
                        "keywords": ["invest", "investment", "portfolio", "stocks", "etf"],
                        "micro_niches": {
                            "stock_trading": {
                                "name": "Stock Trading",
                                "keywords": ["stocks", "day trading", "swing trading", "options", "stock market"],
                                "traits": {"typical_video_length": "30-60s", "pacing": "very-fast", "cta_strength": "high"}
                            },
                            "crypto": {
                                "name": "Cryptocurrency",
                                "keywords": ["crypto", "bitcoin", "ethereum", "blockchain", "defi", "nft"],
                                "traits": {"typical_video_length": "30-45s", "pacing": "very-fast", "cta_strength": "high"}
                            },
                            "passive_income": {
                                "name": "Passive Income",
                                "keywords": ["passive income", "dividends", "index funds", "fire", "financial freedom"],
                                "traits": {"typical_video_length": "45-60s", "pacing": "moderate", "cta_strength": "medium"}
                            }
                        }
                    },
                    "personal_finance": {
                        "name": "Personal Finance",
                        "keywords": ["budget", "saving", "debt", "personal finance", "money management"],
                        "micro_niches": {
                            "budgeting": {
                                "name": "Budgeting & Saving",
                                "keywords": ["budget", "saving money", "frugal", "spending less"],
                                "traits": {"typical_video_length": "45-60s", "pacing": "moderate", "cta_strength": "medium"}
                            },
                            "credit_cards": {
                                "name": "Credit Cards & Points",
                                "keywords": ["credit card", "rewards", "points", "cashback", "travel hacking"],
                                "traits": {"typical_video_length": "30-45s", "pacing": "fast", "cta_strength": "high"}
                            }
                        }
                    },
                    "fintech": {
                        "name": "Fintech Products",
                        "keywords": ["fintech", "financial technology", "banking app", "payment"],
                        "micro_niches": {
                            "banking_apps": {
                                "name": "Banking Apps",
                                "keywords": ["chime", "revolut", "n26", "wise", "digital bank"],
                                "traits": {"typical_video_length": "30-45s", "pacing": "fast", "cta_strength": "high"}
                            },
                            "payment_tools": {
                                "name": "Payment Tools",
                                "keywords": ["stripe", "paypal", "venmo", "payment processing"],
                                "traits": {"typical_video_length": "30-60s", "pacing": "moderate-fast", "cta_strength": "high"}
                            }
                        }
                    }
                }
            },
            
            "marketing": {
                "name": "Marketing & Growth",
                "keywords": ["marketing", "growth", "traffic", "leads"],
                "subcategories": {
                    "content_marketing": {
                        "name": "Content Marketing",
                        "keywords": ["content", "blog", "seo", "organic traffic"],
                        "micro_niches": {
                            "seo": {
                                "name": "SEO & Search Marketing",
                                "keywords": ["seo", "google ranking", "keywords", "backlinks", "search engine"],
                                "traits": {"typical_video_length": "45-90s", "pacing": "moderate", "cta_strength": "medium"}
                            },
                            "social_media": {
                                "name": "Social Media Marketing",
                                "keywords": ["instagram", "tiktok", "youtube", "twitter", "linkedin", "social media growth"],
                                "traits": {"typical_video_length": "30-45s", "pacing": "fast", "cta_strength": "high"}
                            },
                            "video_marketing": {
                                "name": "Video Marketing",
                                "keywords": ["video marketing", "youtube growth", "video seo", "thumbnail"],
                                "traits": {"typical_video_length": "45-60s", "pacing": "moderate", "cta_strength": "medium"}
                            }
                        }
                    },
                    "paid_ads": {
                        "name": "Paid Advertising",
                        "keywords": ["ads", "advertising", "ppc", "paid traffic"],
                        "micro_niches": {
                            "meta_ads": {
                                "name": "Meta/Facebook Ads",
                                "keywords": ["facebook ads", "instagram ads", "meta ads", "fb ads"],
                                "traits": {"typical_video_length": "60-90s", "pacing": "moderate", "cta_strength": "high"}
                            },
                            "google_ads": {
                                "name": "Google Ads",
                                "keywords": ["google ads", "ppc", "search ads", "display ads"],
                                "traits": {"typical_video_length": "60-90s", "pacing": "moderate", "cta_strength": "high"}
                            },
                            "tiktok_ads": {
                                "name": "TikTok Ads",
                                "keywords": ["tiktok ads", "spark ads", "ugc ads"],
                                "traits": {"typical_video_length": "30-45s", "pacing": "very-fast", "cta_strength": "high"}
                            }
                        }
                    },
                    "email_marketing": {
                        "name": "Email Marketing",
                        "keywords": ["email", "newsletter", "mailchimp", "convertkit"],
                        "micro_niches": {
                            "newsletters": {
                                "name": "Newsletter Growth",
                                "keywords": ["newsletter", "substack", "beehiiv", "email list"],
                                "traits": {"typical_video_length": "45-60s", "pacing": "moderate", "cta_strength": "medium"}
                            }
                        }
                    }
                }
            },
            
            "ecommerce": {
                "name": "E-commerce & Retail",
                "keywords": ["ecommerce", "online store", "shopify", "selling online"],
                "subcategories": {
                    "dropshipping": {
                        "name": "Dropshipping",
                        "keywords": ["dropshipping", "drop shipping", "aliexpress"],
                        "micro_niches": {
                            "shopify_dropship": {
                                "name": "Shopify Dropshipping",
                                "keywords": ["shopify", "shopify store", "print on demand"],
                                "traits": {"typical_video_length": "30-60s", "pacing": "fast", "cta_strength": "high"}
                            }
                        }
                    },
                    "amazon": {
                        "name": "Amazon Selling",
                        "keywords": ["amazon fba", "amazon seller", "fba"],
                        "micro_niches": {
                            "amazon_fba": {
                                "name": "Amazon FBA",
                                "keywords": ["amazon fba", "fulfillment by amazon", "private label"],
                                "traits": {"typical_video_length": "45-90s", "pacing": "moderate", "cta_strength": "medium"}
                            }
                        }
                    }
                }
            },
            
            "productivity": {
                "name": "Productivity & Lifestyle",
                "keywords": ["productivity", "efficiency", "life hack", "optimize"],
                "subcategories": {
                    "time_management": {
                        "name": "Time Management",
                        "keywords": ["time management", "schedule", "calendar", "routine"],
                        "micro_niches": {
                            "morning_routine": {
                                "name": "Morning Routines",
                                "keywords": ["morning routine", "5am club", "wake up early"],
                                "traits": {"typical_video_length": "45-60s", "pacing": "moderate", "cta_strength": "low"}
                            },
                            "deep_work": {
                                "name": "Deep Work & Focus",
                                "keywords": ["deep work", "focus", "pomodoro", "distraction free"],
                                "traits": {"typical_video_length": "60-90s", "pacing": "slow", "cta_strength": "low"}
                            }
                        }
                    },
                    "note_taking": {
                        "name": "Note-Taking & PKM",
                        "keywords": ["notes", "note taking", "pkm", "second brain", "zettelkasten"],
                        "micro_niches": {
                            "notion_setup": {
                                "name": "Notion Setups",
                                "keywords": ["notion", "notion setup", "notion template"],
                                "traits": {"typical_video_length": "60-120s", "pacing": "slow", "cta_strength": "medium"}
                            },
                            "obsidian_pkm": {
                                "name": "Obsidian & PKM",
                                "keywords": ["obsidian", "roam", "logseq", "pkm system"],
                                "traits": {"typical_video_length": "90-120s", "pacing": "slow", "cta_strength": "low"}
                            }
                        }
                    }
                }
            },
            
            "education": {
                "name": "Education & Learning",
                "keywords": ["learn", "tutorial", "how to", "guide", "teach"],
                "subcategories": {
                    "tech_tutorials": {
                        "name": "Tech Tutorials",
                        "keywords": ["coding tutorial", "programming tutorial", "tech tutorial"],
                        "micro_niches": {
                            "quick_tips": {
                                "name": "Quick Tech Tips",
                                "keywords": ["tip", "trick", "hack", "quick tutorial"],
                                "traits": {"typical_video_length": "15-30s", "pacing": "very-fast", "cta_strength": "low"}
                            },
                            "full_course": {
                                "name": "Full Course Promos",
                                "keywords": ["full course", "complete guide", "masterclass", "bootcamp"],
                                "traits": {"typical_video_length": "45-60s", "pacing": "moderate", "cta_strength": "high"}
                            }
                        }
                    }
                }
            }
        }
    
    def _build_keyword_index(self) -> Dict[str, List[str]]:
        """
        Create reverse index: keyword -> list of niche paths
        """
        index = defaultdict(list)
        
        def traverse(node, path):
            # Add category keywords
            if "keywords" in node:
                for keyword in node["keywords"]:
                    index[keyword.lower()].append(path)
            
            # Traverse subcategories
            if "subcategories" in node:
                for subcat_key, subcat_data in node["subcategories"].items():
                    traverse(subcat_data, path + [subcat_key])
            
            # Traverse micro-niches
            if "micro_niches" in node:
                for micro_key, micro_data in node["micro_niches"].items():
                    traverse(micro_data, path + [micro_key])
        
        # Build index for all categories
        for category_key, category_data in self.taxonomy.items():
            traverse(category_data, [category_key])
        
        return dict(index)
    
    def classify_content(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Classify content into the most specific niche possible
        
        Args:
            text: Combined text (description, title, transcription, etc.)
            metadata: Optional metadata like hashtags, creator info
            
        Returns:
            Dict with classification results at all levels
        """
        text_lower = text.lower()
        
        # Score each niche path
        scores = defaultdict(int)
        matched_keywords = defaultdict(list)
        
        for keyword, paths in self.keyword_index.items():
            if keyword in text_lower:
                for path in paths:
                    path_str = " > ".join(path)
                    # Weight longer paths (more specific) higher
                    weight = len(path)
                    scores[path_str] += weight
                    matched_keywords[path_str].append(keyword)
        
        if not scores:
            return {
                "category": "unknown",
                "subcategory": None,
                "micro_niche": None,
                "confidence": 0.0,
                "matched_keywords": []
            }
        
        # Get best match
        best_path = max(scores.items(), key=lambda x: x[1])
        path_parts = best_path[0].split(" > ")
        
        # Calculate confidence
        max_possible_score = len(matched_keywords[best_path[0]]) * len(path_parts)
        confidence = min(best_path[1] / max(max_possible_score, 1), 1.0)
        
        result = {
            "category": path_parts[0] if len(path_parts) > 0 else "unknown",
            "subcategory": path_parts[1] if len(path_parts) > 1 else None,
            "micro_niche": path_parts[2] if len(path_parts) > 2 else None,
            "confidence": round(confidence, 2),
            "matched_keywords": matched_keywords[best_path[0]],
            "full_path": " > ".join(path_parts)
        }
        
        # Get niche traits if available
        result["traits"] = self._get_niche_traits(path_parts)
        
        return result
    
    def _get_niche_traits(self, path_parts: List[str]) -> Optional[Dict]:
        """
        Get expected traits for a niche
        """
        try:
            node = self.taxonomy
            for part in path_parts:
                if part in node:
                    node = node[part]
                elif "subcategories" in node and part in node["subcategories"]:
                    node = node["subcategories"][part]
                elif "micro_niches" in node and part in node["micro_niches"]:
                    node = node["micro_niches"][part]
                else:
                    return None
            
            return node.get("traits")
        except:
            return None
    
    def get_niche_hierarchy(self, category: str, subcategory: str = None, 
                           micro_niche: str = None) -> Dict:
        """
        Get full details for a specific niche
        """
        if category not in self.taxonomy:
            return None
        
        result = {
            "category": self.taxonomy[category]["name"],
            "category_key": category
        }
        
        if subcategory and "subcategories" in self.taxonomy[category]:
            if subcategory in self.taxonomy[category]["subcategories"]:
                subcat_data = self.taxonomy[category]["subcategories"][subcategory]
                result["subcategory"] = subcat_data["name"]
                result["subcategory_key"] = subcategory
                
                if micro_niche and "micro_niches" in subcat_data:
                    if micro_niche in subcat_data["micro_niches"]:
                        micro_data = subcat_data["micro_niches"][micro_niche]
                        result["micro_niche"] = micro_data["name"]
                        result["micro_niche_key"] = micro_niche
                        result["traits"] = micro_data.get("traits")
        
        return result
    
    def get_similar_niches(self, niche_path: str, top_n: int = 5) -> List[str]:
        """
        Find similar niches based on shared parent categories
        """
        parts = niche_path.split(" > ")
        similar = []
        
        if len(parts) >= 2:
            category = parts[0]
            subcategory = parts[1]
            
            # Get all micro-niches in same subcategory
            if category in self.taxonomy:
                cat_data = self.taxonomy[category]
                if "subcategories" in cat_data and subcategory in cat_data["subcategories"]:
                    subcat_data = cat_data["subcategories"][subcategory]
                    if "micro_niches" in subcat_data:
                        for micro_key in subcat_data["micro_niches"].keys():
                            path = f"{category} > {subcategory} > {micro_key}"
                            if path != niche_path:
                                similar.append(path)
        
        return similar[:top_n]
    
    def export_taxonomy(self, output_file: str = "niche_taxonomy.json"):
        """
        Export full taxonomy to JSON
        """
        with open(output_file, 'w') as f:
            json.dump(self.taxonomy, f, indent=2)
        print(f"✅ Taxonomy exported to {output_file}")


# Example usage
if __name__ == "__main__":
    taxonomy = NicheTaxonomy()
    
    # Test classification
    print("="*80)
    print("NICHE CLASSIFICATION EXAMPLES")
    print("="*80)
    
    test_cases = [
        "Building a micro SaaS with Next.js - from $0 to $10k MRR in 6 months #buildinpublic",
        "How I use Notion to manage my entire life - complete setup tour",
        "Day trading stocks: My strategy for consistent profits using technical analysis",
        "Shopify dropshipping tutorial - find winning products with this method",
        "Facebook ads for beginners - complete guide to running profitable campaigns"
    ]
    
    for text in test_cases:
        print(f"\n📝 Text: {text[:80]}...")
        result = taxonomy.classify_content(text)
        print(f"   Category: {result['category']}")
        print(f"   Subcategory: {result['subcategory']}")
        print(f"   Micro-niche: {result['micro_niche']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Keywords matched: {result['matched_keywords']}")
        if result['traits']:
            print(f"   Expected traits: {result['traits']}")
    
    # Show similar niches
    print("\n" + "="*80)
    print("SIMILAR NICHES")
    print("="*80)
    sample_path = "tech > saas > productivity_saas"
    similar = taxonomy.get_similar_niches(sample_path)
    print(f"\nNiches similar to '{sample_path}':")
    for s in similar:
        print(f"  • {s}")
    
    # Export
    taxonomy.export_taxonomy()