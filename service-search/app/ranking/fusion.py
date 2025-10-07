"""Result fusion algorithms for hybrid search."""

import math
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import structlog

logger = structlog.get_logger("search_fusion")


class RankFusionAlgorithm:
    """Base class for rank fusion algorithms."""
    
    def fuse_results(
        self,
        semantic_results: List[Tuple[str, str, float, Dict[str, Any]]],
        lexical_results: List[Tuple[str, str, float, Dict[str, Any]]],
        **kwargs
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Fuse semantic and lexical search results."""
        raise NotImplementedError


class ReciprocalRankFusion(RankFusionAlgorithm):
    """Reciprocal Rank Fusion (RRF) algorithm."""
    
    def __init__(self, k: float = 60.0):
        self.k = k  # RRF parameter
    
    def fuse_results(
        self,
        semantic_results: List[Tuple[str, str, float, Dict[str, Any]]],
        lexical_results: List[Tuple[str, str, float, Dict[str, Any]]],
        **kwargs
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Fuse results using RRF algorithm."""
        
        # Create result maps with ranks
        semantic_map = {}
        lexical_map = {}
        
        # Build semantic results map
        for rank, (entity_type, entity_id, score, metadata) in enumerate(semantic_results):
            key = (entity_type, entity_id)
            semantic_map[key] = {
                "rank": rank + 1,
                "score": score,
                "metadata": metadata
            }
        
        # Build lexical results map
        for rank, (entity_type, entity_id, score, metadata) in enumerate(lexical_results):
            key = (entity_type, entity_id)
            lexical_map[key] = {
                "rank": rank + 1,
                "score": score,
                "metadata": metadata
            }
        
        # Get all unique entities
        all_entities = set(semantic_map.keys()) | set(lexical_map.keys())
        
        # Calculate RRF scores
        fused_results = []
        for entity_key in all_entities:
            entity_type, entity_id = entity_key
            
            # Get ranks (use large rank if not present)
            semantic_rank = semantic_map.get(entity_key, {}).get("rank", len(semantic_results) + 100)
            lexical_rank = lexical_map.get(entity_key, {}).get("rank", len(lexical_results) + 100)
            
            # Calculate RRF score: 1/(k + rank)
            rrf_score = (1.0 / (self.k + semantic_rank)) + (1.0 / (self.k + lexical_rank))
            
            # Get metadata from the best available source
            metadata = (semantic_map.get(entity_key, {}).get("metadata") or 
                       lexical_map.get(entity_key, {}).get("metadata") or {})
            
            # Add fusion details to metadata
            metadata["fusion_details"] = {
                "semantic_rank": semantic_rank if entity_key in semantic_map else None,
                "lexical_rank": lexical_rank if entity_key in lexical_map else None,
                "semantic_score": semantic_map.get(entity_key, {}).get("score"),
                "lexical_score": lexical_map.get(entity_key, {}).get("score"),
                "rrf_score": rrf_score,
                "fusion_algorithm": "rrf",
                "k_parameter": self.k
            }
            
            fused_results.append((entity_type, entity_id, rrf_score, metadata))
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(
            "RRF fusion completed",
            semantic_count=len(semantic_results),
            lexical_count=len(lexical_results),
            fused_count=len(fused_results),
            k_parameter=self.k
        )
        
        return fused_results


class WeightedScoreFusion(RankFusionAlgorithm):
    """Weighted score fusion algorithm."""
    
    def __init__(self, semantic_weight: float = 0.7, lexical_weight: float = 0.3):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        
        # Normalize weights
        total_weight = semantic_weight + lexical_weight
        self.semantic_weight = semantic_weight / total_weight
        self.lexical_weight = lexical_weight / total_weight
    
    def fuse_results(
        self,
        semantic_results: List[Tuple[str, str, float, Dict[str, Any]]],
        lexical_results: List[Tuple[str, str, float, Dict[str, Any]]],
        **kwargs
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Fuse results using weighted scores."""
        
        # Normalize scores to [0, 1] range
        semantic_scores = [r[2] for r in semantic_results]
        lexical_scores = [r[2] for r in lexical_results]
        
        # Normalize semantic scores
        if semantic_scores:
            min_sem = min(semantic_scores)
            max_sem = max(semantic_scores)
            if max_sem > min_sem:
                semantic_norm = {
                    (r[0], r[1]): (r[2] - min_sem) / (max_sem - min_sem)
                    for r in semantic_results
                }
            else:
                semantic_norm = {(r[0], r[1]): 1.0 for r in semantic_results}
        else:
            semantic_norm = {}
        
        # Normalize lexical scores
        if lexical_scores:
            min_lex = min(lexical_scores)
            max_lex = max(lexical_scores)
            if max_lex > min_lex:
                lexical_norm = {
                    (r[0], r[1]): (r[2] - min_lex) / (max_lex - min_lex)
                    for r in lexical_results
                }
            else:
                lexical_norm = {(r[0], r[1]): 1.0 for r in lexical_results}
        else:
            lexical_norm = {}
        
        # Create result maps
        semantic_map = {(r[0], r[1]): (r[2], r[3]) for r in semantic_results}
        lexical_map = {(r[0], r[1]): (r[2], r[3]) for r in lexical_results}
        
        # Get all unique entities
        all_entities = set(semantic_map.keys()) | set(lexical_map.keys())
        
        # Calculate weighted scores
        fused_results = []
        for entity_key in all_entities:
            entity_type, entity_id = entity_key
            
            # Get normalized scores
            semantic_score = semantic_norm.get(entity_key, 0.0)
            lexical_score = lexical_norm.get(entity_key, 0.0)
            
            # Calculate weighted score
            weighted_score = (self.semantic_weight * semantic_score + 
                            self.lexical_weight * lexical_score)
            
            # Get metadata
            metadata = (semantic_map.get(entity_key, (0, {}))[1] or 
                       lexical_map.get(entity_key, (0, {}))[1] or {})
            
            # Add fusion details
            metadata["fusion_details"] = {
                "semantic_score": semantic_map.get(entity_key, (0, {}))[0],
                "lexical_score": lexical_map.get(entity_key, (0, {}))[0],
                "semantic_normalized": semantic_score,
                "lexical_normalized": lexical_score,
                "weighted_score": weighted_score,
                "semantic_weight": self.semantic_weight,
                "lexical_weight": self.lexical_weight,
                "fusion_algorithm": "weighted_score"
            }
            
            fused_results.append((entity_type, entity_id, weighted_score, metadata))
        
        # Sort by weighted score (descending)
        fused_results.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(
            "Weighted score fusion completed",
            semantic_count=len(semantic_results),
            lexical_count=len(lexical_results),
            fused_count=len(fused_results),
            semantic_weight=self.semantic_weight,
            lexical_weight=self.lexical_weight
        )
        
        return fused_results


class CombSUMFusion(RankFusionAlgorithm):
    """CombSUM fusion algorithm."""
    
    def fuse_results(
        self,
        semantic_results: List[Tuple[str, str, float, Dict[str, Any]]],
        lexical_results: List[Tuple[str, str, float, Dict[str, Any]]],
        **kwargs
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Fuse results using CombSUM algorithm."""
        
        # Create result maps
        semantic_map = {(r[0], r[1]): (r[2], r[3]) for r in semantic_results}
        lexical_map = {(r[0], r[1]): (r[2], r[3]) for r in lexical_results}
        
        # Get all unique entities
        all_entities = set(semantic_map.keys()) | set(lexical_map.keys())
        
        # Calculate combined scores
        fused_results = []
        for entity_key in all_entities:
            entity_type, entity_id = entity_key
            
            # Get scores (0 if not present)
            semantic_score = semantic_map.get(entity_key, (0.0, {}))[0]
            lexical_score = lexical_map.get(entity_key, (0.0, {}))[0]
            
            # CombSUM: simply add the scores
            combined_score = semantic_score + lexical_score
            
            # Get metadata
            metadata = (semantic_map.get(entity_key, (0, {}))[1] or 
                       lexical_map.get(entity_key, (0, {}))[1] or {})
            
            # Add fusion details
            metadata["fusion_details"] = {
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "combined_score": combined_score,
                "fusion_algorithm": "combsum"
            }
            
            fused_results.append((entity_type, entity_id, combined_score, metadata))
        
        # Sort by combined score (descending)
        fused_results.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(
            "CombSUM fusion completed",
            semantic_count=len(semantic_results),
            lexical_count=len(lexical_results),
            fused_count=len(fused_results)
        )
        
        return fused_results


class AdaptiveFusion(RankFusionAlgorithm):
    """Adaptive fusion that selects algorithm based on query characteristics."""
    
    def __init__(self):
        self.rrf = ReciprocalRankFusion(k=60.0)
        self.weighted = WeightedScoreFusion(semantic_weight=0.7, lexical_weight=0.3)
        self.combsum = CombSUMFusion()
    
    def fuse_results(
        self,
        semantic_results: List[Tuple[str, str, float, Dict[str, Any]]],
        lexical_results: List[Tuple[str, str, float, Dict[str, Any]]],
        query: str = "",
        **kwargs
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Adaptively select fusion algorithm based on query characteristics."""
        
        # Analyze query characteristics
        query_length = len(query.split()) if query else 0
        has_semantic_results = len(semantic_results) > 0
        has_lexical_results = len(lexical_results) > 0
        
        # Select fusion algorithm
        if not has_semantic_results and has_lexical_results:
            # Only lexical results - return as is
            return lexical_results
        elif has_semantic_results and not has_lexical_results:
            # Only semantic results - return as is
            return semantic_results
        elif query_length <= 2:
            # Short queries - prefer lexical matching
            return self.weighted.fuse_results(
                semantic_results, 
                lexical_results,
                semantic_weight=0.3,
                lexical_weight=0.7
            )
        elif query_length >= 5:
            # Long queries - prefer semantic matching
            return self.weighted.fuse_results(
                semantic_results,
                lexical_results, 
                semantic_weight=0.8,
                lexical_weight=0.2
            )
        else:
            # Medium queries - use RRF
            return self.rrf.fuse_results(semantic_results, lexical_results)


def create_fusion_algorithm(algorithm: str = "rrf", **params) -> RankFusionAlgorithm:
    """Create a fusion algorithm instance."""
    
    if algorithm == "rrf":
        k = params.get("k", 60.0)
        return ReciprocalRankFusion(k=k)
    
    elif algorithm == "weighted":
        semantic_weight = params.get("semantic_weight", 0.7)
        lexical_weight = params.get("lexical_weight", 0.3)
        return WeightedScoreFusion(semantic_weight, lexical_weight)
    
    elif algorithm == "combsum":
        return CombSUMFusion()
    
    elif algorithm == "adaptive":
        return AdaptiveFusion()
    
    else:
        raise ValueError(f"Unknown fusion algorithm: {algorithm}")


class QueryPreprocessor:
    """Preprocesses search queries for better matching."""
    
    def __init__(self):
        # Financial domain synonyms
        self.synonyms = {
            "gas": ["natural gas", "ng"],
            "oil": ["crude oil", "petroleum", "cl", "wti"],
            "fx": ["foreign exchange", "currency", "forex"],
            "rates": ["interest rates", "yields", "bonds"],
            "futures": ["forward", "derivative"],
            "equity": ["stock", "shares", "index"]
        }
        
        # Common abbreviations
        self.abbreviations = {
            "ng": "natural gas",
            "cl": "crude oil",
            "wti": "west texas intermediate",
            "hh": "henry hub",
            "usd": "us dollar",
            "eur": "euro",
            "gbp": "british pound",
            "jpy": "japanese yen"
        }
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """Preprocess search query."""
        original_query = query
        processed_query = query.lower().strip()
        
        # Expand abbreviations
        expanded_terms = []
        for term in processed_query.split():
            if term in self.abbreviations:
                expanded_terms.append(self.abbreviations[term])
            else:
                expanded_terms.append(term)
        
        expanded_query = " ".join(expanded_terms)
        
        # Generate synonym variations
        synonym_queries = [expanded_query]
        for term, synonyms in self.synonyms.items():
            if term in expanded_query:
                for synonym in synonyms:
                    variant = expanded_query.replace(term, synonym)
                    if variant not in synonym_queries:
                        synonym_queries.append(variant)
        
        # Extract potential entity types from query
        entity_type_hints = []
        if any(word in processed_query for word in ["curve", "yield", "rate"]):
            entity_type_hints.append("curve")
        if any(word in processed_query for word in ["futures", "contract", "commodity"]):
            entity_type_hints.append("instrument")
        if any(word in processed_query for word in ["scenario", "stress", "test"]):
            entity_type_hints.append("scenario")
        
        return {
            "original_query": original_query,
            "processed_query": processed_query,
            "expanded_query": expanded_query,
            "synonym_queries": synonym_queries,
            "entity_type_hints": entity_type_hints,
            "query_length": len(processed_query.split()),
            "has_numbers": any(char.isdigit() for char in processed_query),
            "has_abbreviations": any(term in self.abbreviations for term in processed_query.split())
        }


class SearchResultRanker:
    """Advanced ranking for search results."""
    
    def __init__(self):
        self.boost_factors = {
            "recency": 1.2,      # Boost recent items
            "popularity": 1.1,   # Boost popular items
            "exact_match": 1.5,  # Boost exact matches
            "entity_type": 1.0   # Base entity type boost
        }
    
    def rerank_results(
        self,
        results: List[Tuple[str, str, float, Dict[str, Any]]],
        query_info: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Rerank search results with additional signals."""
        
        reranked_results = []
        
        for entity_type, entity_id, score, metadata in results:
            boosted_score = score
            boost_details = {}
            
            # Recency boost
            if "updated_at" in metadata:
                # Boost more recent items
                # This is a simplified implementation
                recency_boost = self.boost_factors["recency"]
                boosted_score *= recency_boost
                boost_details["recency_boost"] = recency_boost
            
            # Exact match boost
            original_query = query_info.get("original_query", "").lower()
            entity_text = metadata.get("text", "").lower()
            if original_query in entity_text:
                exact_boost = self.boost_factors["exact_match"]
                boosted_score *= exact_boost
                boost_details["exact_match_boost"] = exact_boost
            
            # Entity type preference boost
            entity_type_hints = query_info.get("entity_type_hints", [])
            if entity_type in entity_type_hints:
                type_boost = 1.3
                boosted_score *= type_boost
                boost_details["entity_type_boost"] = type_boost
            
            # Add boost details to metadata
            if boost_details:
                metadata["ranking_details"] = {
                    "original_score": score,
                    "boosted_score": boosted_score,
                    "boosts_applied": boost_details
                }
            
            reranked_results.append((entity_type, entity_id, boosted_score, metadata))
        
        # Sort by boosted score
        reranked_results.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(
            "Results reranked",
            original_count=len(results),
            reranked_count=len(reranked_results)
        )
        
        return reranked_results


def create_query_preprocessor() -> QueryPreprocessor:
    """Create query preprocessor."""
    return QueryPreprocessor()


def create_result_ranker() -> SearchResultRanker:
    """Create search result ranker."""
    return SearchResultRanker()
