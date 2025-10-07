"""Query understanding and intent classification for financial search."""

import asyncio
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import structlog

logger = structlog.get_logger("query_understanding")


class QueryIntent(Enum):
    """Query intent categories for financial search."""
    INSTRUMENT_LOOKUP = "instrument_lookup"
    CURVE_ANALYSIS = "curve_analysis"
    PRICE_INQUIRY = "price_inquiry"
    MARKET_DATA = "market_data"
    RISK_ANALYSIS = "risk_analysis"
    HISTORICAL_DATA = "historical_data"
    COMPARISON = "comparison"
    GENERAL_SEARCH = "general_search"


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    original_query: str
    processed_query: str
    intent: QueryIntent
    confidence: float
    entities: List[str]
    entity_types: List[str]
    time_references: List[str]
    numerical_values: List[float]
    keywords: List[str]
    suggestions: List[str]
    query_complexity: str  # simple, medium, complex


class FinancialQueryClassifier:
    """Classifies financial search queries by intent."""
    
    def __init__(self):
        """Initialize regex patterns for intents, entities, time and numbers."""
        # Intent classification patterns
        self.intent_patterns = {
            QueryIntent.INSTRUMENT_LOOKUP: [
                r'\b(futures?|contract|instrument|security)\b',
                r'\b(NG|CL|HO|RB|GC|SI|ZN|ZB|ZF)\b',  # Common symbols
                r'\b(front month|back month|calendar)\b'
            ],
            QueryIntent.CURVE_ANALYSIS: [
                r'\b(curve|yield|term structure)\b',
                r'\b(slope|steepening|flattening|inversion)\b',
                r'\b(tenor|maturity|duration)\b'
            ],
            QueryIntent.PRICE_INQUIRY: [
                r'\b(price|cost|value|worth)\b',
                r'\b(current|latest|real[- ]?time)\b',
                r'\b(\$|USD|EUR|GBP|JPY)\b'
            ],
            QueryIntent.MARKET_DATA: [
                r'\b(market|trading|volume|open interest)\b',
                r'\b(high|low|close|settlement)\b',
                r'\b(bid|ask|spread)\b'
            ],
            QueryIntent.RISK_ANALYSIS: [
                r'\b(risk|var|volatility|correlation)\b',
                r'\b(stress|scenario|monte carlo)\b',
                r'\b(hedge|exposure|sensitivity)\b'
            ],
            QueryIntent.HISTORICAL_DATA: [
                r'\b(historical|history|past|previous)\b',
                r'\b(trend|pattern|seasonal)\b',
                r'\b(since|from|until|between)\b'
            ],
            QueryIntent.COMPARISON: [
                r'\b(compare|comparison|versus|vs|against)\b',
                r'\b(difference|spread|relative)\b',
                r'\b(better|worse|higher|lower)\b'
            ]
        }
        
        # Financial entity patterns
        self.entity_patterns = {
            "commodities": [
                r'\b(natural gas|crude oil|heating oil|gasoline|gold|silver|copper)\b',
                r'\b(NG|CL|HO|RB|GC|SI|HG)\b'
            ],
            "currencies": [
                r'\b(EUR/USD|GBP/USD|USD/JPY|AUD/USD|USD/CHF)\b',
                r'\b(euro|dollar|pound|yen|franc)\b'
            ],
            "rates": [
                r'\b(treasury|government|corporate|municipal)\b',
                r'\b(libor|sofr|fed funds|discount rate)\b',
                r'\b(\d+[- ]?year|\d+Y|\d+M)\b'
            ],
            "exchanges": [
                r'\b(NYMEX|ICE|CME|CBOT|LME)\b'
            ]
        }
        
        # Time reference patterns
        self.time_patterns = [
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(this|last|next)\s+(week|month|quarter|year)\b',
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',  # Date patterns
            r'\b(Q[1-4]\s+\d{4})\b',  # Quarter patterns
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
        ]
        
        # Numerical patterns
        self.numerical_patterns = [
            r'\b(\d+\.?\d*)\s*(%|percent|bp|basis points?)\b',
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b',
            r'\b(\d+\.?\d*)\s*(million|billion|trillion)\b'
        ]
    
    def classify_query(self, query: str) -> QueryAnalysis:
        """Classify query intent and extract entities."""
        
        query_lower = query.lower()
        
        # Calculate intent scores
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                score += matches
            intent_scores[intent] = score
        
        # Determine primary intent
        if max(intent_scores.values()) == 0:
            primary_intent = QueryIntent.GENERAL_SEARCH
            confidence = 0.5
        else:
            primary_intent = max(intent_scores, key=intent_scores.get)
            total_score = sum(intent_scores.values())
            confidence = intent_scores[primary_intent] / total_score if total_score > 0 else 0.5
        
        # Extract entities
        entities = []
        entity_types = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    entities.extend(matches)
                    entity_types.extend([entity_type] * len(matches))
        
        # Extract time references
        time_references = []
        for pattern in self.time_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            time_references.extend(matches)
        
        # Extract numerical values
        numerical_values = []
        for pattern in self.numerical_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    numerical_values.append(float(match[0]))
                else:
                    try:
                        numerical_values.append(float(match))
                    except ValueError:
                        pass
        
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(query, primary_intent, entities)
        
        # Determine query complexity
        complexity = self._assess_query_complexity(query, entities, time_references, numerical_values)
        
        analysis = QueryAnalysis(
            original_query=query,
            processed_query=self._preprocess_query(query),
            intent=primary_intent,
            confidence=confidence,
            entities=entities,
            entity_types=list(set(entity_types)),
            time_references=time_references,
            numerical_values=numerical_values,
            keywords=keywords,
            suggestions=suggestions,
            query_complexity=complexity
        )
        
        logger.info("Query classified",
                   query=query[:50],
                   intent=primary_intent.value,
                   confidence=confidence,
                   entities=len(entities))
        
        return analysis
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching."""
        
        # Normalize whitespace
        processed = re.sub(r'\s+', ' ', query.strip())
        
        # Expand common abbreviations
        abbreviations = {
            r'\bng\b': 'natural gas',
            r'\bcl\b': 'crude oil',
            r'\bwti\b': 'west texas intermediate',
            r'\bhh\b': 'henry hub',
            r'\bfx\b': 'foreign exchange',
            r'\busd\b': 'us dollar',
            r'\beur\b': 'euro',
            r'\bgbp\b': 'british pound',
            r'\bjpy\b': 'japanese yen'
        }
        
        for abbrev, expansion in abbreviations.items():
            processed = re.sub(abbrev, expansion, processed, flags=re.IGNORECASE)
        
        return processed
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        
        # Remove stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'how'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _generate_suggestions(
        self,
        query: str,
        intent: QueryIntent,
        entities: List[str]
    ) -> List[str]:
        """Generate query suggestions based on intent and entities."""
        
        suggestions = []
        
        if intent == QueryIntent.INSTRUMENT_LOOKUP:
            if entities:
                suggestions.extend([
                    f"{entities[0]} futures contract",
                    f"{entities[0]} front month",
                    f"{entities[0]} calendar spread"
                ])
            else:
                suggestions.extend([
                    "natural gas futures",
                    "crude oil contract",
                    "treasury bond futures"
                ])
        
        elif intent == QueryIntent.CURVE_ANALYSIS:
            suggestions.extend([
                "yield curve slope",
                "term structure analysis",
                "curve steepening",
                "forward curve shape"
            ])
        
        elif intent == QueryIntent.PRICE_INQUIRY:
            if entities:
                suggestions.extend([
                    f"current {entities[0]} price",
                    f"{entities[0]} settlement price",
                    f"{entities[0]} bid ask spread"
                ])
        
        elif intent == QueryIntent.HISTORICAL_DATA:
            suggestions.extend([
                "historical volatility",
                "price history",
                "seasonal patterns",
                "long term trends"
            ])
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _assess_query_complexity(
        self,
        query: str,
        entities: List[str],
        time_refs: List[str],
        numerical_values: List[float]
    ) -> str:
        """Assess query complexity."""
        
        complexity_score = 0
        
        # Word count
        word_count = len(query.split())
        if word_count > 10:
            complexity_score += 2
        elif word_count > 5:
            complexity_score += 1
        
        # Entity count
        if len(entities) > 3:
            complexity_score += 2
        elif len(entities) > 1:
            complexity_score += 1
        
        # Time references
        if len(time_refs) > 1:
            complexity_score += 1
        
        # Numerical values
        if len(numerical_values) > 2:
            complexity_score += 1
        
        # Boolean operators
        if any(op in query.lower() for op in ['and', 'or', 'not', 'but']):
            complexity_score += 1
        
        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "simple"


class AutoCompleteEngine:
    """Auto-complete engine for financial search queries."""
    
    def __init__(self):
        """Prepare in-memory popularity, history and entity indexes."""
        self.query_history: List[str] = []
        self.popular_queries: Counter = Counter()
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Pre-populate with common financial terms
        self._initialize_financial_terms()
    
    def _initialize_financial_terms(self):
        """Initialize with common financial terms and entities."""
        
        # Common instruments
        instruments = [
            "natural gas futures", "crude oil futures", "heating oil futures",
            "gasoline futures", "gold futures", "silver futures",
            "treasury bonds", "corporate bonds", "municipal bonds",
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"
        ]
        
        # Common terms
        terms = [
            "yield curve", "forward curve", "volatility surface",
            "term structure", "basis spread", "calendar spread",
            "front month", "back month", "prompt month",
            "henry hub", "west texas intermediate", "brent crude",
            "settlement price", "closing price", "bid ask spread"
        ]
        
        all_terms = instruments + terms
        
        for term in all_terms:
            self.popular_queries[term] = 100  # Initial popularity
            
            # Index by words for prefix matching
            words = term.lower().split()
            for word in words:
                self.entity_index[word].add(term)
        
        logger.info("Financial terms initialized", terms=len(all_terms))
    
    def record_query(self, query: str, result_count: int = 0, clicked: bool = False):
        """Record a query for learning popular patterns."""
        
        self.query_history.append(query)
        
        # Update popularity based on engagement
        weight = 1
        if clicked:
            weight += 2
        if result_count > 0:
            weight += 1
        
        self.popular_queries[query.lower()] += weight
        
        # Index query terms
        words = query.lower().split()
        for word in words:
            if len(word) > 2:  # Skip very short words
                self.entity_index[word].add(query)
        
        # Limit history size
        if len(self.query_history) > 10000:
            self.query_history = self.query_history[-5000:]
        
        logger.debug("Query recorded", query=query[:50], weight=weight)
    
    def get_suggestions(
        self,
        partial_query: str,
        max_suggestions: int = 10,
        include_popular: bool = True,
        include_entity_based: bool = True
    ) -> List[Dict[str, Any]]:
        """Get auto-complete suggestions for partial query."""
        
        if len(partial_query) < 2:
            return []
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        # Prefix matching from popular queries
        if include_popular:
            for query, popularity in self.popular_queries.most_common(100):
                if query.startswith(partial_lower):
                    suggestions.append({
                        "text": query,
                        "type": "popular",
                        "score": popularity,
                        "source": "query_history"
                    })
        
        # Entity-based suggestions
        if include_entity_based:
            words = partial_lower.split()
            if words:
                last_word = words[-1]
                
                # Find entities containing the last word
                for indexed_word, entities in self.entity_index.items():
                    if indexed_word.startswith(last_word):
                        for entity in entities:
                            if entity not in [s["text"] for s in suggestions]:
                                suggestions.append({
                                    "text": entity,
                                    "type": "entity",
                                    "score": self.popular_queries.get(entity, 1),
                                    "source": "entity_index"
                                })
        
        # Fuzzy matching for typos
        fuzzy_suggestions = self._get_fuzzy_suggestions(partial_query)
        suggestions.extend(fuzzy_suggestions)
        
        # Sort by score and relevance
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion["text"] not in seen:
                seen.add(suggestion["text"])
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= max_suggestions:
                    break
        
        logger.debug("Suggestions generated", 
                    partial_query=partial_query,
                    suggestion_count=len(unique_suggestions))
        
        return unique_suggestions
    
    def _get_fuzzy_suggestions(self, partial_query: str) -> List[Dict[str, Any]]:
        """Get fuzzy matching suggestions for typos."""
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        # Simple edit distance matching
        for query, popularity in self.popular_queries.most_common(200):
            if self._edit_distance(partial_lower, query) <= 2 and len(query) > len(partial_lower):
                suggestions.append({
                    "text": query,
                    "type": "fuzzy",
                    "score": popularity * 0.8,  # Lower score for fuzzy matches
                    "source": "fuzzy_match"
                })
        
        return suggestions
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings."""
        
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
            distances = new_distances
        
        return distances[-1]


class QueryExpansionEngine:
    """Expands queries with synonyms and related terms."""
    
    def __init__(self):
        """Seed synonym and related-term maps for expansion."""
        # Financial domain synonyms
        self.synonyms = {
            "gas": ["natural gas", "ng"],
            "oil": ["crude oil", "petroleum", "cl", "wti"],
            "rates": ["interest rates", "yields", "bonds"],
            "fx": ["foreign exchange", "currency", "forex"],
            "futures": ["forward", "contract", "derivative"],
            "price": ["cost", "value", "rate", "quote"],
            "volatility": ["vol", "vega", "implied vol"],
            "spread": ["basis", "differential", "margin"]
        }
        
        # Related terms
        self.related_terms = {
            "natural gas": ["henry hub", "pipeline", "storage", "heating"],
            "crude oil": ["wti", "brent", "refining", "gasoline", "heating oil"],
            "treasury": ["government", "bonds", "notes", "bills", "yield"],
            "forex": ["currency", "exchange rate", "cross rate", "spot"],
            "volatility": ["vix", "implied", "realized", "historical"],
            "futures": ["expiry", "settlement", "margin", "open interest"]
        }
    
    def expand_query(
        self,
        query: str,
        max_expansions: int = 5,
        include_synonyms: bool = True,
        include_related: bool = True
    ) -> List[str]:
        """Expand query with synonyms and related terms."""
        
        expanded_queries = [query]  # Include original
        query_lower = query.lower()
        
        # Add synonym expansions
        if include_synonyms:
            for term, synonyms in self.synonyms.items():
                if term in query_lower:
                    for synonym in synonyms:
                        expanded_query = query_lower.replace(term, synonym)
                        if expanded_query != query_lower:
                            expanded_queries.append(expanded_query)
        
        # Add related term expansions
        if include_related:
            for term, related in self.related_terms.items():
                if term in query_lower:
                    for related_term in related[:2]:  # Limit to avoid explosion
                        expanded_query = f"{query} {related_term}"
                        expanded_queries.append(expanded_query)
        
        # Remove duplicates and limit
        unique_expansions = list(set(expanded_queries))
        
        logger.debug("Query expanded", 
                    original=query,
                    expansions=len(unique_expansions))
        
        return unique_expansions[:max_expansions]


class SpellChecker:
    """Spell checker for financial terms."""
    
    def __init__(self):
        """Load a lightweight financial dictionary and word frequencies."""
        # Common financial terms dictionary
        self.financial_dictionary = {
            "natural", "gas", "crude", "oil", "heating", "gasoline", "gold", "silver",
            "treasury", "government", "corporate", "municipal", "bond", "note", "bill",
            "futures", "forward", "option", "swap", "derivative", "contract",
            "yield", "rate", "curve", "term", "structure", "maturity", "duration",
            "volatility", "correlation", "spread", "basis", "margin", "premium",
            "settlement", "expiry", "delivery", "physical", "financial",
            "henry", "hub", "nymex", "ice", "cme", "cbot", "lme",
            "dollar", "euro", "pound", "yen", "franc", "currency", "exchange",
            "front", "back", "prompt", "deferred", "calendar", "butterfly"
        }
        
        self.word_frequencies = Counter(self.financial_dictionary)
    
    def correct_query(self, query: str) -> Tuple[str, List[str]]:
        """Correct spelling errors in query."""
        
        words = query.split()
        corrected_words = []
        corrections_made = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            
            if word_lower in self.financial_dictionary:
                corrected_words.append(word)
            else:
                # Find closest match
                correction = self._find_closest_word(word_lower)
                if correction and correction != word_lower:
                    corrected_words.append(correction)
                    corrections_made.append(f"{word} -> {correction}")
                else:
                    corrected_words.append(word)
        
        corrected_query = " ".join(corrected_words)
        
        if corrections_made:
            logger.info("Spelling corrections made", 
                       corrections=corrections_made)
        
        return corrected_query, corrections_made
    
    def _find_closest_word(self, word: str) -> Optional[str]:
        """Find closest word in financial dictionary."""
        
        if len(word) < 3:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for dict_word in self.financial_dictionary:
            if abs(len(word) - len(dict_word)) > 2:  # Skip very different lengths
                continue
            
            distance = self._edit_distance(word, dict_word)
            if distance < best_distance and distance <= 2:  # Max 2 edits
                best_distance = distance
                best_match = dict_word
        
        return best_match
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings."""
        
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
            distances = new_distances
        
        return distances[-1]


class QueryIntelligenceEngine:
    """Comprehensive query intelligence system."""
    
    def __init__(self):
        """Wire up classifier, autocomplete, expansion, and spell checker."""
        self.classifier = FinancialQueryClassifier()
        self.autocomplete = AutoCompleteEngine()
        self.query_expander = QueryExpansionEngine()
        self.spell_checker = SpellChecker()
        
        # Query analytics
        self.query_analytics = {
            "total_queries": 0,
            "intent_distribution": Counter(),
            "entity_popularity": Counter(),
            "average_query_length": 0,
            "complexity_distribution": Counter()
        }
    
    async def process_query(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        include_suggestions: bool = True,
        include_expansions: bool = True,
        include_spell_check: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive query processing."""
        
        start_time = time.perf_counter()
        
        # Spell check
        corrected_query = query
        spelling_corrections = []
        
        if include_spell_check:
            corrected_query, spelling_corrections = self.spell_checker.correct_query(query)
        
        # Classify query
        analysis = self.classifier.classify_query(corrected_query)
        
        # Generate suggestions
        suggestions = []
        if include_suggestions:
            suggestions = self.autocomplete.get_suggestions(corrected_query)
        
        # Generate query expansions
        expansions = []
        if include_expansions:
            expansions = self.query_expander.expand_query(corrected_query)
        
        # Update analytics
        self._update_analytics(analysis)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        result = {
            "original_query": query,
            "corrected_query": corrected_query,
            "spelling_corrections": spelling_corrections,
            "analysis": {
                "intent": analysis.intent.value,
                "confidence": analysis.confidence,
                "entities": analysis.entities,
                "entity_types": analysis.entity_types,
                "time_references": analysis.time_references,
                "numerical_values": analysis.numerical_values,
                "keywords": analysis.keywords,
                "complexity": analysis.query_complexity
            },
            "suggestions": suggestions,
            "expansions": expansions,
            "processing_time_ms": processing_time,
            "timestamp": time.time()
        }
        
        logger.info("Query processed",
                   original_query=query[:50],
                   intent=analysis.intent.value,
                   confidence=analysis.confidence,
                   processing_time_ms=processing_time)
        
        return result
    
    def _update_analytics(self, analysis: QueryAnalysis):
        """Update query analytics."""
        
        self.query_analytics["total_queries"] += 1
        self.query_analytics["intent_distribution"][analysis.intent.value] += 1
        self.query_analytics["complexity_distribution"][analysis.query_complexity] += 1
        
        for entity in analysis.entities:
            self.query_analytics["entity_popularity"][entity.lower()] += 1
        
        # Update average query length
        current_avg = self.query_analytics["average_query_length"]
        total_queries = self.query_analytics["total_queries"]
        new_length = len(analysis.original_query.split())
        
        self.query_analytics["average_query_length"] = (
            (current_avg * (total_queries - 1) + new_length) / total_queries
        )
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get query analytics summary."""
        
        return {
            "total_queries_processed": self.query_analytics["total_queries"],
            "average_query_length": self.query_analytics["average_query_length"],
            "intent_distribution": dict(self.query_analytics["intent_distribution"]),
            "complexity_distribution": dict(self.query_analytics["complexity_distribution"]),
            "top_entities": dict(self.query_analytics["entity_popularity"].most_common(10)),
            "top_queries": dict(self.autocomplete.popular_queries.most_common(10))
        }
    
    def optimize_search_parameters(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Optimize search parameters based on query analysis."""
        
        search_params = {
            "semantic": True,
            "limit": 10,
            "similarity_threshold": 0.0,
            "filters": {},
            "boost_factors": {}
        }
        
        # Adjust based on intent
        if analysis.intent == QueryIntent.INSTRUMENT_LOOKUP:
            search_params["filters"]["type"] = ["instrument"]
            search_params["boost_factors"]["exact_match"] = 1.5
        
        elif analysis.intent == QueryIntent.CURVE_ANALYSIS:
            search_params["filters"]["type"] = ["curve"]
            search_params["semantic"] = True  # Emphasize semantic search
        
        elif analysis.intent == QueryIntent.PRICE_INQUIRY:
            search_params["limit"] = 5  # Fewer results for price queries
            search_params["boost_factors"]["recency"] = 1.3
        
        elif analysis.intent == QueryIntent.HISTORICAL_DATA:
            search_params["limit"] = 20  # More results for historical queries
            search_params["boost_factors"]["temporal"] = 1.2
        
        # Adjust based on complexity
        if analysis.query_complexity == "complex":
            search_params["limit"] = 20
            search_params["similarity_threshold"] = 0.1  # Higher threshold
        elif analysis.query_complexity == "simple":
            search_params["limit"] = 5
            search_params["boost_factors"]["exact_match"] = 2.0
        
        # Add entity type filters if detected
        if analysis.entity_types:
            if "type" not in search_params["filters"]:
                search_params["filters"]["type"] = []
            search_params["filters"]["type"].extend(analysis.entity_types)
        
        logger.debug("Search parameters optimized",
                    intent=analysis.intent.value,
                    complexity=analysis.query_complexity,
                    filters=search_params["filters"])
        
        return search_params


def create_query_intelligence_engine() -> QueryIntelligenceEngine:
    """Create query intelligence engine."""
    return QueryIntelligenceEngine()


async def main():
    """Demo query intelligence functionality."""
    
    # Configure logging
    from libs.common.logging import configure_logging
    configure_logging("query_intelligence", "INFO", "json")
    
    # Create engine
    engine = create_query_intelligence_engine()
    
    # Test queries
    test_queries = [
        "natural gas futures",
        "crude oil price today",
        "yield curve slope analysis",
        "EUR/USD volatility",
        "treasury bond rates",
        "henry hub storage"
    ]
    
    for query in test_queries:
        result = await engine.process_query(query)
        
        print(f"\nQuery: {query}")
        print(f"Intent: {result['analysis']['intent']} (confidence: {result['analysis']['confidence']:.2f})")
        print(f"Entities: {result['analysis']['entities']}")
        print(f"Suggestions: {[s['text'] for s in result['suggestions'][:3]]}")
        
        # Record query for learning
        engine.autocomplete.record_query(query, result_count=5, clicked=True)
    
    # Show analytics
    analytics = engine.get_analytics_summary()
    print(f"\nAnalytics Summary:")
    print(f"Total queries: {analytics['total_queries_processed']}")
    print(f"Intent distribution: {analytics['intent_distribution']}")


if __name__ == "__main__":
    asyncio.run(main())
