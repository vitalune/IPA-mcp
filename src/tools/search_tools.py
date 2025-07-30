"""
Cross-Platform Search and Retrieval Tools

This module provides comprehensive search capabilities across multiple platforms:
- Universal search across Gmail, Drive, and social media
- Contextual search with relationship mapping
- Smart filtering and ranking
- Privacy-aware search with data minimization
- Real-time and cached search results
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import structlog

from ..models.data_models import (
    EmailMessage, EmailThread, SocialMediaPost, Project,
    TimeRange, DataFilter, QueryParams
)
from ..integrations.client_manager import get_client_manager
from ..utils.nlp_processor import get_nlp_processor, TextAnalysisResult
from ..utils.cache import get_cache_manager
from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class SearchScope(str, Enum):
    """Search scope options"""
    ALL = "all"
    EMAIL = "email"
    DRIVE = "drive"
    SOCIAL = "social"
    DOCUMENTS = "documents"
    CACHED = "cached"


class SortOrder(str, Enum):
    """Search result sorting options"""
    RELEVANCE = "relevance"
    DATE_DESC = "date_desc"
    DATE_ASC = "date_asc"
    IMPORTANCE = "importance"
    ENGAGEMENT = "engagement"


class SearchResultType(str, Enum):
    """Types of search results"""
    EMAIL = "email"
    EMAIL_THREAD = "email_thread"
    DOCUMENT = "document"
    SOCIAL_POST = "social_post"
    PROJECT = "project"
    CONTACT = "contact"


@dataclass
class SearchFilter:
    """Advanced search filter"""
    field: str
    operator: str  # eq, ne, contains, startswith, endswith, gt, lt, gte, lte, in
    value: Any
    weight: float = 1.0


@dataclass
class SearchResult:
    """Universal search result"""
    id: str
    type: SearchResultType
    title: str
    snippet: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    timestamp: datetime
    source: str
    url: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class SearchResults:
    """Collection of search results with metadata"""
    results: List[SearchResult]
    total_count: int
    search_time_ms: float
    query: str
    filters: List[SearchFilter]
    facets: Dict[str, Dict[str, int]]
    suggestions: List[str]
    page: int
    page_size: int


@dataclass
class ContextualSearchResult:
    """Search result with contextual relationships"""
    primary_result: SearchResult
    related_results: List[SearchResult]
    relationship_type: str
    relationship_strength: float
    context_summary: str


class UniversalSearchEngine:
    """Main search engine for cross-platform content"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client_manager = None
        self.nlp_processor = None
        self.cache_manager = None
        
        # Search configuration
        self._max_results_per_source = 100
        self._search_timeout_seconds = 30
        self._enable_content_caching = True
        self._privacy_mode = self.settings.privacy.anonymize_logs
        
        # TF-IDF vectorizer for content similarity
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        self._content_vectors = {}
        self._indexed_content = []
    
    async def initialize(self):
        """Initialize search engine dependencies"""
        self.client_manager = await get_client_manager()
        self.nlp_processor = await get_nlp_processor()
        self.cache_manager = get_cache_manager()
    
    async def universal_search(
        self,
        query: str,
        scope: SearchScope = SearchScope.ALL,
        filters: Optional[List[SearchFilter]] = None,
        sort_by: SortOrder = SortOrder.RELEVANCE,
        page: int = 1,
        page_size: int = 20,
        include_facets: bool = True
    ) -> SearchResults:
        """Perform universal search across all platforms"""
        if not self.client_manager:
            await self.initialize()
        
        start_time = datetime.now()
        filters = filters or []
        
        logger.info(f"Starting universal search", query=query, scope=scope.value)
        
        # Determine which sources to search
        sources_to_search = self._get_search_sources(scope)
        
        # Perform concurrent searches across sources
        search_tasks = []
        for source in sources_to_search:
            task = self._search_source(source, query, filters)
            search_tasks.append(task)
        
        # Collect results from all sources
        source_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine and process results
        all_results = []
        for i, result in enumerate(source_results):
            if isinstance(result, Exception):
                logger.warning(f"Search failed for source {sources_to_search[i]}: {result}")
                continue
            if isinstance(result, list):
                all_results.extend(result)
        
        # Apply additional filtering
        filtered_results = self._apply_filters(all_results, filters)
        
        # Calculate relevance scores
        scored_results = await self._calculate_relevance_scores(filtered_results, query)
        
        # Sort results
        sorted_results = self._sort_results(scored_results, sort_by)
        
        # Generate facets
        facets = {}
        if include_facets:
            facets = self._generate_facets(sorted_results)
        
        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = sorted_results[start_idx:end_idx]
        
        # Generate search suggestions
        suggestions = await self._generate_search_suggestions(query, sorted_results)
        
        # Calculate search time
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResults(
            results=paginated_results,
            total_count=len(sorted_results),
            search_time_ms=search_time,
            query=query,
            filters=filters,
            facets=facets,
            suggestions=suggestions,
            page=page,
            page_size=page_size
        )
    
    def _get_search_sources(self, scope: SearchScope) -> List[str]:
        """Determine which sources to search based on scope"""
        if scope == SearchScope.ALL:
            return ['email', 'drive', 'social', 'cached']
        elif scope == SearchScope.EMAIL:
            return ['email']
        elif scope == SearchScope.DRIVE:
            return ['drive']
        elif scope == SearchScope.SOCIAL:
            return ['social']
        elif scope == SearchScope.DOCUMENTS:
            return ['drive', 'cached']
        elif scope == SearchScope.CACHED:
            return ['cached']
        else:
            return ['cached']
    
    async def _search_source(
        self,
        source: str,
        query: str,
        filters: List[SearchFilter]
    ) -> List[SearchResult]:
        """Search a specific source"""
        try:
            if source == 'email':
                return await self._search_emails(query, filters)
            elif source == 'drive':
                return await self._search_drive(query, filters)
            elif source == 'social':
                return await self._search_social(query, filters)
            elif source == 'cached':
                return await self._search_cached(query, filters)
            else:
                logger.warning(f"Unknown search source: {source}")
                return []
        except Exception as e:
            logger.error(f"Error searching {source}: {e}")
            return []
    
    async def _search_emails(
        self,
        query: str,
        filters: List[SearchFilter]
    ) -> List[SearchResult]:
        """Search Gmail messages"""
        results = []
        
        try:
            gmail_client = await self.client_manager.get_client('gmail')
            if not gmail_client:
                logger.warning("Gmail client not available")
                return results
            
            # Build Gmail search query
            gmail_query = f"in:anywhere {query}"
            
            # Add filter-based query modifications
            for filter_obj in filters:
                if filter_obj.field == 'from' and filter_obj.operator == 'eq':
                    gmail_query += f" from:{filter_obj.value}"
                elif filter_obj.field == 'subject' and filter_obj.operator == 'contains':
                    gmail_query += f" subject:{filter_obj.value}"
                elif filter_obj.field == 'date' and filter_obj.operator == 'gte':
                    if isinstance(filter_obj.value, datetime):
                        date_str = filter_obj.value.strftime('%Y/%m/%d')
                        gmail_query += f" after:{date_str}"
            
            # Search Gmail
            messages = await gmail_client.search_messages(
                query=gmail_query,
                max_results=self._max_results_per_source
            )
            
            # Convert to SearchResult objects
            for message in messages:
                if isinstance(message, dict):
                    result = SearchResult(
                        id=message.get('id', ''),
                        type=SearchResultType.EMAIL,
                        title=message.get('subject', 'No Subject'),
                        snippet=message.get('snippet', ''),
                        content=message.get('body', ''),
                        metadata={
                            'from': message.get('from', ''),
                            'to': message.get('to', []),
                            'labels': message.get('labels', []),
                            'thread_id': message.get('thread_id', '')
                        },
                        relevance_score=0.0,  # Will be calculated later
                        timestamp=message.get('date', datetime.now()),
                        source='gmail',
                        url=f"https://mail.google.com/mail/u/0/#inbox/{message.get('id', '')}"
                    )
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Gmail search failed: {e}")
        
        return results
    
    async def _search_drive(
        self,
        query: str,
        filters: List[SearchFilter]
    ) -> List[SearchResult]:
        """Search Google Drive files"""
        results = []
        
        try:
            drive_client = await self.client_manager.get_client('drive')
            if not drive_client:
                logger.warning("Drive client not available")
                return results
            
            # Build Drive search query
            drive_query = f"fullText contains '{query}'"
            
            # Add filters
            for filter_obj in filters:
                if filter_obj.field == 'file_type' and filter_obj.operator == 'eq':
                    drive_query += f" and mimeType = '{filter_obj.value}'"
                elif filter_obj.field == 'modified' and filter_obj.operator == 'gte':
                    if isinstance(filter_obj.value, datetime):
                        date_str = filter_obj.value.isoformat()
                        drive_query += f" and modifiedTime >= '{date_str}'"
            
            # Search Drive
            files = await drive_client.search_files(
                query=drive_query,
                max_results=self._max_results_per_source
            )
            
            # Convert to SearchResult objects
            for file_info in files:
                if isinstance(file_info, dict):
                    result = SearchResult(
                        id=file_info.get('id', ''),
                        type=SearchResultType.DOCUMENT,
                        title=file_info.get('name', 'Untitled'),
                        snippet=file_info.get('description', '')[:200],
                        content='',  # Would need to fetch content separately
                        metadata={
                            'mime_type': file_info.get('mimeType', ''),
                            'size': file_info.get('size', 0),
                            'owners': file_info.get('owners', []),
                            'shared': file_info.get('shared', False)
                        },
                        relevance_score=0.0,
                        timestamp=file_info.get('modifiedTime', datetime.now()),
                        source='drive',
                        url=file_info.get('webViewLink', '')
                    )
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Drive search failed: {e}")
        
        return results
    
    async def _search_social(
        self,
        query: str,
        filters: List[SearchFilter]
    ) -> List[SearchResult]:
        """Search social media posts"""
        results = []
        
        try:
            # Search Twitter
            twitter_client = await self.client_manager.get_client('twitter')
            if twitter_client:
                tweets = await twitter_client.search_tweets(
                    query=query,
                    max_results=self._max_results_per_source // 2
                )
                
                for tweet in tweets:
                    if isinstance(tweet, dict):
                        result = SearchResult(
                            id=tweet.get('id', ''),
                            type=SearchResultType.SOCIAL_POST,
                            title=f"Tweet from {tweet.get('author', 'Unknown')}",
                            snippet=tweet.get('text', '')[:200],
                            content=tweet.get('text', ''),
                            metadata={
                                'platform': 'twitter',
                                'author': tweet.get('author', ''),
                                'likes': tweet.get('public_metrics', {}).get('like_count', 0),
                                'retweets': tweet.get('public_metrics', {}).get('retweet_count', 0)
                            },
                            relevance_score=0.0,
                            timestamp=tweet.get('created_at', datetime.now()),
                            source='twitter',
                            url=tweet.get('url', '')
                        )
                        results.append(result)
            
            # Search LinkedIn
            linkedin_client = await self.client_manager.get_client('linkedin')
            if linkedin_client:
                posts = await linkedin_client.search_posts(
                    query=query,
                    max_results=self._max_results_per_source // 2
                )
                
                for post in posts:
                    if isinstance(post, dict):
                        result = SearchResult(
                            id=post.get('id', ''),
                            type=SearchResultType.SOCIAL_POST,
                            title=f"LinkedIn post from {post.get('author', 'Unknown')}",
                            snippet=post.get('text', '')[:200],
                            content=post.get('text', ''),
                            metadata={
                                'platform': 'linkedin',
                                'author': post.get('author', ''),
                                'likes': post.get('likes', 0),
                                'comments': post.get('comments', 0)
                            },
                            relevance_score=0.0,
                            timestamp=post.get('created_at', datetime.now()),
                            source='linkedin',
                            url=post.get('url', '')
                        )
                        results.append(result)
        
        except Exception as e:
            logger.error(f"Social media search failed: {e}")
        
        return results
    
    async def _search_cached(
        self,
        query: str,
        filters: List[SearchFilter]
    ) -> List[SearchResult]:
        """Search cached content"""
        results = []
        
        try:
            # Search cached content
            cached_items = await self.cache_manager.search_cache(query)
            
            for item in cached_items:
                if isinstance(item, dict):
                    result = SearchResult(
                        id=item.get('id', ''),
                        type=SearchResultType.DOCUMENT,
                        title=item.get('title', 'Cached Item'),
                        snippet=item.get('snippet', ''),
                        content=item.get('content', ''),
                        metadata=item.get('metadata', {}),
                        relevance_score=0.0,
                        timestamp=item.get('timestamp', datetime.now()),
                        source='cache',
                        url=item.get('url', '')
                    )
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Cache search failed: {e}")
        
        return results
    
    def _apply_filters(
        self,
        results: List[SearchResult],
        filters: List[SearchFilter]
    ) -> List[SearchResult]:
        """Apply additional filters to search results"""
        filtered_results = results
        
        for filter_obj in filters:
            filtered_results = self._apply_single_filter(filtered_results, filter_obj)
        
        return filtered_results
    
    def _apply_single_filter(
        self,
        results: List[SearchResult],
        filter_obj: SearchFilter
    ) -> List[SearchResult]:
        """Apply a single filter to results"""
        filtered = []
        
        for result in results:
            value = self._get_filter_value(result, filter_obj.field)
            
            if self._filter_matches(value, filter_obj.operator, filter_obj.value):
                filtered.append(result)
        
        return filtered
    
    def _get_filter_value(self, result: SearchResult, field: str) -> Any:
        """Get the value for a filter field from a search result"""
        if field == 'type':
            return result.type.value
        elif field == 'source':
            return result.source
        elif field == 'timestamp':
            return result.timestamp
        elif field in result.metadata:
            return result.metadata[field]
        else:
            return None
    
    def _filter_matches(self, value: Any, operator: str, filter_value: Any) -> bool:
        """Check if a value matches a filter condition"""
        if value is None:
            return False
        
        try:
            if operator == 'eq':
                return value == filter_value
            elif operator == 'ne':
                return value != filter_value
            elif operator == 'contains':
                return str(filter_value).lower() in str(value).lower()
            elif operator == 'startswith':
                return str(value).lower().startswith(str(filter_value).lower())
            elif operator == 'endswith':
                return str(value).lower().endswith(str(filter_value).lower())
            elif operator == 'gt':
                return value > filter_value
            elif operator == 'lt':
                return value < filter_value
            elif operator == 'gte':
                return value >= filter_value
            elif operator == 'lte':
                return value <= filter_value
            elif operator == 'in':
                return value in filter_value
            else:
                return False
        except Exception:
            return False
    
    async def _calculate_relevance_scores(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """Calculate relevance scores for search results"""
        if not results:
            return results
        
        # Prepare documents for TF-IDF
        documents = []
        for result in results:
            # Combine title, snippet, and content for scoring
            doc_text = f"{result.title} {result.snippet} {result.content}"
            documents.append(doc_text)
        
        # Add query as first document for comparison
        documents.insert(0, query)
        
        try:
            # Calculate TF-IDF vectors
            tfidf_matrix = self._vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity between query and each document
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Update relevance scores
            for i, result in enumerate(results):
                base_score = similarities[i]
                
                # Apply boosting factors
                boost_factor = 1.0
                
                # Boost recent content
                days_old = (datetime.now() - result.timestamp).days
                if days_old < 7:
                    boost_factor *= 1.2
                elif days_old < 30:
                    boost_factor *= 1.1
                
                # Boost based on source authority
                if result.source == 'gmail':
                    boost_factor *= 1.1
                elif result.source == 'drive':
                    boost_factor *= 1.05
                
                # Boost based on engagement (for social posts)
                if result.type == SearchResultType.SOCIAL_POST:
                    likes = result.metadata.get('likes', 0)
                    shares = result.metadata.get('retweets', 0) + result.metadata.get('comments', 0)
                    if likes + shares > 10:
                        boost_factor *= 1.15
                
                result.relevance_score = base_score * boost_factor
        
        except Exception as e:
            logger.warning(f"Failed to calculate relevance scores: {e}")
            # Fallback: assign equal scores
            for result in results:
                result.relevance_score = 0.5
        
        return results
    
    def _sort_results(
        self,
        results: List[SearchResult],
        sort_by: SortOrder
    ) -> List[SearchResult]:
        """Sort search results"""
        if sort_by == SortOrder.RELEVANCE:
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == SortOrder.DATE_DESC:
            return sorted(results, key=lambda x: x.timestamp, reverse=True)
        elif sort_by == SortOrder.DATE_ASC:
            return sorted(results, key=lambda x: x.timestamp)
        elif sort_by == SortOrder.IMPORTANCE:
            # Custom importance scoring
            return sorted(results, key=self._calculate_importance_score, reverse=True)
        elif sort_by == SortOrder.ENGAGEMENT:
            return sorted(results, key=self._calculate_engagement_score, reverse=True)
        else:
            return results
    
    def _calculate_importance_score(self, result: SearchResult) -> float:
        """Calculate importance score for a result"""
        score = result.relevance_score
        
        # Boost emails marked as important
        if result.type == SearchResultType.EMAIL:
            if 'IMPORTANT' in result.metadata.get('labels', []):
                score *= 1.5
        
        # Boost starred content
        if 'starred' in result.metadata and result.metadata['starred']:
            score *= 1.3
        
        return score
    
    def _calculate_engagement_score(self, result: SearchResult) -> float:
        """Calculate engagement score for social media posts"""
        if result.type != SearchResultType.SOCIAL_POST:
            return result.relevance_score
        
        likes = result.metadata.get('likes', 0)
        shares = result.metadata.get('retweets', 0) + result.metadata.get('comments', 0)
        
        # Normalize engagement metrics
        engagement_score = (likes + shares * 2) / 100.0  # Arbitrary normalization
        
        return result.relevance_score + engagement_score
    
    def _generate_facets(self, results: List[SearchResult]) -> Dict[str, Dict[str, int]]:
        """Generate facets for search results"""
        facets = {
            'type': {},
            'source': {},
            'date_range': {}
        }
        
        for result in results:
            # Type facets
            type_val = result.type.value
            facets['type'][type_val] = facets['type'].get(type_val, 0) + 1
            
            # Source facets
            source_val = result.source
            facets['source'][source_val] = facets['source'].get(source_val, 0) + 1
            
            # Date range facets
            days_old = (datetime.now() - result.timestamp).days
            if days_old < 1:
                date_range = 'today'
            elif days_old < 7:
                date_range = 'this_week'
            elif days_old < 30:
                date_range = 'this_month'
            elif days_old < 365:
                date_range = 'this_year'
            else:
                date_range = 'older'
            
            facets['date_range'][date_range] = facets['date_range'].get(date_range, 0) + 1
        
        return facets
    
    async def _generate_search_suggestions(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[str]:
        """Generate search suggestions based on results"""
        suggestions = []
        
        try:
            # Extract common terms from high-scoring results
            top_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)[:10]
            
            # Analyze content for keywords
            content_texts = [f"{r.title} {r.snippet}" for r in top_results]
            if content_texts:
                keywords = await self.nlp_processor._extract_keywords(' '.join(content_texts), max_keywords=5)
                
                for keyword, score in keywords:
                    if keyword.lower() not in query.lower():
                        suggestions.append(f"{query} {keyword}")
            
            # Add common refinements
            if 'email' in [r.type.value for r in results]:
                suggestions.append(f"{query} from:important")
                suggestions.append(f"{query} has:attachment")
            
            if 'social_post' in [r.type.value for r in results]:
                suggestions.append(f"{query} engagement:high")
        
        except Exception as e:
            logger.warning(f"Failed to generate suggestions: {e}")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    async def contextual_search(
        self,
        query: str,
        context_items: List[str],
        relationship_types: Optional[List[str]] = None
    ) -> List[ContextualSearchResult]:
        """Perform contextual search with relationship mapping"""
        # First, perform regular search
        search_results = await self.universal_search(query)
        
        # Then find related content for each result
        contextual_results = []
        
        for result in search_results.results[:10]:  # Limit to top 10 for performance
            related_results = await self._find_related_content(result, context_items)
            
            if related_results:
                context_summary = await self._generate_context_summary(result, related_results)
                
                contextual_result = ContextualSearchResult(
                    primary_result=result,
                    related_results=related_results,
                    relationship_type="content_similarity",
                    relationship_strength=0.8,  # Would be calculated based on actual relationships
                    context_summary=context_summary
                )
                contextual_results.append(contextual_result)
        
        return contextual_results
    
    async def _find_related_content(
        self,
        primary_result: SearchResult,
        context_items: List[str]
    ) -> List[SearchResult]:
        """Find content related to a primary search result"""
        related_results = []
        
        # Extract key terms from primary result
        key_terms = await self.nlp_processor._extract_keywords(
            f"{primary_result.title} {primary_result.content}",
            max_keywords=5
        )
        
        # Search for content containing these key terms
        for term, score in key_terms:
            related_search = await self.universal_search(
                term,
                page_size=5
            )
            
            # Filter out the primary result itself
            for related_result in related_search.results:
                if related_result.id != primary_result.id:
                    related_results.append(related_result)
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_related = []
        for result in related_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_related.append(result)
                if len(unique_related) >= 5:
                    break
        
        return unique_related
    
    async def _generate_context_summary(
        self,
        primary_result: SearchResult,
        related_results: List[SearchResult]
    ) -> str:
        """Generate a summary of the contextual relationships"""
        if not related_results:
            return "No related content found."
        
        # Simple summary generation
        source_counts = {}
        for result in related_results:
            source_counts[result.source] = source_counts.get(result.source, 0) + 1
        
        summary_parts = []
        summary_parts.append(f"Found {len(related_results)} related items")
        
        for source, count in source_counts.items():
            summary_parts.append(f"{count} from {source}")
        
        return ", ".join(summary_parts)
    
    async def smart_filter_suggestions(
        self,
        query: str,
        current_results: List[SearchResult]
    ) -> List[SearchFilter]:
        """Generate smart filter suggestions based on current results"""
        suggestions = []
        
        # Analyze current results for filter opportunities
        sources = set(r.source for r in current_results)
        types = set(r.type.value for r in current_results)
        
        # Suggest source filters if multiple sources
        if len(sources) > 1:
            for source in sources:
                count = len([r for r in current_results if r.source == source])
                suggestions.append(SearchFilter(
                    field='source',
                    operator='eq',
                    value=source,
                    weight=count / len(current_results)
                ))
        
        # Suggest type filters if multiple types
        if len(types) > 1:
            for type_val in types:
                count = len([r for r in current_results if r.type.value == type_val])
                suggestions.append(SearchFilter(
                    field='type',
                    operator='eq',
                    value=type_val,
                    weight=count / len(current_results)
                ))
        
        # Suggest date filters
        now = datetime.now()
        recent_count = len([r for r in current_results if (now - r.timestamp).days < 7])
        if recent_count > 0:
            suggestions.append(SearchFilter(
                field='timestamp',
                operator='gte',
                value=now - timedelta(days=7),
                weight=recent_count / len(current_results)
            ))
        
        return sorted(suggestions, key=lambda x: x.weight, reverse=True)[:5]


# Global search engine instance
_search_engine: Optional[UniversalSearchEngine] = None


async def get_search_engine() -> UniversalSearchEngine:
    """Get the global search engine instance"""
    global _search_engine
    if _search_engine is None:
        _search_engine = UniversalSearchEngine()
        await _search_engine.initialize()
    return _search_engine


async def quick_search(
    query: str,
    scope: SearchScope = SearchScope.ALL,
    max_results: int = 10
) -> List[SearchResult]:
    """Quick search function for simple queries"""
    engine = await get_search_engine()
    results = await engine.universal_search(query, scope=scope, page_size=max_results)
    return results.results


async def search_with_context(
    query: str,
    context_items: List[str]
) -> List[ContextualSearchResult]:
    """Search with contextual relationships"""
    engine = await get_search_engine()
    return await engine.contextual_search(query, context_items)