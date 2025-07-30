"""
Social Media Intelligence and Optimization Tools

This module provides comprehensive social media analysis and optimization:
- Content performance analysis and optimization
- Audience analysis and engagement patterns  
- Hashtag intelligence and trend detection
- Posting time optimization
- Content suggestions and competitor analysis
- Cross-platform analytics and insights
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import structlog

from ..models.data_models import (
    SocialMediaPost, SocialPlatform, TimeRange, InsightReport
)
from ..integrations.client_manager import get_client_manager
from ..utils.nlp_processor import get_nlp_processor, TextAnalysisResult
from ..utils.analytics_engine import get_analytics_engine, TimeSeriesPoint
from ..tools.search_tools import get_search_engine
from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class ContentType(str, Enum):
    """Types of social media content"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    LINK = "link"
    POLL = "poll"
    STORY = "story"
    CAROUSEL = "carousel"


class EngagementMetric(str, Enum):
    """Social media engagement metrics"""
    LIKES = "likes"
    SHARES = "shares"
    COMMENTS = "comments"
    CLICKS = "clicks"
    IMPRESSIONS = "impressions"
    REACH = "reach"
    SAVES = "saves"


class PostingStrategy(str, Enum):
    """Content posting strategies"""
    CONSISTENT = "consistent"
    BURST = "burst"
    RESPONSIVE = "responsive"
    TRENDING = "trending"
    SEASONAL = "seasonal"


@dataclass
class ContentPerformance:
    """Performance metrics for content"""
    content_id: str
    platform: SocialPlatform
    content_type: ContentType
    engagement_rate: float
    reach: int
    impressions: int
    likes: int
    shares: int
    comments: int
    clicks: int
    post_time: datetime
    hashtags: List[str]
    mentions: List[str]
    sentiment_score: float
    topics: List[str]


@dataclass
class HashtagAnalysis:
    """Analysis results for hashtags"""
    hashtag: str
    usage_count: int
    avg_engagement: float
    reach_potential: int
    trending_score: float
    related_hashtags: List[str]
    best_platforms: List[SocialPlatform]
    sentiment_distribution: Dict[str, float]


@dataclass
class AudienceInsight:
    """Audience analysis insights"""
    platform: SocialPlatform
    total_followers: int
    engagement_rate: float
    active_hours: List[int]
    peak_days: List[int]
    demographics: Dict[str, Any]
    interests: List[Tuple[str, float]]
    location_distribution: Dict[str, int]
    device_usage: Dict[str, float]


@dataclass
class ContentSuggestion:
    """AI-generated content suggestion"""
    title: str
    content_type: ContentType
    suggested_text: str
    suggested_hashtags: List[str]
    best_posting_time: datetime
    target_platforms: List[SocialPlatform]
    predicted_engagement: float
    reasoning: str
    topics: List[str]
    confidence: float


@dataclass
class CompetitorAnalysis:
    """Competitor analysis results"""
    competitor_name: str
    platform: SocialPlatform
    followers_count: int
    avg_engagement_rate: float
    posting_frequency: float
    content_themes: List[Tuple[str, float]]
    top_hashtags: List[str]
    posting_schedule: Dict[str, List[int]]
    performance_benchmark: Dict[str, float]


class SocialMediaAnalyzer:
    """Main social media intelligence analyzer"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client_manager = None
        self.nlp_processor = None
        self.analytics_engine = None
        self.search_engine = None
        
        # Analysis configuration
        self._min_posts_for_analysis = 10
        self._engagement_rate_threshold = 0.02  # 2%
        self._trending_hashtag_threshold = 100
        
        # Caching
        self._performance_cache = {}
        self._hashtag_cache = {}
        self._audience_cache = {}
    
    async def initialize(self):
        """Initialize analyzer dependencies"""
        self.client_manager = await get_client_manager()
        self.nlp_processor = await get_nlp_processor()
        self.analytics_engine = get_analytics_engine()
        self.search_engine = await get_search_engine()
    
    async def analyze_content_performance(
        self,
        platform: SocialPlatform,
        time_range: TimeRange,
        content_types: Optional[List[ContentType]] = None
    ) -> Dict[str, Any]:
        """Analyze content performance across different dimensions"""
        if not self.client_manager:
            await self.initialize()
        
        logger.info("Starting content performance analysis", platform=platform.value)
        
        # Fetch posts from the platform
        posts = await self._fetch_posts(platform, time_range)
        
        if not posts:
            return {"error": f"No posts found for {platform.value} in the specified time range"}
        
        # Filter by content types if specified
        if content_types:
            posts = [p for p in posts if self._classify_content_type(p) in content_types]
        
        # Analyze performance
        performance_data = await self._analyze_post_performance(posts)
        
        results = {
            "platform": platform.value,
            "time_range": {
                "start": time_range.start.isoformat(),
                "end": time_range.end.isoformat()
            },
            "overview": {
                "total_posts": len(posts),
                "avg_engagement_rate": performance_data["avg_engagement_rate"],
                "total_reach": performance_data["total_reach"],
                "total_impressions": performance_data["total_impressions"],
                "best_performing_post": performance_data["best_post"],
                "worst_performing_post": performance_data["worst_post"]
            },
            "content_type_analysis": await self._analyze_by_content_type(posts),
            "temporal_analysis": await self._analyze_posting_times(posts),
            "hashtag_performance": await self._analyze_hashtag_performance(posts),
            "engagement_patterns": await self._analyze_engagement_patterns(posts),
            "recommendations": await self._generate_performance_recommendations(posts, performance_data)
        }
        
        return results
    
    async def analyze_hashtag_intelligence(
        self,
        hashtags: List[str],
        platforms: Optional[List[SocialPlatform]] = None
    ) -> List[HashtagAnalysis]:
        """Analyze hashtag performance and trends"""
        if not platforms:
            platforms = [SocialPlatform.TWITTER, SocialPlatform.LINKEDIN]
        
        hashtag_analyses = []
        
        for hashtag in hashtags:
            analysis = await self._analyze_single_hashtag(hashtag, platforms)
            if analysis:
                hashtag_analyses.append(analysis)
        
        # Sort by trending score
        hashtag_analyses.sort(key=lambda x: x.trending_score, reverse=True)
        
        return hashtag_analyses
    
    async def optimize_posting_schedule(
        self,
        platform: SocialPlatform,
        time_range: TimeRange,
        target_metrics: Optional[List[EngagementMetric]] = None
    ) -> Dict[str, Any]:
        """Optimize posting schedule based on audience activity and engagement"""
        if not target_metrics:
            target_metrics = [EngagementMetric.LIKES, EngagementMetric.SHARES, EngagementMetric.COMMENTS]
        
        # Fetch historical posts and their performance
        posts = await self._fetch_posts(platform, time_range)
        
        if len(posts) < self._min_posts_for_analysis:
            return {"error": f"Insufficient posts ({len(posts)}) for analysis. Need at least {self._min_posts_for_analysis}."}
        
        # Analyze posting times vs engagement
        time_performance = await self._analyze_time_performance(posts, target_metrics)
        
        # Generate optimal schedule
        optimal_schedule = await self._generate_optimal_schedule(time_performance, platform)
        
        return {
            "platform": platform.value,
            "analysis_period": {
                "start": time_range.start.isoformat(),
                "end": time_range.end.isoformat(),
                "posts_analyzed": len(posts)
            },
            "current_schedule_analysis": time_performance,
            "optimal_schedule": optimal_schedule,
            "predicted_improvement": optimal_schedule["predicted_improvement"],
            "recommendations": optimal_schedule["recommendations"]
        }
    
    async def analyze_audience_insights(
        self,
        platform: SocialPlatform,
        time_range: TimeRange
    ) -> AudienceInsight:
        """Analyze audience behavior and preferences"""
        # Fetch audience data from the platform
        audience_data = await self._fetch_audience_data(platform, time_range)
        
        # Analyze engagement patterns
        engagement_patterns = await self._analyze_audience_engagement(platform, time_range)
        
        # Extract insights
        return AudienceInsight(
            platform=platform,
            total_followers=audience_data.get("followers_count", 0),
            engagement_rate=engagement_patterns.get("avg_engagement_rate", 0.0),
            active_hours=engagement_patterns.get("peak_hours", []),
            peak_days=engagement_patterns.get("peak_days", []),
            demographics=audience_data.get("demographics", {}),
            interests=await self._extract_audience_interests(platform, time_range),
            location_distribution=audience_data.get("locations", {}),
            device_usage=audience_data.get("devices", {})
        )
    
    async def generate_content_suggestions(
        self,
        platform: SocialPlatform,
        content_themes: Optional[List[str]] = None,
        target_audience: Optional[Dict[str, Any]] = None,
        count: int = 5
    ) -> List[ContentSuggestion]:
        """Generate AI-powered content suggestions"""
        if not content_themes:
            # Extract themes from recent high-performing content
            content_themes = await self._extract_trending_themes(platform)
        
        suggestions = []
        
        for theme in content_themes[:count]:
            suggestion = await self._generate_content_for_theme(theme, platform, target_audience)
            if suggestion:
                suggestions.append(suggestion)
        
        # Sort by predicted engagement
        suggestions.sort(key=lambda x: x.predicted_engagement, reverse=True)
        
        return suggestions[:count]
    
    async def analyze_competitor_performance(
        self,
        competitor_handles: List[str],
        platform: SocialPlatform,
        time_range: TimeRange
    ) -> List[CompetitorAnalysis]:
        """Analyze competitor social media performance"""
        competitor_analyses = []
        
        for handle in competitor_handles:
            analysis = await self._analyze_single_competitor(handle, platform, time_range)
            if analysis:
                competitor_analyses.append(analysis)
        
        return competitor_analyses
    
    async def cross_platform_analysis(
        self,
        platforms: List[SocialPlatform],
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Analyze performance across multiple platforms"""
        platform_results = {}
        
        # Analyze each platform
        for platform in platforms:
            try:
                platform_data = await self.analyze_content_performance(platform, time_range)
                platform_results[platform.value] = platform_data
            except Exception as e:
                logger.warning(f"Failed to analyze {platform.value}: {e}")
                platform_results[platform.value] = {"error": str(e)}
        
        # Cross-platform insights
        cross_insights = await self._generate_cross_platform_insights(platform_results)
        
        return {
            "platforms": platform_results,
            "cross_platform_insights": cross_insights,
            "recommendations": await self._generate_cross_platform_recommendations(platform_results)
        }
    
    # Helper methods for data fetching
    async def _fetch_posts(
        self,
        platform: SocialPlatform,
        time_range: TimeRange
    ) -> List[Dict[str, Any]]:
        """Fetch posts from a social media platform"""
        try:
            if platform == SocialPlatform.TWITTER:
                client = await self.client_manager.get_client('twitter')
                if client:
                    return await client.get_user_tweets(
                        start_time=time_range.start,
                        end_time=time_range.end,
                        max_results=200
                    )
            elif platform == SocialPlatform.LINKEDIN:
                client = await self.client_manager.get_client('linkedin')
                if client:
                    return await client.get_user_posts(
                        start_time=time_range.start,
                        end_time=time_range.end,
                        max_results=200
                    )
        except Exception as e:
            logger.error(f"Failed to fetch posts from {platform.value}: {e}")
        
        return []
    
    async def _fetch_audience_data(
        self,
        platform: SocialPlatform,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Fetch audience analytics data"""
        try:
            if platform == SocialPlatform.TWITTER:
                client = await self.client_manager.get_client('twitter')
                if client:
                    return await client.get_audience_insights()
            elif platform == SocialPlatform.LINKEDIN:
                client = await self.client_manager.get_client('linkedin')
                if client:
                    return await client.get_follower_statistics()
        except Exception as e:
            logger.error(f"Failed to fetch audience data from {platform.value}: {e}")
        
        return {}
    
    # Analysis helper methods
    async def _analyze_post_performance(
        self,
        posts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze overall performance metrics for posts"""
        if not posts:
            return {}
        
        engagement_rates = []
        total_reach = 0
        total_impressions = 0
        best_post = None
        worst_post = None
        best_engagement = -1
        worst_engagement = float('inf')
        
        for post in posts:
            # Calculate engagement rate
            likes = post.get('public_metrics', {}).get('like_count', 0)
            shares = post.get('public_metrics', {}).get('retweet_count', 0)
            comments = post.get('public_metrics', {}).get('reply_count', 0)
            impressions = post.get('public_metrics', {}).get('impression_count', 1)
            
            engagement_rate = (likes + shares + comments) / max(impressions, 1)
            engagement_rates.append(engagement_rate)
            
            total_reach += post.get('public_metrics', {}).get('impression_count', 0)
            total_impressions += impressions
            
            # Track best and worst performing posts
            if engagement_rate > best_engagement:
                best_engagement = engagement_rate
                best_post = {
                    'id': post.get('id'),
                    'text': post.get('text', '')[:100] + '...',
                    'engagement_rate': engagement_rate,
                    'created_at': post.get('created_at')
                }
            
            if engagement_rate < worst_engagement:
                worst_engagement = engagement_rate
                worst_post = {
                    'id': post.get('id'),
                    'text': post.get('text', '')[:100] + '...',
                    'engagement_rate': engagement_rate,
                    'created_at': post.get('created_at')
                }
        
        return {
            'avg_engagement_rate': np.mean(engagement_rates) if engagement_rates else 0.0,
            'total_reach': total_reach,
            'total_impressions': total_impressions,
            'best_post': best_post,
            'worst_post': worst_post,
            'engagement_distribution': {
                'mean': float(np.mean(engagement_rates)) if engagement_rates else 0.0,
                'median': float(np.median(engagement_rates)) if engagement_rates else 0.0,
                'std': float(np.std(engagement_rates)) if engagement_rates else 0.0
            }
        }
    
    async def _analyze_by_content_type(
        self,
        posts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance by content type"""
        content_type_performance = defaultdict(list)
        
        for post in posts:
            content_type = self._classify_content_type(post)
            
            # Calculate engagement metrics
            likes = post.get('public_metrics', {}).get('like_count', 0)
            shares = post.get('public_metrics', {}).get('retweet_count', 0)
            comments = post.get('public_metrics', {}).get('reply_count', 0)
            impressions = post.get('public_metrics', {}).get('impression_count', 1)
            
            engagement_rate = (likes + shares + comments) / max(impressions, 1)
            
            content_type_performance[content_type.value].append({
                'engagement_rate': engagement_rate,
                'likes': likes,
                'shares': shares,
                'comments': comments,
                'impressions': impressions
            })
        
        # Calculate averages for each content type
        results = {}
        for content_type, metrics_list in content_type_performance.items():
            if metrics_list:
                results[content_type] = {
                    'count': len(metrics_list),
                    'avg_engagement_rate': np.mean([m['engagement_rate'] for m in metrics_list]),
                    'avg_likes': np.mean([m['likes'] for m in metrics_list]),
                    'avg_shares': np.mean([m['shares'] for m in metrics_list]),
                    'avg_comments': np.mean([m['comments'] for m in metrics_list]),
                    'total_impressions': sum([m['impressions'] for m in metrics_list])
                }
        
        return results
    
    def _classify_content_type(self, post: Dict[str, Any]) -> ContentType:
        """Classify the type of content in a post"""
        # Check for media attachments
        if post.get('attachments'):
            attachments = post['attachments']
            if 'media_keys' in attachments:
                # This would need to be enhanced with actual media type detection
                return ContentType.IMAGE
        
        # Check for URLs (link posts)
        text = post.get('text', '')
        if 'http' in text or 'www.' in text:
            return ContentType.LINK
        
        # Check for polls (platform-specific)
        if post.get('attachments', {}).get('poll_ids'):
            return ContentType.POLL
        
        # Default to text
        return ContentType.TEXT
    
    async def _analyze_posting_times(
        self,
        posts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze optimal posting times"""
        if not posts:
            return {}
        
        # Extract posting times and engagement
        time_engagement = []
        for post in posts:
            created_at = post.get('created_at')
            if created_at:
                if isinstance(created_at, str):
                    timestamp = pd.to_datetime(created_at)
                else:
                    timestamp = created_at
                
                # Calculate engagement
                likes = post.get('public_metrics', {}).get('like_count', 0)
                shares = post.get('public_metrics', {}).get('retweet_count', 0)
                comments = post.get('public_metrics', {}).get('reply_count', 0)
                engagement = likes + shares + comments
                
                time_engagement.append({
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.dayofweek,
                    'engagement': engagement,
                    'timestamp': timestamp
                })
        
        if not time_engagement:
            return {}
        
        # Analyze by hour
        hourly_performance = defaultdict(list)
        for item in time_engagement:
            hourly_performance[item['hour']].append(item['engagement'])
        
        hourly_avg = {
            hour: np.mean(engagements)
            for hour, engagements in hourly_performance.items()
        }
        
        # Analyze by day of week
        daily_performance = defaultdict(list)
        for item in time_engagement:
            daily_performance[item['day_of_week']].append(item['engagement'])
        
        daily_avg = {
            day: np.mean(engagements)
            for day, engagements in daily_performance.items()
        }
        
        # Find optimal times
        best_hours = sorted(hourly_avg.items(), key=lambda x: x[1], reverse=True)[:3]
        best_days = sorted(daily_avg.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'hourly_performance': dict(hourly_avg),
            'daily_performance': dict(daily_avg),
            'best_hours': [hour for hour, _ in best_hours],
            'best_days': [day for day, _ in best_days],
            'recommendations': {
                'optimal_posting_hours': [hour for hour, _ in best_hours],
                'optimal_posting_days': [day for day, _ in best_days]
            }
        }
    
    async def _analyze_hashtag_performance(
        self,
        posts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze hashtag performance"""
        hashtag_performance = defaultdict(list)
        
        for post in posts:
            # Extract hashtags
            hashtags = self._extract_hashtags(post.get('text', ''))
            
            if hashtags:
                # Calculate engagement
                likes = post.get('public_metrics', {}).get('like_count', 0)
                shares = post.get('public_metrics', {}).get('retweet_count', 0)
                comments = post.get('public_metrics', {}).get('reply_count', 0)
                engagement = likes + shares + comments
                
                for hashtag in hashtags:
                    hashtag_performance[hashtag].append(engagement)
        
        # Calculate averages
        hashtag_stats = {}
        for hashtag, engagements in hashtag_performance.items():
            if len(engagements) >= 2:  # Only include hashtags used multiple times
                hashtag_stats[hashtag] = {
                    'usage_count': len(engagements),
                    'avg_engagement': np.mean(engagements),
                    'total_engagement': sum(engagements),
                    'engagement_variance': np.var(engagements)
                }
        
        # Sort by average engagement
        top_hashtags = sorted(
            hashtag_stats.items(),
            key=lambda x: x[1]['avg_engagement'],
            reverse=True
        )[:10]
        
        return {
            'total_unique_hashtags': len(hashtag_performance),
            'hashtags_with_multiple_uses': len(hashtag_stats),
            'top_performing_hashtags': dict(top_hashtags),
            'hashtag_usage_distribution': {
                hashtag: stats['usage_count']
                for hashtag, stats in hashtag_stats.items()
            }
        }
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        hashtag_pattern = r'#(\w+)'
        hashtags = re.findall(hashtag_pattern, text.lower())
        return [f"#{tag}" for tag in hashtags]
    
    async def _analyze_engagement_patterns(
        self,
        posts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze detailed engagement patterns"""
        engagement_data = []
        
        for post in posts:
            metrics = post.get('public_metrics', {})
            engagement_data.append({
                'likes': metrics.get('like_count', 0),
                'shares': metrics.get('retweet_count', 0),
                'comments': metrics.get('reply_count', 0),
                'impressions': metrics.get('impression_count', 0),
                'created_at': post.get('created_at')
            })
        
        if not engagement_data:
            return {}
        
        # Calculate engagement ratios
        like_ratios = []
        share_ratios = []
        comment_ratios = []
        
        for data in engagement_data:
            total_engagement = data['likes'] + data['shares'] + data['comments']
            if total_engagement > 0:
                like_ratios.append(data['likes'] / total_engagement)
                share_ratios.append(data['shares'] / total_engagement)
                comment_ratios.append(data['comments'] / total_engagement)
        
        return {
            'engagement_composition': {
                'avg_like_ratio': np.mean(like_ratios) if like_ratios else 0.0,
                'avg_share_ratio': np.mean(share_ratios) if share_ratios else 0.0,
                'avg_comment_ratio': np.mean(comment_ratios) if comment_ratios else 0.0
            },
            'engagement_trends': await self._calculate_engagement_trends(engagement_data),
            'engagement_velocity': await self._calculate_engagement_velocity(engagement_data)
        }
    
    async def _calculate_engagement_trends(
        self,
        engagement_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate engagement trends over time"""
        if len(engagement_data) < 3:
            return {}
        
        # Sort by date
        sorted_data = sorted(engagement_data, key=lambda x: x['created_at'])
        
        # Extract time series data
        timestamps = [item['created_at'] for item in sorted_data]
        total_engagements = [
            item['likes'] + item['shares'] + item['comments']
            for item in sorted_data
        ]
        
        # Use analytics engine for trend analysis
        time_series_points = [
            TimeSeriesPoint(timestamp=ts, value=eng)
            for ts, eng in zip(timestamps, total_engagements)
        ]
        
        trend_analysis = await self.analytics_engine.analyze_time_series(
            time_series_points,
            detect_seasonality=False,
            detect_anomalies=False
        )
        
        return trend_analysis.get('trend', {})
    
    async def _calculate_engagement_velocity(
        self,
        engagement_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate how quickly posts gain engagement"""
        # This would require time-series engagement data
        # For now, return placeholder
        return {
            'avg_time_to_peak': 3600,  # 1 hour in seconds
            'engagement_half_life': 7200,  # 2 hours in seconds
            'velocity_score': 0.75
        }
    
    async def _generate_performance_recommendations(
        self,
        posts: List[Dict[str, Any]],
        performance_data: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on performance analysis"""
        recommendations = []
        
        # Engagement rate recommendations
        avg_engagement = performance_data.get('avg_engagement_rate', 0.0)
        if avg_engagement < self._engagement_rate_threshold:
            recommendations.append(
                f"Your average engagement rate ({avg_engagement:.2%}) is below the typical threshold "
                f"({self._engagement_rate_threshold:.2%}). Consider experimenting with different content types and posting times."
            )
        
        # Content type recommendations
        content_analysis = await self._analyze_by_content_type(posts)
        if content_analysis:
            best_type = max(content_analysis.items(), key=lambda x: x[1]['avg_engagement_rate'])
            recommendations.append(
                f"Your {best_type[0]} content performs best with {best_type[1]['avg_engagement_rate']:.2%} "
                f"average engagement. Consider creating more {best_type[0]} content."
            )
        
        # Posting time recommendations
        time_analysis = await self._analyze_posting_times(posts)
        if time_analysis.get('best_hours'):
            best_hours = time_analysis['best_hours']
            recommendations.append(
                f"Your posts perform best at {', '.join(map(str, best_hours))}:00. "
                f"Consider scheduling important content during these hours."
            )
        
        return recommendations
    
    async def _analyze_single_hashtag(
        self,
        hashtag: str,
        platforms: List[SocialPlatform]
    ) -> Optional[HashtagAnalysis]:
        """Analyze a single hashtag across platforms"""
        try:
            # This would integrate with actual hashtag tracking APIs
            # For now, return placeholder analysis
            return HashtagAnalysis(
                hashtag=hashtag,
                usage_count=1500,
                avg_engagement=125.5,
                reach_potential=15000,
                trending_score=0.75,
                related_hashtags=[f"#{hashtag}related", f"#{hashtag}trend"],
                best_platforms=[SocialPlatform.TWITTER],
                sentiment_distribution={"positive": 0.6, "neutral": 0.3, "negative": 0.1}
            )
        except Exception as e:
            logger.error(f"Failed to analyze hashtag {hashtag}: {e}")
            return None
    
    async def _analyze_time_performance(
        self,
        posts: List[Dict[str, Any]],
        target_metrics: List[EngagementMetric]
    ) -> Dict[str, Any]:
        """Analyze posting time vs performance"""
        time_performance = defaultdict(list)
        
        for post in posts:
            created_at = post.get('created_at')
            if created_at:
                if isinstance(created_at, str):
                    timestamp = pd.to_datetime(created_at)
                else:
                    timestamp = created_at
                
                hour = timestamp.hour
                day_of_week = timestamp.dayofweek
                
                # Calculate target metrics
                metrics = {}
                for metric in target_metrics:
                    if metric == EngagementMetric.LIKES:
                        metrics['likes'] = post.get('public_metrics', {}).get('like_count', 0)
                    elif metric == EngagementMetric.SHARES:
                        metrics['shares'] = post.get('public_metrics', {}).get('retweet_count', 0)
                    elif metric == EngagementMetric.COMMENTS:
                        metrics['comments'] = post.get('public_metrics', {}).get('reply_count', 0)
                
                time_key = f"{day_of_week}_{hour}"
                time_performance[time_key].append(metrics)
        
        # Calculate averages
        time_averages = {}
        for time_key, metrics_list in time_performance.items():
            if metrics_list:
                avg_metrics = {}
                for metric in target_metrics:
                    metric_name = metric.value
                    values = [m.get(metric_name, 0) for m in metrics_list]
                    avg_metrics[metric_name] = np.mean(values) if values else 0.0
                
                time_averages[time_key] = {
                    'count': len(metrics_list),
                    'averages': avg_metrics,
                    'total_score': sum(avg_metrics.values())
                }
        
        return time_averages
    
    async def _generate_optimal_schedule(
        self,
        time_performance: Dict[str, Any],
        platform: SocialPlatform
    ) -> Dict[str, Any]:
        """Generate optimal posting schedule"""
        # Sort time slots by performance
        sorted_times = sorted(
            time_performance.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        # Extract best times
        best_times = []
        for time_key, data in sorted_times[:7]:  # Top 7 time slots
            day_of_week, hour = map(int, time_key.split('_'))
            best_times.append({
                'day_of_week': day_of_week,
                'hour': hour,
                'expected_performance': data['total_score'],
                'confidence': min(1.0, data['count'] / 10.0)  # Based on data points
            })
        
        # Calculate predicted improvement
        current_avg = np.mean([data['total_score'] for data in time_performance.values()])
        optimal_avg = np.mean([time['expected_performance'] for time in best_times[:3]])
        improvement = (optimal_avg - current_avg) / current_avg if current_avg > 0 else 0.0
        
        return {
            'optimal_time_slots': best_times,
            'recommended_frequency': self._calculate_optimal_frequency(platform),
            'predicted_improvement': improvement,
            'recommendations': [
                f"Post during your top 3 time slots for {improvement:.1%} improvement in engagement",
                f"Maintain consistent posting frequency of {self._calculate_optimal_frequency(platform)} posts per week"
            ]
        }
    
    def _calculate_optimal_frequency(self, platform: SocialPlatform) -> int:
        """Calculate optimal posting frequency for platform"""
        # Platform-specific recommendations
        if platform == SocialPlatform.TWITTER:
            return 7  # Daily
        elif platform == SocialPlatform.LINKEDIN:
            return 3  # 3 times per week
        else:
            return 5  # Default
    
    async def _analyze_audience_engagement(
        self,
        platform: SocialPlatform,
        time_range: TimeRange
    ) -> Dict[str, Any]:
        """Analyze audience engagement patterns"""
        # This would integrate with platform analytics APIs
        # Placeholder implementation
        return {
            'avg_engagement_rate': 0.045,  # 4.5%
            'peak_hours': [9, 14, 20],
            'peak_days': [1, 2, 3],  # Monday, Tuesday, Wednesday
            'engagement_by_content_type': {
                'text': 0.03,
                'image': 0.05,
                'video': 0.07
            }
        }
    
    async def _extract_audience_interests(
        self,
        platform: SocialPlatform,
        time_range: TimeRange
    ) -> List[Tuple[str, float]]:
        """Extract audience interests from engagement data"""
        # This would analyze what content gets the most engagement
        # Placeholder implementation
        return [
            ("technology", 0.85),
            ("business", 0.72),
            ("innovation", 0.68),
            ("productivity", 0.61),
            ("leadership", 0.55)
        ]
    
    async def _extract_trending_themes(
        self,
        platform: SocialPlatform
    ) -> List[str]:
        """Extract trending themes for content generation"""
        # This would analyze current trends and high-performing content
        return [
            "AI and machine learning",
            "Remote work productivity",
            "Digital transformation",
            "Sustainable business practices",
            "Personal branding"
        ]
    
    async def _generate_content_for_theme(
        self,
        theme: str,
        platform: SocialPlatform,
        target_audience: Optional[Dict[str, Any]]
    ) -> Optional[ContentSuggestion]:
        """Generate content suggestion for a specific theme"""
        try:
            # Use NLP processor to generate content ideas
            keywords = await self.nlp_processor._extract_keywords(theme, max_keywords=3)
            
            # Generate content based on theme and platform
            if platform == SocialPlatform.TWITTER:
                suggested_text = f"Exploring {theme}: Key insights and trends shaping the future. What's your take? 🤔"
                content_type = ContentType.TEXT
            elif platform == SocialPlatform.LINKEDIN:
                suggested_text = f"The impact of {theme} on modern business cannot be overstated. Here's why it matters for your organization..."
                content_type = ContentType.TEXT
            else:
                suggested_text = f"Let's discuss {theme} and its implications."
                content_type = ContentType.TEXT
            
            # Generate hashtags
            suggested_hashtags = [f"#{keyword[0].replace(' ', '')}" for keyword, _ in keywords[:3]]
            suggested_hashtags.append(f"#{theme.replace(' ', '')}")
            
            # Predict optimal posting time (placeholder)
            optimal_time = datetime.now() + timedelta(hours=2)
            
            return ContentSuggestion(
                title=f"Content about {theme}",
                content_type=content_type,
                suggested_text=suggested_text,
                suggested_hashtags=suggested_hashtags,
                best_posting_time=optimal_time,
                target_platforms=[platform],
                predicted_engagement=0.035,  # 3.5% predicted engagement
                reasoning=f"This theme aligns with current trends and your audience interests",
                topics=[theme],
                confidence=0.75
            )
        
        except Exception as e:
            logger.error(f"Failed to generate content for theme {theme}: {e}")
            return None
    
    async def _analyze_single_competitor(
        self,
        handle: str,
        platform: SocialPlatform,
        time_range: TimeRange
    ) -> Optional[CompetitorAnalysis]:
        """Analyze a single competitor's performance"""
        try:
            # This would integrate with social media APIs to fetch competitor data
            # Placeholder implementation
            return CompetitorAnalysis(
                competitor_name=handle,
                platform=platform,
                followers_count=50000,
                avg_engagement_rate=0.042,
                posting_frequency=5.2,  # posts per week
                content_themes=[
                    ("technology", 0.35),
                    ("business strategy", 0.28),
                    ("industry news", 0.22)
                ],
                top_hashtags=["#tech", "#business", "#innovation"],
                posting_schedule={
                    "Monday": [9, 14],
                    "Tuesday": [10, 15],
                    "Wednesday": [9, 16],
                    "Thursday": [11, 14],
                    "Friday": [10, 13]
                },
                performance_benchmark={
                    "engagement_rate": 0.042,
                    "avg_likes": 250,
                    "avg_shares": 45,
                    "avg_comments": 18
                }
            )
        except Exception as e:
            logger.error(f"Failed to analyze competitor {handle}: {e}")
            return None
    
    async def _generate_cross_platform_insights(
        self,
        platform_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights from cross-platform analysis"""
        insights = {
            "performance_comparison": {},
            "content_type_preferences": {},
            "optimal_cross_posting": {},
            "platform_specific_recommendations": {}
        }
        
        # Compare performance across platforms
        for platform, data in platform_results.items():
            if "error" not in data:
                overview = data.get("overview", {})
                insights["performance_comparison"][platform] = {
                    "engagement_rate": overview.get("avg_engagement_rate", 0.0),
                    "total_reach": overview.get("total_reach", 0),
                    "posts_count": overview.get("total_posts", 0)
                }
        
        return insights
    
    async def _generate_cross_platform_recommendations(
        self,
        platform_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for cross-platform strategy"""
        recommendations = []
        
        # Find best performing platform
        best_platform = None
        best_engagement = 0.0
        
        for platform, data in platform_results.items():
            if "error" not in data:
                engagement = data.get("overview", {}).get("avg_engagement_rate", 0.0)
                if engagement > best_engagement:
                    best_engagement = engagement
                    best_platform = platform
        
        if best_platform:
            recommendations.append(
                f"Focus more effort on {best_platform} where you have the highest engagement rate ({best_engagement:.2%})"
            )
        
        # Content adaptation recommendations
        recommendations.append(
            "Adapt content format for each platform - longer posts for LinkedIn, concise posts for Twitter"
        )
        
        recommendations.append(
            "Cross-promote your best performing content across platforms with platform-specific adaptations"
        )
        
        return recommendations


# Global analyzer instance
_social_analyzer: Optional[SocialMediaAnalyzer] = None


async def get_social_analyzer() -> SocialMediaAnalyzer:
    """Get the global social media analyzer instance"""
    global _social_analyzer
    if _social_analyzer is None:
        _social_analyzer = SocialMediaAnalyzer()
        await _social_analyzer.initialize()
    return _social_analyzer


async def quick_content_performance_analysis(
    platform: SocialPlatform,
    days_back: int = 30
) -> Dict[str, Any]:
    """Quick content performance analysis for the last N days"""
    analyzer = await get_social_analyzer()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    time_range = TimeRange(start=start_date, end=end_date)
    
    return await analyzer.analyze_content_performance(platform, time_range)


async def generate_content_ideas(
    platform: SocialPlatform,
    themes: Optional[List[str]] = None,
    count: int = 3
) -> List[ContentSuggestion]:
    """Generate content ideas for a platform"""
    analyzer = await get_social_analyzer()
    return await analyzer.generate_content_suggestions(platform, themes, count=count)


async def optimize_posting_times(
    platform: SocialPlatform,
    days_back: int = 60
) -> Dict[str, Any]:
    """Optimize posting schedule based on recent performance"""
    analyzer = await get_social_analyzer()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    time_range = TimeRange(start=start_date, end=end_date)
    
    return await analyzer.optimize_posting_schedule(platform, time_range)