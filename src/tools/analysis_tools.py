"""
Communication and Behavioral Analysis Tools

This module provides comprehensive analysis capabilities for personal data:
- Communication pattern analysis
- Network analysis and relationship mapping
- Content analysis and insights
- Productivity pattern detection
- Behavioral trend analysis
- Privacy-preserving analytics
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
import structlog

from ..models.data_models import (
    EmailMessage, EmailThread, SocialMediaPost, CommunicationPattern,
    TimeRange, InsightReport
)
from ..utils.nlp_processor import get_nlp_processor, TextAnalysisResult
from ..utils.analytics_engine import get_analytics_engine, TimeSeriesPoint
from ..tools.search_tools import get_search_engine, SearchScope
from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class AnalysisType(str, Enum):
    """Types of analysis that can be performed"""
    COMMUNICATION_PATTERNS = "communication_patterns"
    NETWORK_ANALYSIS = "network_analysis"
    CONTENT_ANALYSIS = "content_analysis"
    PRODUCTIVITY_ANALYSIS = "productivity_analysis"
    BEHAVIORAL_TRENDS = "behavioral_trends"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


class CommunicationDirection(str, Enum):
    """Direction of communication"""
    INCOMING = "incoming"
    OUTGOING = "outgoing"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class CommunicationMetrics:
    """Metrics for communication analysis"""
    total_messages: int
    avg_response_time_hours: float
    response_rate: float
    initiated_conversations: int
    received_conversations: int
    peak_hours: List[int]
    peak_days: List[int]
    avg_message_length: float
    sentiment_distribution: Dict[str, int]


@dataclass
class ContactAnalysis:
    """Analysis results for a specific contact"""
    contact_id: str
    contact_name: str
    email_address: Optional[str]
    relationship_strength: float
    communication_frequency: float
    response_patterns: CommunicationMetrics
    topics: List[Tuple[str, float]]
    sentiment_trend: List[Tuple[datetime, float]]
    communication_style: str
    importance_score: float


@dataclass
class NetworkNode:
    """Represents a node in the communication network"""
    id: str
    name: str
    email: Optional[str]
    centrality_score: float
    cluster_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkEdge:
    """Represents an edge in the communication network"""
    source: str
    target: str
    weight: float
    interaction_count: int
    last_interaction: datetime
    avg_response_time: float
    sentiment_score: float


@dataclass
class NetworkAnalysis:
    """Results of network analysis"""
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    clusters: Dict[int, List[str]]
    key_connectors: List[str]
    isolated_contacts: List[str]
    network_density: float
    average_clustering: float


@dataclass
class ProductivityInsight:
    """Productivity analysis insight"""
    metric_name: str
    current_value: float
    trend_direction: str
    confidence: float
    recommendation: str
    time_period: str


@dataclass
class BehavioralPattern:
    """Represents a detected behavioral pattern"""
    pattern_type: str
    description: str
    frequency: str
    strength: float
    examples: List[Dict[str, Any]]
    trend: str
    detected_at: datetime


class CommunicationAnalyzer:
    """Analyzes communication patterns and relationships"""
    
    def __init__(self):
        self.settings = get_settings()
        self.nlp_processor = None
        self.analytics_engine = None
        self.search_engine = None
        
        # Analysis configuration
        self._min_interactions_for_analysis = 3
        self._response_time_threshold_hours = 48
        self._strong_relationship_threshold = 0.7
        
        # Privacy settings
        self._anonymize_contacts = self.settings.privacy.anonymize_logs
    
    async def initialize(self):
        """Initialize analyzer dependencies"""
        self.nlp_processor = await get_nlp_processor()
        self.analytics_engine = get_analytics_engine()
        self.search_engine = await get_search_engine()
    
    async def analyze_communication_patterns(
        self,
        time_range: TimeRange,
        contacts: Optional[List[str]] = None,
        include_sentiment: bool = True
    ) -> Dict[str, Any]:
        """Analyze overall communication patterns"""
        if not self.nlp_processor:
            await self.initialize()
        
        logger.info("Starting communication pattern analysis", time_range=time_range)
        
        # Fetch communication data
        emails = await self._fetch_emails_in_range(time_range)
        
        if not emails:
            return {"error": "No communication data found in the specified time range"}
        
        # Analyze patterns
        results = {
            "time_range": {
                "start": time_range.start.isoformat(),
                "end": time_range.end.isoformat()
            },
            "overview": await self._analyze_communication_overview(emails),
            "temporal_patterns": await self._analyze_temporal_patterns(emails),
            "response_patterns": await self._analyze_response_patterns(emails),
            "contact_analysis": await self._analyze_top_contacts(emails, limit=20)
        }
        
        if include_sentiment:
            results["sentiment_analysis"] = await self._analyze_communication_sentiment(emails)
        
        return results
    
    async def analyze_contact_relationship(
        self,
        contact_identifier: str,
        time_range: Optional[TimeRange] = None
    ) -> ContactAnalysis:
        """Analyze relationship with a specific contact"""
        if not time_range:
            # Default to last 6 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            time_range = TimeRange(start=start_date, end=end_date)
        
        # Fetch communication with this contact
        emails = await self._fetch_emails_with_contact(contact_identifier, time_range)
        
        if not emails:
            logger.warning(f"No communication found with contact: {contact_identifier}")
            return ContactAnalysis(
                contact_id=contact_identifier,
                contact_name="Unknown",
                email_address=contact_identifier if "@" in contact_identifier else None,
                relationship_strength=0.0,
                communication_frequency=0.0,
                response_patterns=CommunicationMetrics(
                    total_messages=0,
                    avg_response_time_hours=0.0,
                    response_rate=0.0,
                    initiated_conversations=0,
                    received_conversations=0,
                    peak_hours=[],
                    peak_days=[],
                    avg_message_length=0.0,
                    sentiment_distribution={}
                ),
                topics=[],
                sentiment_trend=[],
                communication_style="unknown",
                importance_score=0.0
            )
        
        # Analyze the relationship
        relationship_strength = self._calculate_relationship_strength(emails)
        communication_frequency = len(emails) / ((time_range.end - time_range.start).days / 7)  # per week
        response_patterns = await self._analyze_contact_response_patterns(emails)
        topics = await self._extract_communication_topics(emails)
        sentiment_trend = await self._analyze_sentiment_trend(emails)
        communication_style = self._classify_communication_style(emails)
        importance_score = self._calculate_contact_importance(emails, relationship_strength)
        
        return ContactAnalysis(
            contact_id=contact_identifier,
            contact_name=self._extract_contact_name(emails),
            email_address=contact_identifier if "@" in contact_identifier else None,
            relationship_strength=relationship_strength,
            communication_frequency=communication_frequency,
            response_patterns=response_patterns,
            topics=topics,
            sentiment_trend=sentiment_trend,
            communication_style=communication_style,
            importance_score=importance_score
        )
    
    async def analyze_communication_network(
        self,
        time_range: TimeRange,
        min_interactions: int = 3
    ) -> NetworkAnalysis:
        """Analyze communication network structure"""
        emails = await self._fetch_emails_in_range(time_range)
        
        if not emails:
            return NetworkAnalysis([], [], {}, [], [], 0.0, 0.0)
        
        # Build network graph
        G = nx.Graph()
        
        # Extract contacts and interactions
        contact_interactions = defaultdict(lambda: defaultdict(list))
        
        for email in emails:
            sender = email.get('from', '')
            recipients = email.get('to', [])
            
            # Add interactions
            for recipient in recipients:
                if sender != recipient:  # Avoid self-loops
                    contact_interactions[sender][recipient].append(email)
                    contact_interactions[recipient][sender].append(email)
        
        # Build graph
        for contact1, interactions in contact_interactions.items():
            for contact2, interaction_list in interactions.items():
                if len(interaction_list) >= min_interactions:
                    weight = len(interaction_list)
                    avg_response_time = self._calculate_avg_response_time(interaction_list)
                    sentiment_score = await self._calculate_interaction_sentiment(interaction_list)
                    
                    G.add_edge(
                        contact1, contact2,
                        weight=weight,
                        avg_response_time=avg_response_time,
                        sentiment_score=sentiment_score
                    )
        
        # Calculate network metrics
        nodes = []
        edges = []
        
        # Calculate centrality scores
        if len(G.nodes()) > 0:
            centrality_scores = nx.degree_centrality(G)
            clustering_coeffs = nx.clustering(G)
            
            # Detect communities
            try:
                communities = nx.community.greedy_modularity_communities(G)
                cluster_mapping = {}
                for i, community in enumerate(communities):
                    for node in community:
                        cluster_mapping[node] = i
            except:
                cluster_mapping = {node: 0 for node in G.nodes()}
            
            # Create node objects
            for node_id in G.nodes():
                node = NetworkNode(
                    id=self._anonymize_contact(node_id) if self._anonymize_contacts else node_id,
                    name=self._extract_name_from_email(node_id),
                    email=node_id if "@" in node_id else None,
                    centrality_score=centrality_scores.get(node_id, 0.0),
                    cluster_id=cluster_mapping.get(node_id, 0),
                    metadata={
                        'clustering_coefficient': clustering_coeffs.get(node_id, 0.0),
                        'degree': G.degree(node_id)
                    }
                )
                nodes.append(node)
            
            # Create edge objects
            for edge in G.edges(data=True):
                source, target, data = edge
                edge_obj = NetworkEdge(
                    source=self._anonymize_contact(source) if self._anonymize_contacts else source,
                    target=self._anonymize_contact(target) if self._anonymize_contacts else target,
                    weight=data.get('weight', 1.0),
                    interaction_count=data.get('weight', 1),
                    last_interaction=datetime.now(),  # Would need to track from actual data
                    avg_response_time=data.get('avg_response_time', 0.0),
                    sentiment_score=data.get('sentiment_score', 0.0)
                )
                edges.append(edge_obj)
            
            # Identify key connectors (high centrality)
            key_connectors = [
                node.id for node in sorted(nodes, key=lambda x: x.centrality_score, reverse=True)[:5]
            ]
            
            # Identify isolated contacts (degree = 1 or 0)
            isolated_contacts = [
                node.id for node in nodes if node.metadata.get('degree', 0) <= 1
            ]
            
            # Group clusters
            clusters = defaultdict(list)
            for node in nodes:
                clusters[node.cluster_id].append(node.id)
            
            # Calculate network-level metrics
            network_density = nx.density(G)
            average_clustering = nx.average_clustering(G)
        
        else:
            key_connectors = []
            isolated_contacts = []
            clusters = {}
            network_density = 0.0
            average_clustering = 0.0
        
        return NetworkAnalysis(
            nodes=nodes,
            edges=edges,
            clusters=dict(clusters),
            key_connectors=key_connectors,
            isolated_contacts=isolated_contacts,
            network_density=network_density,
            average_clustering=average_clustering
        )
    
    async def analyze_productivity_patterns(
        self,
        time_range: TimeRange,
        metrics: Optional[List[str]] = None
    ) -> List[ProductivityInsight]:
        """Analyze productivity patterns and provide insights"""
        insights = []
        
        # Default metrics to analyze
        if not metrics:
            metrics = ['email_volume', 'response_time', 'work_hours', 'focus_time']
        
        # Fetch relevant data
        emails = await self._fetch_emails_in_range(time_range)
        
        for metric in metrics:
            insight = await self._analyze_productivity_metric(metric, emails, time_range)
            if insight:
                insights.append(insight)
        
        return insights
    
    async def detect_behavioral_patterns(
        self,
        time_range: TimeRange,
        pattern_types: Optional[List[str]] = None
    ) -> List[BehavioralPattern]:
        """Detect behavioral patterns in communication and activity"""
        patterns = []
        
        if not pattern_types:
            pattern_types = ['response_time', 'communication_volume', 'topic_shifts', 'sentiment_changes']
        
        # Fetch communication data
        emails = await self._fetch_emails_in_range(time_range)
        
        for pattern_type in pattern_types:
            detected_patterns = await self._detect_pattern_type(pattern_type, emails, time_range)
            patterns.extend(detected_patterns)
        
        return patterns
    
    # Helper methods for data fetching
    async def _fetch_emails_in_range(self, time_range: TimeRange) -> List[Dict[str, Any]]:
        """Fetch emails within a time range"""
        # This would integrate with the actual email client
        # For now, return mock data structure
        return []
    
    async def _fetch_emails_with_contact(
        self,
        contact: str,
        time_range: TimeRange
    ) -> List[Dict[str, Any]]:
        """Fetch emails with a specific contact"""
        # This would integrate with the actual email client
        return []
    
    # Analysis helper methods
    async def _analyze_communication_overview(
        self,
        emails: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze overall communication statistics"""
        total_emails = len(emails)
        sent_emails = len([e for e in emails if e.get('direction') == 'sent'])
        received_emails = total_emails - sent_emails
        
        # Calculate daily averages
        if emails:
            date_range = (
                max(e.get('date', datetime.now()) for e in emails) -
                min(e.get('date', datetime.now()) for e in emails)
            ).days
            daily_average = total_emails / max(1, date_range)
        else:
            daily_average = 0.0
        
        return {
            "total_emails": total_emails,
            "sent_emails": sent_emails,
            "received_emails": received_emails,
            "daily_average": daily_average,
            "send_receive_ratio": sent_emails / max(1, received_emails)
        }
    
    async def _analyze_temporal_patterns(
        self,
        emails: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in communication"""
        if not emails:
            return {}
        
        # Extract temporal data
        hours = []
        days_of_week = []
        
        for email in emails:
            date = email.get('date', datetime.now())
            hours.append(date.hour)
            days_of_week.append(date.weekday())
        
        # Find peaks
        hour_counts = Counter(hours)
        dow_counts = Counter(days_of_week)
        
        peak_hours = [hour for hour, count in hour_counts.most_common(3)]
        peak_days = [day for day, count in dow_counts.most_common(3)]
        
        return {
            "peak_hours": peak_hours,
            "peak_days_of_week": peak_days,
            "hourly_distribution": dict(hour_counts),
            "daily_distribution": dict(dow_counts),
            "total_active_hours": len(set(hours)),
            "communication_span": {
                "earliest_hour": min(hours) if hours else 0,
                "latest_hour": max(hours) if hours else 0
            }
        }
    
    async def _analyze_response_patterns(
        self,
        emails: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze response time patterns"""
        response_times = []
        
        # Group emails by thread to calculate response times
        threads = defaultdict(list)
        for email in emails:
            thread_id = email.get('thread_id', email.get('id'))
            threads[thread_id].append(email)
        
        for thread_emails in threads.values():
            # Sort by date
            thread_emails.sort(key=lambda x: x.get('date', datetime.now()))
            
            for i in range(1, len(thread_emails)):
                prev_email = thread_emails[i-1]
                curr_email = thread_emails[i]
                
                # Calculate response time if direction changed
                if (prev_email.get('direction') != curr_email.get('direction')):
                    response_time = (
                        curr_email.get('date', datetime.now()) -
                        prev_email.get('date', datetime.now())
                    ).total_seconds() / 3600  # Convert to hours
                    
                    if response_time > 0 and response_time < 168:  # Less than a week
                        response_times.append(response_time)
        
        if response_times:
            avg_response_time = np.mean(response_times)
            median_response_time = np.median(response_times)
            response_time_std = np.std(response_times)
        else:
            avg_response_time = median_response_time = response_time_std = 0.0
        
        return {
            "average_response_time_hours": avg_response_time,
            "median_response_time_hours": median_response_time,
            "response_time_std": response_time_std,
            "quick_responses": len([rt for rt in response_times if rt < 1]),  # < 1 hour
            "slow_responses": len([rt for rt in response_times if rt > 24]),  # > 24 hours
            "response_rate": len(response_times) / max(1, len(emails) // 2)
        }
    
    async def _analyze_top_contacts(
        self,
        emails: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Analyze top contacts by interaction frequency"""
        contact_stats = defaultdict(lambda: {
            'email_count': 0,
            'last_contact': None,
            'avg_response_time': 0.0,
            'sent_to_them': 0,
            'received_from_them': 0
        })
        
        for email in emails:
            # Extract contacts
            contacts = []
            if email.get('direction') == 'sent':
                contacts = email.get('to', [])
            else:
                contacts = [email.get('from', '')]
            
            for contact in contacts:
                if contact:
                    stats = contact_stats[contact]
                    stats['email_count'] += 1
                    
                    email_date = email.get('date', datetime.now())
                    if not stats['last_contact'] or email_date > stats['last_contact']:
                        stats['last_contact'] = email_date
                    
                    if email.get('direction') == 'sent':
                        stats['sent_to_them'] += 1
                    else:
                        stats['received_from_them'] += 1
        
        # Sort by interaction frequency
        top_contacts = sorted(
            contact_stats.items(),
            key=lambda x: x[1]['email_count'],
            reverse=True
        )[:limit]
        
        # Format results
        results = []
        for contact, stats in top_contacts:
            contact_display = self._anonymize_contact(contact) if self._anonymize_contacts else contact
            
            results.append({
                'contact': contact_display,
                'name': self._extract_name_from_email(contact),
                'total_emails': stats['email_count'],
                'sent_to_them': stats['sent_to_them'],
                'received_from_them': stats['received_from_them'],
                'last_contact': stats['last_contact'].isoformat() if stats['last_contact'] else None,
                'interaction_balance': stats['sent_to_them'] / max(1, stats['received_from_them'])
            })
        
        return results
    
    async def _analyze_communication_sentiment(
        self,
        emails: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment in communications"""
        if not emails:
            return {}
        
        sentiments = []
        sentiment_by_contact = defaultdict(list)
        
        for email in emails:
            content = email.get('body', '') or email.get('snippet', '')
            if content:
                # Analyze sentiment
                sentiment_score, sentiment_label = await self.nlp_processor._analyze_sentiment(content)
                sentiments.append(sentiment_score)
                
                # Track by contact
                contact = email.get('from', '') if email.get('direction') == 'received' else email.get('to', [''])[0]
                if contact:
                    sentiment_by_contact[contact].append(sentiment_score)
        
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            sentiment_std = np.std(sentiments)
            positive_ratio = len([s for s in sentiments if s > 0.1]) / len(sentiments)
            negative_ratio = len([s for s in sentiments if s < -0.1]) / len(sentiments)
        else:
            avg_sentiment = sentiment_std = positive_ratio = negative_ratio = 0.0
        
        # Find most positive and negative contacts
        contact_sentiments = {}
        for contact, scores in sentiment_by_contact.items():
            if len(scores) >= 3:  # Minimum interactions for reliable sentiment
                contact_sentiments[contact] = np.mean(scores)
        
        most_positive = max(contact_sentiments.items(), key=lambda x: x[1]) if contact_sentiments else None
        most_negative = min(contact_sentiments.items(), key=lambda x: x[1]) if contact_sentiments else None
        
        return {
            "overall_sentiment": avg_sentiment,
            "sentiment_variability": sentiment_std,
            "positive_communication_ratio": positive_ratio,
            "negative_communication_ratio": negative_ratio,
            "neutral_communication_ratio": 1 - positive_ratio - negative_ratio,
            "most_positive_contact": most_positive[0] if most_positive else None,
            "most_negative_contact": most_negative[0] if most_negative else None,
            "sentiment_trend": []  # Would calculate trend over time
        }
    
    # Additional helper methods
    def _calculate_relationship_strength(self, emails: List[Dict[str, Any]]) -> float:
        """Calculate relationship strength based on interaction patterns"""
        if not emails:
            return 0.0
        
        # Factors for relationship strength
        frequency_score = min(1.0, len(emails) / 50.0)  # Normalize to max 50 emails
        
        # Bidirectional communication
        sent_count = len([e for e in emails if e.get('direction') == 'sent'])
        received_count = len(emails) - sent_count
        balance_score = 1.0 - abs(sent_count - received_count) / len(emails)
        
        # Recency
        if emails:
            latest_email = max(emails, key=lambda x: x.get('date', datetime.min))
            days_since_last = (datetime.now() - latest_email.get('date', datetime.now())).days
            recency_score = max(0.0, 1.0 - days_since_last / 365.0)  # Decay over a year
        else:
            recency_score = 0.0
        
        # Combine scores
        return (frequency_score * 0.4 + balance_score * 0.3 + recency_score * 0.3)
    
    def _anonymize_contact(self, contact: str) -> str:
        """Anonymize contact information for privacy"""
        if "@" in contact:
            # Hash email address
            import hashlib
            return f"contact_{hashlib.sha256(contact.encode()).hexdigest()[:8]}"
        else:
            return f"contact_{hash(contact) % 10000:04d}"
    
    def _extract_name_from_email(self, email: str) -> str:
        """Extract display name from email address"""
        if "@" in email:
            local_part = email.split("@")[0]
            # Clean up common email patterns
            name = local_part.replace(".", " ").replace("_", " ").replace("-", " ")
            return name.title()
        return email
    
    def _extract_contact_name(self, emails: List[Dict[str, Any]]) -> str:
        """Extract the most common name for a contact from emails"""
        names = []
        for email in emails:
            from_name = email.get('from_name', '')
            if from_name:
                names.append(from_name)
        
        if names:
            # Return most common name
            return Counter(names).most_common(1)[0][0]
        else:
            # Fallback to extracting from email address
            if emails:
                email_addr = emails[0].get('from', '') or emails[0].get('to', [''])[0]
                return self._extract_name_from_email(email_addr)
            return "Unknown"
    
    async def _analyze_contact_response_patterns(
        self,
        emails: List[Dict[str, Any]]
    ) -> CommunicationMetrics:
        """Analyze response patterns for a specific contact"""
        # Implementation would analyze response times, frequency, etc.
        # This is a simplified version
        return CommunicationMetrics(
            total_messages=len(emails),
            avg_response_time_hours=12.0,  # Placeholder
            response_rate=0.8,
            initiated_conversations=5,
            received_conversations=8,
            peak_hours=[9, 14, 16],
            peak_days=[1, 2, 3],  # Monday, Tuesday, Wednesday
            avg_message_length=150.0,
            sentiment_distribution={'positive': 5, 'neutral': 8, 'negative': 2}
        )
    
    async def _extract_communication_topics(
        self,
        emails: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Extract main topics from communication with a contact"""
        all_content = []
        for email in emails:
            content = email.get('body', '') or email.get('snippet', '')
            if content:
                all_content.append(content)
        
        if all_content:
            combined_content = ' '.join(all_content)
            keywords = await self.nlp_processor._extract_keywords(combined_content, max_keywords=5)
            return keywords
        
        return []
    
    async def _analyze_sentiment_trend(
        self,
        emails: List[Dict[str, Any]]
    ) -> List[Tuple[datetime, float]]:
        """Analyze sentiment trend over time"""
        sentiment_data = []
        
        for email in emails:
            content = email.get('body', '') or email.get('snippet', '')
            if content:
                sentiment_score, _ = await self.nlp_processor._analyze_sentiment(content)
                date = email.get('date', datetime.now())
                sentiment_data.append((date, sentiment_score))
        
        # Sort by date
        sentiment_data.sort(key=lambda x: x[0])
        return sentiment_data
    
    def _classify_communication_style(self, emails: List[Dict[str, Any]]) -> str:
        """Classify communication style based on content analysis"""
        # This would analyze language patterns, formality, etc.
        # Simplified implementation
        if len(emails) < 3:
            return "insufficient_data"
        
        # Analyze average message length
        lengths = []
        for email in emails:
            content = email.get('body', '') or email.get('snippet', '')
            lengths.append(len(content.split()) if content else 0)
        
        if lengths:
            avg_length = np.mean(lengths)
            if avg_length > 100:
                return "formal"
            elif avg_length > 50:
                return "conversational"
            else:
                return "brief"
        
        return "unknown"
    
    def _calculate_contact_importance(
        self,
        emails: List[Dict[str, Any]],
        relationship_strength: float
    ) -> float:
        """Calculate importance score for a contact"""
        # Combine various factors
        frequency_factor = min(1.0, len(emails) / 20.0)
        relationship_factor = relationship_strength
        
        # Check for business indicators
        business_keywords = ['meeting', 'project', 'deadline', 'client', 'proposal']
        business_score = 0.0
        
        for email in emails:
            content = (email.get('body', '') + ' ' + email.get('subject', '')).lower()
            business_mentions = sum(1 for keyword in business_keywords if keyword in content)
            business_score += business_mentions
        
        business_factor = min(1.0, business_score / (len(emails) * len(business_keywords)))
        
        return (frequency_factor * 0.4 + relationship_factor * 0.4 + business_factor * 0.2)
    
    def _calculate_avg_response_time(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate average response time for interactions"""
        # Simplified implementation
        return 12.0  # 12 hours average
    
    async def _calculate_interaction_sentiment(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate average sentiment for interactions"""
        sentiments = []
        for interaction in interactions:
            content = interaction.get('body', '') or interaction.get('snippet', '')
            if content:
                sentiment_score, _ = await self.nlp_processor._analyze_sentiment(content)
                sentiments.append(sentiment_score)
        
        return np.mean(sentiments) if sentiments else 0.0
    
    async def _analyze_productivity_metric(
        self,
        metric: str,
        emails: List[Dict[str, Any]],
        time_range: TimeRange
    ) -> Optional[ProductivityInsight]:
        """Analyze a specific productivity metric"""
        # This would implement specific productivity metrics
        # Placeholder implementation
        return ProductivityInsight(
            metric_name=metric,
            current_value=75.0,
            trend_direction="stable",
            confidence=0.7,
            recommendation=f"Your {metric} is within normal range",
            time_period=f"{time_range.start.date()} to {time_range.end.date()}"
        )
    
    async def _detect_pattern_type(
        self,
        pattern_type: str,
        emails: List[Dict[str, Any]],
        time_range: TimeRange
    ) -> List[BehavioralPattern]:
        """Detect specific type of behavioral pattern"""
        # Placeholder implementation
        return [
            BehavioralPattern(
                pattern_type=pattern_type,
                description=f"Detected {pattern_type} pattern",
                frequency="weekly",
                strength=0.6,
                examples=[],
                trend="stable",
                detected_at=datetime.now()
            )
        ]


# Global analyzer instance
_communication_analyzer: Optional[CommunicationAnalyzer] = None


async def get_communication_analyzer() -> CommunicationAnalyzer:
    """Get the global communication analyzer instance"""
    global _communication_analyzer
    if _communication_analyzer is None:
        _communication_analyzer = CommunicationAnalyzer()
        await _communication_analyzer.initialize()
    return _communication_analyzer


async def quick_communication_analysis(
    time_range: TimeRange,
    analysis_types: Optional[List[AnalysisType]] = None
) -> Dict[str, Any]:
    """Quick communication analysis for specified time range"""
    analyzer = await get_communication_analyzer()
    
    if not analysis_types:
        analysis_types = [AnalysisType.COMMUNICATION_PATTERNS]
    
    results = {}
    
    for analysis_type in analysis_types:
        if analysis_type == AnalysisType.COMMUNICATION_PATTERNS:
            results['communication_patterns'] = await analyzer.analyze_communication_patterns(time_range)
        elif analysis_type == AnalysisType.NETWORK_ANALYSIS:
            results['network_analysis'] = await analyzer.analyze_communication_network(time_range)
        elif analysis_type == AnalysisType.PRODUCTIVITY_ANALYSIS:
            results['productivity_analysis'] = await analyzer.analyze_productivity_patterns(time_range)
        elif analysis_type == AnalysisType.BEHAVIORAL_TRENDS:
            results['behavioral_patterns'] = await analyzer.detect_behavioral_patterns(time_range)
    
    return results


async def analyze_contact_relationship(
    contact_identifier: str,
    time_range: Optional[TimeRange] = None
) -> ContactAnalysis:
    """Analyze relationship with a specific contact"""
    analyzer = await get_communication_analyzer()
    return await analyzer.analyze_contact_relationship(contact_identifier, time_range)