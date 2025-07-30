"""
Intelligent Task Management and Project Intelligence Tools

This module provides comprehensive task and project management capabilities:
- Task extraction from emails and messages
- Follow-up detection and deadline tracking
- Project context aggregation and analysis
- Priority scoring and collaboration analysis
- Productivity insights and workflow optimization
- Intelligent reminders and scheduling
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
from sklearn.naive_bayes import MultinomialNB
import structlog

from ..models.data_models import (
    ProjectTask, Project, PriorityLevel, ProjectStatus,
    EmailMessage, TimeRange
)
from ..integrations.client_manager import get_client_manager
from ..utils.nlp_processor import get_nlp_processor, TextAnalysisResult, UrgencyLevel
from ..utils.analytics_engine import get_analytics_engine, TimeSeriesPoint
from ..tools.search_tools import get_search_engine, SearchScope
from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class TaskSource(str, Enum):
    """Sources where tasks can be extracted from"""
    EMAIL = "email"
    CALENDAR = "calendar"
    DOCUMENT = "document"
    CHAT = "chat"
    MANUAL = "manual"


class TaskType(str, Enum):
    """Types of tasks"""
    ACTION_ITEM = "action_item"
    FOLLOW_UP = "follow_up"
    DEADLINE = "deadline"
    MEETING = "meeting"
    REVIEW = "review"
    APPROVAL = "approval"
    RESEARCH = "research"
    COMMUNICATION = "communication"


class CollaborationType(str, Enum):
    """Types of collaboration patterns"""
    SOLO = "solo"
    PAIR = "pair"
    TEAM = "team"
    CROSS_FUNCTIONAL = "cross_functional"
    EXTERNAL = "external"


@dataclass
class ExtractedTask:
    """A task extracted from content"""
    id: str
    title: str
    description: str
    source: TaskSource
    source_id: str
    task_type: TaskType
    priority: PriorityLevel
    urgency: UrgencyLevel
    assignee: Optional[str]
    deadline: Optional[datetime]
    context: Dict[str, Any]
    confidence: float
    extracted_at: datetime
    keywords: List[str] = field(default_factory=list)
    related_people: List[str] = field(default_factory=list)


@dataclass
class FollowUpItem:
    """A follow-up item that needs attention"""
    id: str
    original_message_id: str
    follow_up_type: str
    description: str
    expected_response_from: Optional[str]
    created_at: datetime
    due_date: Optional[datetime]
    priority: PriorityLevel
    status: str
    related_tasks: List[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    """Aggregated context for a project"""
    project_id: str
    project_name: str
    related_emails: List[str]
    related_documents: List[str]
    related_meetings: List[str]
    key_participants: List[str]
    timeline: List[Tuple[datetime, str]]
    current_status: str
    next_actions: List[ExtractedTask]
    risks: List[str]
    dependencies: List[str]


@dataclass
class ProductivityInsight:
    """Productivity insights from task analysis"""
    metric_name: str
    current_value: float
    trend: str
    benchmark: Optional[float]
    recommendations: List[str]
    confidence: float
    time_period: str


@dataclass
class CollaborationPattern:
    """Detected collaboration pattern"""
    pattern_type: CollaborationType
    participants: List[str]
    frequency: float
    effectiveness_score: float
    communication_channels: List[str]
    typical_duration: float
    success_indicators: Dict[str, float]


class TaskIntelligenceEngine:
    """Main engine for task and project intelligence"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client_manager = None
        self.nlp_processor = None
        self.analytics_engine = None
        self.search_engine = None
        
        # Task extraction patterns
        self._task_patterns = self._initialize_task_patterns()
        self._deadline_patterns = self._initialize_deadline_patterns()
        self._priority_keywords = self._initialize_priority_keywords()
        
        # Machine learning models (would be trained in production)
        self._task_classifier = None
        self._priority_classifier = None
        
        # Caching
        self._extracted_tasks = {}
        self._project_contexts = {}
        self._collaboration_cache = {}
    
    async def initialize(self):
        """Initialize the task intelligence engine"""
        self.client_manager = await get_client_manager()
        self.nlp_processor = await get_nlp_processor()
        self.analytics_engine = get_analytics_engine()
        self.search_engine = await get_search_engine()
        
        # Initialize ML models (placeholder)
        await self._initialize_ml_models()
    
    def _initialize_task_patterns(self) -> Dict[TaskType, List[str]]:
        """Initialize regex patterns for task detection"""
        return {
            TaskType.ACTION_ITEM: [
                r'\b(?:please|could you|can you|need to|should|must)\s+(.+?)(?:\.|$)',
                r'\b(?:action item|todo|task):\s*(.+?)(?:\.|$)',
                r'\b(?:i will|we will|you should|let\'s)\s+(.+?)(?:\.|$)',
                r'^\s*[-*•]\s*(.+?)(?:\.|$)'  # Bullet points
            ],
            TaskType.FOLLOW_UP: [
                r'\b(?:follow up|check back|circle back|get back to)\s+(.+?)(?:\.|$)',
                r'\b(?:pending|waiting for|expecting)\s+(.+?)(?:\.|$)',
                r'\b(?:let me know|update me|keep me posted)\s+(.+?)(?:\.|$)'
            ],
            TaskType.DEADLINE: [
                r'\b(?:due|deadline|by|before)\s+(.+?)(?:\.|$)',
                r'\b(?:needs to be|must be)\s+(?:done|completed|finished)\s+(.+?)(?:\.|$)',
                r'\b(?:deliver|submit|provide)\s+(.+?)\s+(?:by|before)\s+(.+?)(?:\.|$)'
            ],
            TaskType.MEETING: [
                r'\b(?:meeting|call|discussion|conference)\s+(.+?)(?:\.|$)',
                r'\b(?:let\'s meet|schedule|arrange)\s+(.+?)(?:\.|$)',
                r'\b(?:calendar|appointment|booking)\s+(.+?)(?:\.|$)'
            ],
            TaskType.REVIEW: [
                r'\b(?:review|check|verify|validate)\s+(.+?)(?:\.|$)',
                r'\b(?:look at|examine|assess)\s+(.+?)(?:\.|$)',
                r'\b(?:feedback|comments)\s+on\s+(.+?)(?:\.|$)'
            ],
            TaskType.APPROVAL: [
                r'\b(?:approve|approval|sign off)\s+(.+?)(?:\.|$)',
                r'\b(?:authorize|permit|allow)\s+(.+?)(?:\.|$)',
                r'\b(?:needs approval|requires approval)\s+(.+?)(?:\.|$)'
            ]
        }
    
    def _initialize_deadline_patterns(self) -> List[str]:
        """Initialize patterns for deadline extraction"""
        return [
            r'\b(?:by|before|due)\s+((?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}))',
            r'\b(?:deadline|due date):\s*(.+?)(?:\.|$)',
            r'\b(?:eod|end of day|by close of business|cob)\b',
            r'\b(?:this week|next week|end of week|eow)\b',
            r'\b(?:asap|as soon as possible|urgent|immediately)\b'
        ]
    
    def _initialize_priority_keywords(self) -> Dict[PriorityLevel, List[str]]:
        """Initialize keywords for priority classification"""
        return {
            PriorityLevel.URGENT: [
                'urgent', 'critical', 'emergency', 'asap', 'immediately',
                'high priority', 'top priority', 'rush', 'escalated'
            ],
            PriorityLevel.HIGH: [
                'important', 'priority', 'significant', 'key', 'major',
                'deadline approaching', 'time sensitive', 'crucial'
            ],
            PriorityLevel.MEDIUM: [
                'moderate', 'standard', 'normal', 'regular', 'routine',
                'when possible', 'at your convenience'
            ],
            PriorityLevel.LOW: [
                'low priority', 'minor', 'whenever', 'eventually',
                'nice to have', 'optional', 'if time permits'
            ]
        }
    
    async def _initialize_ml_models(self):
        """Initialize machine learning models for task classification"""
        # In production, these would be trained models
        # For now, using rule-based classification
        self._task_classifier = "rule-based"
        self._priority_classifier = "rule-based"
    
    async def extract_tasks_from_email(
        self,
        email_content: str,
        email_metadata: Dict[str, Any]
    ) -> List[ExtractedTask]:
        """Extract tasks from email content"""
        if not self.nlp_processor:
            await self.initialize()
        
        # Analyze the email content
        analysis = await self.nlp_processor.analyze_text(
            email_content,
            include_entities=True,
            include_topics=True
        )
        
        extracted_tasks = []
        
        # Extract tasks using pattern matching
        pattern_tasks = self._extract_tasks_by_patterns(email_content, email_metadata)
        extracted_tasks.extend(pattern_tasks)
        
        # Extract tasks using NLP analysis
        nlp_tasks = await self._extract_tasks_by_nlp(email_content, analysis, email_metadata)
        extracted_tasks.extend(nlp_tasks)
        
        # Remove duplicates and refine
        unique_tasks = self._deduplicate_tasks(extracted_tasks)
        refined_tasks = await self._refine_extracted_tasks(unique_tasks, email_content)
        
        return refined_tasks
    
    def _extract_tasks_by_patterns(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[ExtractedTask]:
        """Extract tasks using regex patterns"""
        tasks = []
        
        for task_type, patterns in self._task_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    task_text = match.group(1) if match.groups() else match.group(0)
                    task_text = task_text.strip()
                    
                    if len(task_text) > 10:  # Filter out very short matches
                        task = ExtractedTask(
                            id=f"task_{len(tasks)}_{hash(task_text) % 10000}",
                            title=task_text[:100],  # Truncate long titles
                            description=task_text,
                            source=TaskSource.EMAIL,
                            source_id=metadata.get('id', ''),
                            task_type=task_type,
                            priority=self._classify_priority(task_text),
                            urgency=self._classify_urgency(task_text),
                            assignee=self._extract_assignee(task_text, metadata),
                            deadline=self._extract_deadline(task_text),
                            context=metadata,
                            confidence=0.7,  # Pattern-based confidence
                            extracted_at=datetime.now()
                        )
                        
                        tasks.append(task)
        
        return tasks
    
    async def _extract_tasks_by_nlp(
        self,
        content: str,
        analysis: TextAnalysisResult,
        metadata: Dict[str, Any]
    ) -> List[ExtractedTask]:
        """Extract tasks using NLP analysis"""
        tasks = []
        
        # Use urgency and category from NLP analysis
        if analysis.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.URGENT]:
            # High urgency content likely contains tasks
            sentences = content.split('.')
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) > 20:  # Minimum sentence length
                    # Check if sentence contains task indicators
                    task_indicators = ['need', 'should', 'must', 'will', 'please', 'can you']
                    if any(indicator in sentence.lower() for indicator in task_indicators):
                        task = ExtractedTask(
                            id=f"nlp_task_{i}_{hash(sentence) % 10000}",
                            title=sentence[:100],
                            description=sentence,
                            source=TaskSource.EMAIL,
                            source_id=metadata.get('id', ''),
                            task_type=TaskType.ACTION_ITEM,
                            priority=self._urgency_to_priority(analysis.urgency_level),
                            urgency=analysis.urgency_level,
                            assignee=self._extract_assignee(sentence, metadata),
                            deadline=self._extract_deadline(sentence),
                            context=metadata,
                            confidence=0.6,  # NLP-based confidence
                            extracted_at=datetime.now(),
                            keywords=[kw[0] for kw in analysis.keywords[:5]]
                        )
                        
                        tasks.append(task)
        
        return tasks
    
    def _classify_priority(self, text: str) -> PriorityLevel:
        """Classify task priority based on text content"""
        text_lower = text.lower()
        
        for priority, keywords in self._priority_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return priority
        
        return PriorityLevel.MEDIUM  # Default
    
    def _classify_urgency(self, text: str) -> UrgencyLevel:
        """Classify task urgency"""
        text_lower = text.lower()
        
        urgent_keywords = ['urgent', 'asap', 'immediately', 'emergency', 'critical']
        high_keywords = ['important', 'priority', 'soon', 'deadline', 'today']
        
        if any(keyword in text_lower for keyword in urgent_keywords):
            return UrgencyLevel.URGENT
        elif any(keyword in text_lower for keyword in high_keywords):
            return UrgencyLevel.HIGH
        else:
            return UrgencyLevel.MEDIUM
    
    def _urgency_to_priority(self, urgency: UrgencyLevel) -> PriorityLevel:
        """Convert urgency level to priority level"""
        mapping = {
            UrgencyLevel.URGENT: PriorityLevel.URGENT,
            UrgencyLevel.HIGH: PriorityLevel.HIGH,
            UrgencyLevel.MEDIUM: PriorityLevel.MEDIUM,
            UrgencyLevel.LOW: PriorityLevel.LOW
        }
        return mapping.get(urgency, PriorityLevel.MEDIUM)
    
    def _extract_assignee(self, text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract task assignee from text and metadata"""
        # Look for direct assignments
        assignment_patterns = [
            r'\b(?:assign|assigned to|for)\s+([a-zA-Z\s]+)',
            r'\b([a-zA-Z\s]+)\s+(?:please|could you|can you)',
            r'\b@([a-zA-Z0-9_]+)'  # Mentions
        ]
        
        for pattern in assignment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no explicit assignment, assume sender assigns to recipient
        if metadata.get('direction') == 'received':
            return 'me'  # Task assigned to the user
        
        return None
    
    def _extract_deadline(self, text: str) -> Optional[datetime]:
        """Extract deadline from text"""
        for pattern in self._deadline_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_text = match.group(1) if match.groups() else match.group(0)
                deadline = self._parse_deadline_text(date_text)
                if deadline:
                    return deadline
        
        return None
    
    def _parse_deadline_text(self, date_text: str) -> Optional[datetime]:
        """Parse deadline text into datetime"""
        now = datetime.now()
        date_text = date_text.lower().strip()
        
        # Handle relative dates
        if 'today' in date_text:
            return now.replace(hour=17, minute=0, second=0, microsecond=0)
        elif 'tomorrow' in date_text:
            return (now + timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
        elif 'eod' in date_text or 'end of day' in date_text:
            return now.replace(hour=17, minute=0, second=0, microsecond=0)
        elif 'this week' in date_text or 'eow' in date_text:
            days_until_friday = (4 - now.weekday()) % 7
            return (now + timedelta(days=days_until_friday)).replace(hour=17, minute=0, second=0, microsecond=0)
        elif 'next week' in date_text:
            days_until_next_friday = ((4 - now.weekday()) % 7) + 7
            return (now + timedelta(days=days_until_next_friday)).replace(hour=17, minute=0, second=0, microsecond=0)
        
        # Handle day names
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(days):
            if day in date_text:
                days_ahead = (i - now.weekday()) % 7
                if days_ahead == 0:  # Today
                    days_ahead = 7  # Next week
                return (now + timedelta(days=days_ahead)).replace(hour=17, minute=0, second=0, microsecond=0)
        
        # Try to parse standard date formats
        try:
            import dateutil.parser
            return dateutil.parser.parse(date_text)
        except:
            pass
        
        return None
    
    def _deduplicate_tasks(self, tasks: List[ExtractedTask]) -> List[ExtractedTask]:
        """Remove duplicate tasks based on similarity"""
        if len(tasks) <= 1:
            return tasks
        
        unique_tasks = []
        seen_tasks = set()
        
        for task in tasks:
            # Create a signature for the task
            signature = f"{task.title.lower()[:50]}_{task.task_type.value}"
            
            if signature not in seen_tasks:
                seen_tasks.add(signature)
                unique_tasks.append(task)
        
        return unique_tasks
    
    async def _refine_extracted_tasks(
        self,
        tasks: List[ExtractedTask],
        original_content: str
    ) -> List[ExtractedTask]:
        """Refine and enrich extracted tasks"""
        refined_tasks = []
        
        for task in tasks:
            # Extract related people
            task.related_people = await self._extract_related_people(task.description, original_content)
            
            # Enhance task description
            task.description = await self._enhance_task_description(task, original_content)
            
            # Adjust confidence based on multiple factors
            task.confidence = self._calculate_task_confidence(task, original_content)
            
            if task.confidence > 0.3:  # Only keep tasks with reasonable confidence
                refined_tasks.append(task)
        
        return refined_tasks
    
    async def _extract_related_people(
        self,
        task_description: str,
        context: str
    ) -> List[str]:
        """Extract people related to the task"""
        # Use NLP to extract person entities
        analysis = await self.nlp_processor.analyze_text(context, include_entities=True)
        
        people = []
        for entity in analysis.entities:
            if entity.label in ['PERSON', 'ORG']:
                people.append(entity.text)
        
        return people[:5]  # Limit to 5 most relevant people
    
    async def _enhance_task_description(
        self,
        task: ExtractedTask,
        context: str
    ) -> str:
        """Enhance task description with context"""
        # Add context if the description is too brief
        if len(task.description) < 30:
            # Find surrounding context
            task_pos = context.find(task.title)
            if task_pos != -1:
                start = max(0, task_pos - 100)
                end = min(len(context), task_pos + len(task.title) + 100)
                surrounding_context = context[start:end].strip()
                
                if len(surrounding_context) > len(task.description):
                    return surrounding_context
        
        return task.description
    
    def _calculate_task_confidence(
        self,
        task: ExtractedTask,
        context: str
    ) -> float:
        """Calculate confidence score for extracted task"""
        confidence = task.confidence
        
        # Boost confidence for tasks with deadlines
        if task.deadline:
            confidence += 0.2
        
        # Boost confidence for tasks with specific assignees
        if task.assignee and task.assignee != 'me':
            confidence += 0.1
        
        # Boost confidence for urgent tasks
        if task.urgency in [UrgencyLevel.HIGH, UrgencyLevel.URGENT]:
            confidence += 0.1
        
        # Reduce confidence for very short descriptions
        if len(task.description) < 20:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    async def detect_follow_ups(
        self,
        time_range: TimeRange,
        include_overdue: bool = True
    ) -> List[FollowUpItem]:
        """Detect items that need follow-up"""
        follow_ups = []
        
        # Fetch recent emails
        emails = await self._fetch_emails_in_range(time_range)
        
        for email in emails:
            # Check for follow-up indicators
            if self._contains_follow_up_request(email):
                follow_up = await self._create_follow_up_item(email)
                if follow_up:
                    follow_ups.append(follow_up)
        
        # Check for overdue responses
        if include_overdue:
            overdue_items = await self._detect_overdue_responses(emails)
            follow_ups.extend(overdue_items)
        
        # Sort by priority and due date
        follow_ups.sort(key=lambda x: (x.priority.value, x.due_date or datetime.max))
        
        return follow_ups
    
    def _contains_follow_up_request(self, email: Dict[str, Any]) -> bool:
        """Check if email contains follow-up requests"""
        content = email.get('body', '') + ' ' + email.get('subject', '')
        follow_up_keywords = [
            'follow up', 'check back', 'get back to', 'let me know',
            'update me', 'keep me posted', 'circle back', 'pending',
            'waiting for', 'expecting'
        ]
        
        return any(keyword in content.lower() for keyword in follow_up_keywords)
    
    async def _create_follow_up_item(self, email: Dict[str, Any]) -> Optional[FollowUpItem]:
        """Create a follow-up item from an email"""
        try:
            content = email.get('body', '')
            
            # Extract follow-up type and description
            follow_up_patterns = {
                'response_needed': r'(?:let me know|get back to me|update me)(.+?)(?:\.|$)',
                'status_update': r'(?:check on|follow up on|status of)(.+?)(?:\.|$)',
                'action_pending': r'(?:waiting for|pending|expecting)(.+?)(?:\.|$)'
            }
            
            follow_up_type = 'general'
            description = 'Follow-up needed'
            
            for ftype, pattern in follow_up_patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    follow_up_type = ftype
                    description = match.group(1).strip()
                    break
            
            # Determine who should respond
            expected_response_from = None
            if email.get('direction') == 'sent':
                expected_response_from = email.get('to', [None])[0]
            else:
                expected_response_from = email.get('from')
            
            # Calculate due date (default: 3 business days)
            due_date = self._calculate_follow_up_due_date(content)
            
            return FollowUpItem(
                id=f"followup_{email.get('id', '')[:8]}_{hash(description) % 1000}",
                original_message_id=email.get('id', ''),
                follow_up_type=follow_up_type,
                description=description,
                expected_response_from=expected_response_from,
                created_at=email.get('date', datetime.now()),
                due_date=due_date,
                priority=self._classify_priority(content),
                status='pending'
            )
        
        except Exception as e:
            logger.error(f"Failed to create follow-up item: {e}")
            return None
    
    def _calculate_follow_up_due_date(self, content: str) -> datetime:
        """Calculate when a follow-up is due"""
        # Look for explicit timeframes
        if any(keyword in content.lower() for keyword in ['urgent', 'asap', 'immediately']):
            return datetime.now() + timedelta(hours=24)
        elif any(keyword in content.lower() for keyword in ['this week', 'soon']):
            return datetime.now() + timedelta(days=3)
        elif any(keyword in content.lower() for keyword in ['next week']):
            return datetime.now() + timedelta(days=7)
        else:
            # Default: 3 business days
            return datetime.now() + timedelta(days=3)
    
    async def _detect_overdue_responses(
        self,
        emails: List[Dict[str, Any]]
    ) -> List[FollowUpItem]:
        """Detect overdue responses in email threads"""
        overdue_items = []
        
        # Group emails by thread
        threads = defaultdict(list)
        for email in emails:
            thread_id = email.get('thread_id', email.get('id'))
            threads[thread_id].append(email)
        
        # Check each thread for overdue responses
        for thread_id, thread_emails in threads.items():
            thread_emails.sort(key=lambda x: x.get('date', datetime.min))
            
            last_sent = None
            last_received = None
            
            for email in thread_emails:
                if email.get('direction') == 'sent':
                    last_sent = email
                else:
                    last_received = email
            
            # Check if we're waiting for a response
            if last_sent and (not last_received or last_sent['date'] > last_received['date']):
                days_waiting = (datetime.now() - last_sent['date']).days
                
                if days_waiting > 2:  # More than 2 days without response
                    overdue_item = FollowUpItem(
                        id=f"overdue_{thread_id}_{days_waiting}",
                        original_message_id=last_sent.get('id', ''),
                        follow_up_type='overdue_response',
                        description=f"No response for {days_waiting} days: {last_sent.get('subject', '')}",
                        expected_response_from=last_sent.get('to', [None])[0],
                        created_at=last_sent.get('date', datetime.now()),
                        due_date=datetime.now(),  # Already overdue
                        priority=PriorityLevel.HIGH if days_waiting > 5 else PriorityLevel.MEDIUM,
                        status='overdue'
                    )
                    overdue_items.append(overdue_item)
        
        return overdue_items
    
    async def aggregate_project_context(
        self,
        project_name: str,
        time_range: Optional[TimeRange] = None
    ) -> ProjectContext:
        """Aggregate context for a project from multiple sources"""
        if not time_range:
            # Default to last 3 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            time_range = TimeRange(start=start_date, end=end_date)
        
        # Search for project-related content
        search_results = await self.search_engine.universal_search(
            query=project_name,
            scope=SearchScope.ALL
        )
        
        # Categorize results
        related_emails = []
        related_documents = []
        related_meetings = []
        
        for result in search_results.results:
            if result.type.value == 'email':
                related_emails.append(result.id)
            elif result.type.value == 'document':
                related_documents.append(result.id)
            elif 'meeting' in result.metadata:
                related_meetings.append(result.id)
        
        # Extract key participants
        key_participants = await self._extract_project_participants(search_results.results)
        
        # Build timeline
        timeline = await self._build_project_timeline(search_results.results)
        
        # Assess current status
        current_status = await self._assess_project_status(search_results.results)
        
        # Extract next actions
        next_actions = await self._extract_project_next_actions(search_results.results)
        
        # Identify risks and dependencies
        risks = await self._identify_project_risks(search_results.results)
        dependencies = await self._identify_project_dependencies(search_results.results)
        
        return ProjectContext(
            project_id=f"project_{hash(project_name) % 100000}",
            project_name=project_name,
            related_emails=related_emails,
            related_documents=related_documents,
            related_meetings=related_meetings,
            key_participants=key_participants,
            timeline=timeline,
            current_status=current_status,
            next_actions=next_actions,
            risks=risks,
            dependencies=dependencies
        )
    
    async def _extract_project_participants(
        self,
        search_results: List[Any]
    ) -> List[str]:
        """Extract key project participants from search results"""
        participant_count = Counter()
        
        for result in search_results:
            # Extract people from metadata
            if 'from' in result.metadata:
                participant_count[result.metadata['from']] += 1
            if 'to' in result.metadata:
                for recipient in result.metadata.get('to', []):
                    participant_count[recipient] += 1
            
            # Extract people from content using NLP
            if hasattr(result, 'content') and result.content:
                analysis = await self.nlp_processor.analyze_text(result.content, include_entities=True)
                for entity in analysis.entities:
                    if entity.label == 'PERSON':
                        participant_count[entity.text] += 0.5  # Lower weight for content mentions
        
        # Return top participants
        return [person for person, count in participant_count.most_common(10)]
    
    async def _build_project_timeline(
        self,
        search_results: List[Any]
    ) -> List[Tuple[datetime, str]]:
        """Build project timeline from search results"""
        timeline_events = []
        
        for result in search_results:
            if result.timestamp:
                event_description = f"{result.type.value}: {result.title[:50]}..."
                timeline_events.append((result.timestamp, event_description))
        
        # Sort by timestamp
        timeline_events.sort(key=lambda x: x[0])
        
        return timeline_events[-20:]  # Return last 20 events
    
    async def _assess_project_status(
        self,
        search_results: List[Any]
    ) -> str:
        """Assess current project status from content"""
        status_keywords = {
            'completed': ['completed', 'finished', 'done', 'delivered', 'launched'],
            'in_progress': ['working on', 'in progress', 'developing', 'building'],
            'blocked': ['blocked', 'stuck', 'waiting', 'delayed', 'issue'],
            'planning': ['planning', 'designing', 'scoping', 'requirements']
        }
        
        status_scores = defaultdict(int)
        
        for result in search_results:
            content = (result.title + ' ' + result.snippet).lower()
            for status, keywords in status_keywords.items():
                for keyword in keywords:
                    if keyword in content:
                        status_scores[status] += 1
        
        if status_scores:
            return max(status_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'unknown'
    
    async def _extract_project_next_actions(
        self,
        search_results: List[Any]
    ) -> List[ExtractedTask]:
        """Extract next actions for the project"""
        next_actions = []
        
        # Look for recent tasks in project-related content
        for result in search_results:
            if result.content:
                tasks = await self.extract_tasks_from_email(
                    result.content,
                    {'id': result.id, 'source': result.source}
                )
                
                # Filter for future-oriented tasks
                for task in tasks:
                    if any(word in task.description.lower() for word in ['will', 'next', 'upcoming', 'plan']):
                        next_actions.append(task)
        
        return next_actions[:5]  # Return top 5 next actions
    
    async def _identify_project_risks(
        self,
        search_results: List[Any]
    ) -> List[str]:
        """Identify potential project risks"""
        risk_indicators = [
            'risk', 'problem', 'issue', 'concern', 'blocker', 'delay',
            'challenge', 'difficulty', 'obstacle', 'threat'
        ]
        
        risks = []
        for result in search_results:
            content = (result.title + ' ' + result.snippet).lower()
            for indicator in risk_indicators:
                if indicator in content:
                    # Extract sentence containing the risk
                    sentences = result.snippet.split('.')
                    for sentence in sentences:
                        if indicator in sentence.lower():
                            risks.append(sentence.strip())
                            break
        
        return list(set(risks))[:5]  # Return unique risks, max 5
    
    async def _identify_project_dependencies(
        self,
        search_results: List[Any]
    ) -> List[str]:
        """Identify project dependencies"""
        dependency_indicators = [
            'depends on', 'waiting for', 'blocked by', 'requires',
            'needs', 'prerequisite', 'after', 'once'
        ]
        
        dependencies = []
        for result in search_results:
            content = (result.title + ' ' + result.snippet).lower()
            for indicator in dependency_indicators:
                if indicator in content:
                    sentences = result.snippet.split('.')
                    for sentence in sentences:
                        if indicator in sentence.lower():
                            dependencies.append(sentence.strip())
                            break
        
        return list(set(dependencies))[:5]  # Return unique dependencies, max 5
    
    async def analyze_productivity_patterns(
        self,
        time_range: TimeRange
    ) -> List[ProductivityInsight]:
        """Analyze productivity patterns from task and email data"""
        insights = []
        
        # Fetch data
        emails = await self._fetch_emails_in_range(time_range)
        tasks = []
        
        # Extract tasks from emails
        for email in emails:
            email_tasks = await self.extract_tasks_from_email(
                email.get('body', ''),
                email
            )
            tasks.extend(email_tasks)
        
        # Analyze task completion patterns
        completion_insight = await self._analyze_task_completion_patterns(tasks, time_range)
        if completion_insight:
            insights.append(completion_insight)
        
        # Analyze email response patterns
        response_insight = await self._analyze_email_response_patterns(emails, time_range)
        if response_insight:
            insights.append(response_insight)
        
        # Analyze workload distribution
        workload_insight = await self._analyze_workload_distribution(tasks, emails, time_range)
        if workload_insight:
            insights.append(workload_insight)
        
        return insights
    
    async def _analyze_task_completion_patterns(
        self,
        tasks: List[ExtractedTask],
        time_range: TimeRange
    ) -> Optional[ProductivityInsight]:
        """Analyze task completion patterns"""
        if not tasks:
            return None
        
        # Calculate task completion rate (placeholder logic)
        total_tasks = len(tasks)
        urgent_tasks = len([t for t in tasks if t.urgency == UrgencyLevel.URGENT])
        completion_rate = 0.75  # Would be calculated from actual completion data
        
        return ProductivityInsight(
            metric_name="Task Completion Rate",
            current_value=completion_rate,
            trend="stable",
            benchmark=0.80,
            recommendations=[
                f"You have {urgent_tasks} urgent tasks out of {total_tasks} total tasks",
                "Consider prioritizing urgent tasks first",
                "Break down large tasks into smaller, manageable pieces"
            ],
            confidence=0.7,
            time_period=f"{time_range.start.date()} to {time_range.end.date()}"
        )
    
    async def _analyze_email_response_patterns(
        self,
        emails: List[Dict[str, Any]],
        time_range: TimeRange
    ) -> Optional[ProductivityInsight]:
        """Analyze email response patterns"""
        if not emails:
            return None
        
        # Calculate average response time
        response_times = []
        threads = defaultdict(list)
        
        for email in emails:
            thread_id = email.get('thread_id', email.get('id'))
            threads[thread_id].append(email)
        
        for thread_emails in threads.values():
            thread_emails.sort(key=lambda x: x.get('date', datetime.min))
            
            for i in range(1, len(thread_emails)):
                prev_email = thread_emails[i-1]
                curr_email = thread_emails[i]
                
                if (prev_email.get('direction') != curr_email.get('direction')):
                    response_time = (
                        curr_email.get('date', datetime.now()) -
                        prev_email.get('date', datetime.now())
                    ).total_seconds() / 3600  # Convert to hours
                    
                    if 0 < response_time < 168:  # Between 0 and 168 hours (1 week)
                        response_times.append(response_time)
        
        if response_times:
            avg_response_time = np.mean(response_times)
        else:
            avg_response_time = 24.0  # Default
        
        return ProductivityInsight(
            metric_name="Email Response Time",
            current_value=avg_response_time,
            trend="stable",
            benchmark=12.0,  # 12 hours benchmark
            recommendations=[
                f"Your average response time is {avg_response_time:.1f} hours",
                "Consider setting specific times for checking email",
                "Use email filters to prioritize important messages"
            ],
            confidence=0.8,
            time_period=f"{time_range.start.date()} to {time_range.end.date()}"
        )
    
    async def _analyze_workload_distribution(
        self,
        tasks: List[ExtractedTask],
        emails: List[Dict[str, Any]],
        time_range: TimeRange
    ) -> Optional[ProductivityInsight]:
        """Analyze workload distribution patterns"""
        # Calculate daily workload
        daily_tasks = defaultdict(int)
        daily_emails = defaultdict(int)
        
        for task in tasks:
            day = task.extracted_at.date()
            daily_tasks[day] += 1
        
        for email in emails:
            day = email.get('date', datetime.now()).date()
            daily_emails[day] += 1
        
        # Calculate workload variance
        task_counts = list(daily_tasks.values())
        email_counts = list(daily_emails.values())
        
        if task_counts:
            task_variance = np.var(task_counts)
            avg_daily_tasks = np.mean(task_counts)
        else:
            task_variance = 0
            avg_daily_tasks = 0
        
        return ProductivityInsight(
            metric_name="Workload Distribution",
            current_value=float(task_variance),
            trend="stable",
            benchmark=2.0,  # Low variance is better
            recommendations=[
                f"Average {avg_daily_tasks:.1f} tasks per day",
                "Consider spreading workload more evenly across days",
                "Use time blocking to manage daily capacity"
            ],
            confidence=0.6,
            time_period=f"{time_range.start.date()} to {time_range.end.date()}"
        )
    
    async def detect_collaboration_patterns(
        self,
        time_range: TimeRange
    ) -> List[CollaborationPattern]:
        """Detect collaboration patterns from communication data"""
        patterns = []
        
        # Fetch communication data
        emails = await self._fetch_emails_in_range(time_range)
        
        # Analyze communication networks
        collaboration_data = await self._analyze_collaboration_networks(emails)
        
        # Identify different collaboration patterns
        for pattern_data in collaboration_data:
            pattern = CollaborationPattern(
                pattern_type=pattern_data['type'],
                participants=pattern_data['participants'],
                frequency=pattern_data['frequency'],
                effectiveness_score=pattern_data['effectiveness'],
                communication_channels=['email'],  # Would expand to include other channels
                typical_duration=pattern_data['duration'],
                success_indicators=pattern_data['success_metrics']
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _analyze_collaboration_networks(
        self,
        emails: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze collaboration networks from email data"""
        # Build collaboration graph
        G = nx.Graph()
        
        for email in emails:
            sender = email.get('from', '')
            recipients = email.get('to', [])
            
            for recipient in recipients:
                if sender and recipient and sender != recipient:
                    if G.has_edge(sender, recipient):
                        G[sender][recipient]['weight'] += 1
                    else:
                        G.add_edge(sender, recipient, weight=1)
        
        # Detect collaboration patterns
        patterns = []
        
        # Find highly connected pairs (pair collaboration)
        for edge in G.edges(data=True):
            if edge[2]['weight'] > 10:  # Threshold for frequent collaboration
                patterns.append({
                    'type': CollaborationType.PAIR,
                    'participants': [edge[0], edge[1]],
                    'frequency': edge[2]['weight'],
                    'effectiveness': 0.8,  # Would be calculated based on response times, etc.
                    'duration': 30.0,  # Average collaboration duration in days
                    'success_metrics': {'response_rate': 0.9, 'completion_rate': 0.85}
                })
        
        # Find teams (groups with high internal connectivity)
        try:
            communities = nx.community.greedy_modularity_communities(G)
            for community in communities:
                if len(community) > 2:
                    patterns.append({
                        'type': CollaborationType.TEAM,
                        'participants': list(community),
                        'frequency': sum(G[u][v]['weight'] for u, v in G.subgraph(community).edges()) / len(community),
                        'effectiveness': 0.7,
                        'duration': 45.0,
                        'success_metrics': {'coordination_score': 0.75, 'output_quality': 0.8}
                    })
        except:
            pass  # Skip if community detection fails
        
        return patterns
    
    async def _fetch_emails_in_range(self, time_range: TimeRange) -> List[Dict[str, Any]]:
        """Fetch emails within a time range"""
        try:
            gmail_client = await self.client_manager.get_client('gmail')
            if gmail_client:
                return await gmail_client.get_messages(
                    start_date=time_range.start,
                    end_date=time_range.end,
                    max_results=500
                )
        except Exception as e:
            logger.error(f"Failed to fetch emails: {e}")
        
        return []


# Global task intelligence engine instance
_task_engine: Optional[TaskIntelligenceEngine] = None


async def get_task_engine() -> TaskIntelligenceEngine:
    """Get the global task intelligence engine instance"""
    global _task_engine
    if _task_engine is None:
        _task_engine = TaskIntelligenceEngine()
        await _task_engine.initialize()
    return _task_engine


async def extract_tasks_from_content(
    content: str,
    source: TaskSource = TaskSource.EMAIL,
    metadata: Optional[Dict[str, Any]] = None
) -> List[ExtractedTask]:
    """Quick task extraction from any content"""
    engine = await get_task_engine()
    return await engine.extract_tasks_from_email(content, metadata or {})


async def get_follow_ups(days_back: int = 7) -> List[FollowUpItem]:
    """Get follow-up items from the last N days"""
    engine = await get_task_engine()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    time_range = TimeRange(start=start_date, end=end_date)
    
    return await engine.detect_follow_ups(time_range)


async def analyze_project(project_name: str) -> ProjectContext:
    """Quick project context analysis"""
    engine = await get_task_engine()
    return await engine.aggregate_project_context(project_name)