"""
Data Models for Personal Knowledge Assistant MCP Server

This module defines Pydantic models for all data structures used throughout the system.
These models ensure type safety, validation, and serialization consistency.
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, EmailStr, HttpUrl, validator
import uuid


class PriorityLevel(str, Enum):
    """Priority levels for tasks and projects."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ProjectStatus(str, Enum):
    """Project status options."""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class SocialPlatform(str, Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"


class MetricType(str, Enum):
    """Types of personal metrics that can be tracked."""
    PRODUCTIVITY = "productivity"
    HABITS = "habits"
    GOALS = "goals"
    MOOD = "mood"
    ENERGY = "energy"


# Email Data Models
class EmailAddress(BaseModel):
    """Represents an email address with optional display name."""
    email: EmailStr
    name: Optional[str] = None

    def __str__(self) -> str:
        if self.name:
            return f"{self.name} <{self.email}>"
        return str(self.email)


class EmailAttachment(BaseModel):
    """Represents an email attachment."""
    filename: str
    content_type: str
    size_bytes: int
    file_path: Optional[str] = None
    content_id: Optional[str] = None  # For inline attachments


class EmailMessage(BaseModel):
    """Represents an email message."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_id: Optional[str] = None  # External message ID from email provider
    thread_id: Optional[str] = None
    
    # Recipients and sender
    from_address: EmailAddress
    to_addresses: List[EmailAddress]
    cc_addresses: Optional[List[EmailAddress]] = []
    bcc_addresses: Optional[List[EmailAddress]] = []
    reply_to: Optional[EmailAddress] = None
    
    # Content
    subject: str
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    attachments: Optional[List[EmailAttachment]] = []
    
    # Metadata
    sent_at: datetime
    received_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    
    # Flags and labels
    is_read: bool = False
    is_starred: bool = False
    is_important: bool = False
    labels: Optional[List[str]] = []
    
    # Analysis fields
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    priority_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    response_required: Optional[bool] = None


class EmailThread(BaseModel):
    """Represents a conversation thread of emails."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str
    participants: List[EmailAddress]
    messages: List[EmailMessage]
    
    created_at: datetime
    last_message_at: datetime
    message_count: int
    
    # Thread analysis
    avg_response_time_hours: Optional[float] = None
    conversation_sentiment: Optional[float] = Field(None, ge=-1.0, le=1.0)


# Social Media Data Models
class SocialMediaPost(BaseModel):
    """Represents a social media post."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    platform: SocialPlatform
    external_id: Optional[str] = None  # Platform-specific post ID
    
    # Content
    content: str
    media_urls: Optional[List[HttpUrl]] = []
    hashtags: Optional[List[str]] = []
    mentions: Optional[List[str]] = []
    
    # Scheduling and publishing
    scheduled_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    is_published: bool = False
    
    # Engagement metrics
    likes_count: int = 0
    shares_count: int = 0
    comments_count: int = 0
    clicks_count: int = 0
    impressions_count: int = 0
    reach_count: int = 0
    
    # Analysis
    engagement_rate: Optional[float] = Field(None, ge=0.0)
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    performance_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class SocialMediaAccount(BaseModel):
    """Represents a social media account configuration."""
    platform: SocialPlatform
    username: str
    display_name: Optional[str] = None
    account_id: Optional[str] = None
    
    # Authentication (will be handled by security-privacy-guardian)
    is_authenticated: bool = False
    auth_expires_at: Optional[datetime] = None
    
    # Account metrics
    followers_count: Optional[int] = None
    following_count: Optional[int] = None
    posts_count: Optional[int] = None
    
    # Configuration
    auto_post_enabled: bool = False
    optimal_posting_times: Optional[List[str]] = []  # Hour strings like "09:00", "14:30"


# Project and Task Management Models
class ProjectTask(BaseModel):
    """Represents a task within a project."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    
    # Status and priority
    is_completed: bool = False
    priority: PriorityLevel = PriorityLevel.MEDIUM
    
    # Dates
    created_at: datetime = Field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Organization
    tags: Optional[List[str]] = []
    assignee: Optional[str] = None
    
    # Time tracking
    estimated_hours: Optional[float] = Field(None, gt=0)
    actual_hours: Optional[float] = Field(None, gt=0)


class Project(BaseModel):
    """Represents a project with tasks and context."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    
    # Status and priority
    status: ProjectStatus = ProjectStatus.PLANNING
    priority: PriorityLevel = PriorityLevel.MEDIUM
    
    # Dates
    created_at: datetime = Field(default_factory=datetime.now)
    start_date: Optional[date] = None
    due_date: Optional[date] = None
    completed_at: Optional[datetime] = None
    
    # Organization
    tags: Optional[List[str]] = []
    team_members: Optional[List[str]] = []
    
    # Tasks and progress
    tasks: Optional[List[ProjectTask]] = []
    completion_percentage: float = Field(0.0, ge=0.0, le=100.0)
    
    # Context and notes
    notes: Optional[str] = None
    related_emails: Optional[List[str]] = []  # Email IDs
    related_documents: Optional[List[str]] = []  # File paths or URLs


# Personal Metrics Models
class MetricEntry(BaseModel):
    """Base class for metric entries."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metric_type: MetricType
    timestamp: datetime = Field(default_factory=datetime.now)
    value: Union[float, int, str, bool]
    notes: Optional[str] = None
    tags: Optional[List[str]] = []


class ProductivityMetric(MetricEntry):
    """Productivity-specific metric entry."""
    metric_type: MetricType = MetricType.PRODUCTIVITY
    
    # Productivity-specific fields
    hours_worked: Optional[float] = Field(None, ge=0)
    tasks_completed: Optional[int] = Field(None, ge=0)
    focus_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    interruptions_count: Optional[int] = Field(None, ge=0)


class HabitMetric(MetricEntry):
    """Habit tracking metric entry."""
    metric_type: MetricType = MetricType.HABITS
    
    # Habit-specific fields
    habit_name: str
    completed: bool
    streak_count: Optional[int] = Field(None, ge=0)
    target_frequency: Optional[str] = None  # e.g., "daily", "weekly"


class GoalMetric(MetricEntry):
    """Goal tracking metric entry."""
    metric_type: MetricType = MetricType.GOALS
    
    # Goal-specific fields
    goal_name: str
    current_value: float
    target_value: float
    unit: Optional[str] = None
    deadline: Optional[date] = None
    
    @validator('current_value', 'target_value')
    def validate_values(cls, v):
        if v < 0:
            raise ValueError('Values must be non-negative')
        return v


class MoodMetric(MetricEntry):
    """Mood tracking metric entry."""
    metric_type: MetricType = MetricType.MOOD
    
    # Mood-specific fields
    mood_score: float = Field(ge=1.0, le=10.0)  # 1-10 scale
    energy_level: Optional[float] = Field(None, ge=1.0, le=10.0)
    stress_level: Optional[float] = Field(None, ge=1.0, le=10.0)
    sleep_hours: Optional[float] = Field(None, ge=0, le=24)
    weather: Optional[str] = None
    activities: Optional[List[str]] = []


# Communication Pattern Models
class CommunicationPattern(BaseModel):
    """Represents patterns in communication behavior."""
    contact: str  # Email address or social media handle
    contact_type: str  # "email" or social platform name
    
    # Temporal patterns
    avg_response_time_hours: Optional[float] = Field(None, ge=0)
    preferred_contact_hours: Optional[List[int]] = []  # Hours 0-23
    preferred_contact_days: Optional[List[int]] = []  # Days 0-6 (Monday=0)
    
    # Communication frequency
    messages_per_week: Optional[float] = Field(None, ge=0)
    initiated_conversations_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Content analysis
    avg_message_length: Optional[float] = Field(None, ge=0)
    common_topics: Optional[List[str]] = []
    sentiment_trend: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    # Relationship insights
    relationship_strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    communication_style: Optional[str] = None  # e.g., "formal", "casual", "technical"
    
    # Time-based data
    first_contact_date: Optional[date] = None
    last_contact_date: Optional[date] = None
    analysis_period_start: date
    analysis_period_end: date


# Report and Analysis Models
class InsightReport(BaseModel):
    """Represents a generated insights report."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str
    title: str
    
    # Generation metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    period_start: date
    period_end: date
    data_sources: List[str]
    
    # Content
    summary: str
    sections: Dict[str, Any]  # Flexible structure for different report types
    recommendations: Optional[List[str]] = []
    key_metrics: Optional[Dict[str, Any]] = {}
    
    # Export options
    format: str = "markdown"  # markdown, html, json, pdf
    file_path: Optional[str] = None


# Utility Models
class TimeRange(BaseModel):
    """Represents a time range for queries and analysis."""
    start: datetime
    end: datetime
    
    @validator('end')
    def end_after_start(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError('End time must be after start time')
        return v


class DataFilter(BaseModel):
    """Generic filter for data queries."""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, contains
    value: Any
    
    @validator('operator')
    def valid_operator(cls, v):
        valid_ops = {'eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'in', 'contains', 'startswith', 'endswith'}
        if v not in valid_ops:
            raise ValueError(f'Operator must be one of: {valid_ops}')
        return v


class QueryParams(BaseModel):
    """Parameters for data queries."""
    filters: Optional[List[DataFilter]] = []
    time_range: Optional[TimeRange] = None
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # asc or desc
    limit: Optional[int] = Field(None, gt=0, le=10000)
    offset: int = Field(0, ge=0)
    
    @validator('sort_order')
    def valid_sort_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError('Sort order must be "asc" or "desc"')
        return v