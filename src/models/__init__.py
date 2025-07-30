"""
Data Models Package

This package contains all Pydantic models for data structures and MCP responses.
"""

from .data_models import (
    # Email models
    EmailAddress,
    EmailAttachment,
    EmailMessage,
    EmailThread,
    
    # Social media models
    SocialMediaPost,
    SocialMediaAccount,
    
    # Project models
    Project,
    ProjectTask,
    
    # Metrics models
    MetricEntry,
    ProductivityMetric,
    HabitMetric,
    GoalMetric,
    MoodMetric,
    
    # Communication patterns
    CommunicationPattern,
    
    # Reports and analysis
    InsightReport,
    
    # Utility models
    TimeRange,
    DataFilter,
    QueryParams,
    
    # Enums
    PriorityLevel,
    ProjectStatus,
    SocialPlatform,
    MetricType,
)

from .response_models import (
    # Base responses
    BaseToolResponse,
    MCPTextResponse,
    MCPImageResponse,
    MCPResourceResponse,
    
    # Tool-specific responses
    EmailSendResponse,
    EmailAnalysisResponse,
    SocialMediaPostResponse,
    SocialEngagementResponse,
    ProjectManagementResponse,
    PersonalMetricsResponse,
    InsightsReportResponse,
    
    # Analysis responses
    ComprehensiveAnalysisResponse,
    TrendAnalysis,
    PerformanceMetrics,
    AnalysisInsight,
    
    # Utility responses
    ValidationResult,
    ProcessingStatus,
    BatchOperationResponse,
    HealthCheckResponse,
    DiagnosticsResponse,
    
    # Enums
    ResponseStatus,
    MCPErrorCode,
)

__all__ = [
    # Data models
    "EmailAddress",
    "EmailAttachment", 
    "EmailMessage",
    "EmailThread",
    "SocialMediaPost",
    "SocialMediaAccount",
    "Project",
    "ProjectTask",
    "MetricEntry",
    "ProductivityMetric",
    "HabitMetric",
    "GoalMetric",
    "MoodMetric",
    "CommunicationPattern",
    "InsightReport",
    "TimeRange",
    "DataFilter",
    "QueryParams",
    
    # Response models
    "BaseToolResponse",
    "MCPTextResponse",
    "MCPImageResponse", 
    "MCPResourceResponse",
    "EmailSendResponse",
    "EmailAnalysisResponse",
    "SocialMediaPostResponse",
    "SocialEngagementResponse",
    "ProjectManagementResponse",
    "PersonalMetricsResponse",
    "InsightsReportResponse",
    "ComprehensiveAnalysisResponse",
    "TrendAnalysis",
    "PerformanceMetrics",
    "AnalysisInsight",
    "ValidationResult",
    "ProcessingStatus",
    "BatchOperationResponse",
    "HealthCheckResponse",
    "DiagnosticsResponse",
    
    # Enums
    "PriorityLevel",
    "ProjectStatus", 
    "SocialPlatform",
    "MetricType",
    "ResponseStatus",
    "MCPErrorCode",
]