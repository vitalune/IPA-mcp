"""
MCP Response Models for Personal Knowledge Assistant

This module defines MCP-compliant response models that ensure proper protocol
adherence for all tool responses and server communications.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum

from mcp.types import TextContent, ImageContent, EmbeddedResource


class ResponseStatus(str, Enum):
    """Standard response status codes."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"


class MCPErrorCode(str, Enum):
    """MCP-compliant error codes."""
    INVALID_REQUEST = "InvalidRequest"
    METHOD_NOT_FOUND = "MethodNotFound"
    INVALID_PARAMS = "InvalidParams"
    INTERNAL_ERROR = "InternalError"
    PARSE_ERROR = "ParseError"
    REQUEST_TIMEOUT = "RequestTimeout"
    RESOURCE_NOT_FOUND = "ResourceNotFound"
    PERMISSION_DENIED = "PermissionDenied"
    RATE_LIMITED = "RateLimited"


class MCPError(BaseModel):
    """MCP-compliant error response."""
    code: MCPErrorCode
    message: str
    data: Optional[Dict[str, Any]] = None


class BaseToolResponse(BaseModel):
    """Base response model for all tool responses."""
    status: ResponseStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_ms: Optional[float] = None
    error: Optional[MCPError] = None


# Email Tool Responses
class EmailSendResponse(BaseToolResponse):
    """Response for send_email tool."""
    message_id: Optional[str] = None
    recipients_count: Optional[int] = None
    scheduled_for: Optional[datetime] = None
    delivery_status: Optional[str] = None


class EmailAnalysisResponse(BaseToolResponse):
    """Response for analyze_email_patterns tool."""
    analysis_type: str
    timeframe: str
    total_emails: int
    patterns: Dict[str, Any]
    insights: List[str]
    recommendations: Optional[List[str]] = []


# Social Media Tool Responses
class SocialMediaPostResponse(BaseToolResponse):
    """Response for post_social_media tool."""
    platforms: List[str]
    post_ids: Dict[str, str]  # platform -> post_id mapping
    scheduled_for: Optional[datetime] = None
    estimated_reach: Optional[Dict[str, int]] = None


class SocialEngagementResponse(BaseToolResponse):
    """Response for analyze_social_engagement tool."""
    platforms: List[str]
    timeframe: str
    total_posts: int
    engagement_metrics: Dict[str, Any]
    top_performing_posts: List[Dict[str, Any]]
    insights: List[str]
    recommendations: Optional[List[str]] = []


# Project Management Tool Responses
class ProjectManagementResponse(BaseToolResponse):
    """Response for manage_project_context tool."""
    action: str
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    affected_count: Optional[int] = None
    projects_summary: Optional[Dict[str, Any]] = None


# Personal Metrics Tool Responses
class PersonalMetricsResponse(BaseToolResponse):
    """Response for track_personal_metrics tool."""
    metric_type: str
    action: str
    entry_id: Optional[str] = None
    current_value: Optional[Union[float, int, str, bool]] = None
    trend_direction: Optional[str] = None  # "up", "down", "stable"
    insights: Optional[List[str]] = []


# Insights Report Tool Responses
class InsightsReportResponse(BaseToolResponse):
    """Response for generate_insights_report tool."""
    report_id: str
    report_type: str
    format: str
    file_path: Optional[str] = None
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    data_sources_used: List[str]
    period_covered: Dict[str, str]  # start_date, end_date


# Generic Data Response Models
class DataListResponse(BaseToolResponse):
    """Generic response for list operations."""
    items: List[Dict[str, Any]]
    total_count: int
    page_size: Optional[int] = None
    page_number: Optional[int] = None
    has_more: bool = False


class DataItemResponse(BaseToolResponse):
    """Generic response for single item operations."""
    item: Dict[str, Any]
    item_id: str
    item_type: str


# MCP Content Wrappers
class MCPTextResponse(BaseModel):
    """Wrapper for MCP TextContent responses."""
    content: List[TextContent]
    
    @classmethod
    def from_text(cls, text: str) -> "MCPTextResponse":
        """Create response from plain text."""
        return cls(content=[TextContent(type="text", text=text)])
    
    @classmethod
    def from_response(cls, response: BaseToolResponse) -> "MCPTextResponse":
        """Create response from tool response model."""
        if response.error:
            text = f"Error: {response.error.message}"
        else:
            text = response.model_dump_json(indent=2)
        
        return cls(content=[TextContent(type="text", text=text)])


class MCPImageResponse(BaseModel):
    """Wrapper for MCP ImageContent responses."""
    content: List[Union[TextContent, ImageContent]]
    
    @classmethod
    def with_image(cls, image_data: str, mime_type: str, text: Optional[str] = None) -> "MCPImageResponse":
        """Create response with image content."""
        content_list = []
        
        if text:
            content_list.append(TextContent(type="text", text=text))
        
        content_list.append(ImageContent(
            type="image",
            data=image_data,
            mimeType=mime_type
        ))
        
        return cls(content=content_list)


class MCPResourceResponse(BaseModel):
    """Wrapper for MCP EmbeddedResource responses."""
    content: List[Union[TextContent, EmbeddedResource]]
    
    @classmethod
    def with_resource(cls, resource: EmbeddedResource, description: Optional[str] = None) -> "MCPResourceResponse":
        """Create response with embedded resource."""
        content_list = []
        
        if description:
            content_list.append(TextContent(type="text", text=description))
        
        content_list.append(resource)
        
        return cls(content=content_list)


# Analysis and Reporting Response Models
class AnalysisInsight(BaseModel):
    """Individual insight from data analysis."""
    type: str  # e.g., "trend", "anomaly", "pattern", "recommendation"
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    data_points: Optional[Dict[str, Any]] = None
    suggested_actions: Optional[List[str]] = []


class TrendAnalysis(BaseModel):
    """Trend analysis results."""
    metric_name: str
    direction: str  # "increasing", "decreasing", "stable", "volatile"
    change_percentage: Optional[float] = None
    statistical_significance: Optional[float] = Field(None, ge=0.0, le=1.0)
    time_period: str
    data_points: List[Dict[str, Any]] = []


class PerformanceMetrics(BaseModel):
    """Performance metrics summary."""
    metric_name: str
    current_value: Union[float, int]
    previous_value: Optional[Union[float, int]] = None
    change_absolute: Optional[Union[float, int]] = None
    change_percentage: Optional[float] = None
    benchmark_value: Optional[Union[float, int]] = None
    performance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    unit: Optional[str] = None


class ComprehensiveAnalysisResponse(BaseToolResponse):
    """Comprehensive analysis response with multiple data types."""
    analysis_type: str
    period_analyzed: Dict[str, str]
    insights: List[AnalysisInsight]
    trends: List[TrendAnalysis]
    performance_metrics: List[PerformanceMetrics]
    summary_statistics: Dict[str, Any]
    visualizations: Optional[List[Dict[str, Any]]] = []  # Visualization configs
    raw_data_available: bool = False
    export_formats: List[str] = ["json", "csv", "pdf"]


# Validation and Helper Models
class ValidationResult(BaseModel):
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    validated_data: Optional[Dict[str, Any]] = None


class ProcessingStatus(BaseModel):
    """Status of long-running operations."""
    operation_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    estimated_completion: Optional[datetime] = None
    result_available: bool = False
    error_message: Optional[str] = None


# Batch Operation Responses
class BatchOperationResponse(BaseToolResponse):
    """Response for batch operations."""
    operation_id: str
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    processing_status: ProcessingStatus
    item_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []


# Configuration and Settings Responses
class ConfigurationResponse(BaseToolResponse):
    """Response for configuration operations."""
    config_section: str
    settings: Dict[str, Any]
    validation_result: ValidationResult
    restart_required: bool = False
    backup_created: bool = False


# Health and Diagnostics Responses
class HealthCheckResponse(BaseToolResponse):
    """Health check response."""
    service_name: str
    version: str
    uptime_seconds: float
    connected_services: Dict[str, bool]
    performance_metrics: Dict[str, float]
    last_error: Optional[str] = None
    warnings: List[str] = []


class DiagnosticsResponse(BaseToolResponse):
    """Diagnostics response."""
    system_info: Dict[str, Any]
    service_status: Dict[str, str]
    recent_errors: List[Dict[str, Any]]
    performance_stats: Dict[str, Any]
    recommendations: List[str] = []