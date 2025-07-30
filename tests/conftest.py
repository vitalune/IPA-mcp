"""
Pytest configuration and shared fixtures for the Personal Knowledge Assistant MCP Server

This module provides:
- Shared test fixtures for all test modules
- Mock objects for external services
- Test data generation utilities
- Environment setup and cleanup
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import structlog

# Import project modules
from src.config.settings import Settings, Environment, get_settings
from src.config.auth import AuthProvider, get_auth_manager
from src.integrations.client_manager import APIClientManager
from src.utils.cache import get_cache_manager
from src.utils.encryption import get_encryption_manager
from src.models.data_models import TimeRange, EmailMessage, SocialMediaPost

# Configure test logging
structlog.configure(
    processors=[
        structlog.testing.LogCapture(),
    ],
    wrapper_class=structlog.testing.TestingLoggerFactory(),
    logger_factory=structlog.testing.TestingLoggerFactory(),
    cache_logger_on_first_use=False,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def test_settings(temp_dir):
    """Create test settings with temporary directories."""
    settings = Settings(
        environment=Environment.DEVELOPMENT,
        debug=True,
        encryption__master_key="test_key_32_bytes_for_testing_only",
        database__data_directory=temp_dir / "data",
        database__cache_directory=temp_dir / "cache",
        database__logs_directory=temp_dir / "logs",
        privacy__anonymize_logs=False,  # Easier testing
        audit__audit_enabled=True,
    )
    
    # Ensure directories exist
    settings.database.data_directory.mkdir(parents=True, exist_ok=True)
    settings.database.cache_directory.mkdir(parents=True, exist_ok=True)
    settings.database.logs_directory.mkdir(parents=True, exist_ok=True)
    
    return settings


@pytest.fixture(scope="function")
async def mock_auth_manager():
    """Create a mock authentication manager."""
    auth_manager = AsyncMock()
    auth_manager.get_user_credentials.return_value = {
        'access_token': 'test_access_token',
        'refresh_token': 'test_refresh_token',
        'expires_at': datetime.now() + timedelta(hours=1)
    }
    auth_manager.validate_token.return_value = True
    auth_manager.refresh_token.return_value = True
    auth_manager.list_tokens.return_value = [
        {'status': 'valid', 'provider': 'gmail', 'expires_at': datetime.now() + timedelta(hours=1)}
    ]
    return auth_manager


@pytest.fixture(scope="function")
async def mock_client_manager():
    """Create a mock API client manager."""
    client_manager = AsyncMock()
    
    # Mock clients
    mock_gmail_client = AsyncMock()
    mock_gmail_client.search_messages.return_value = []
    mock_gmail_client.send_message.return_value = {'id': 'test_message_id'}
    mock_gmail_client.get_health_status.return_value = {'status': 'healthy'}
    
    mock_drive_client = AsyncMock()
    mock_drive_client.search_files.return_value = []
    mock_drive_client.get_health_status.return_value = {'status': 'healthy'}
    
    mock_twitter_client = AsyncMock()
    mock_twitter_client.search_tweets.return_value = []
    mock_twitter_client.post_tweet.return_value = {'id': 'test_tweet_id'}
    mock_twitter_client.get_health_status.return_value = {'status': 'healthy'}
    
    mock_linkedin_client = AsyncMock()
    mock_linkedin_client.search_posts.return_value = []
    mock_linkedin_client.create_post.return_value = {'id': 'test_post_id'}
    mock_linkedin_client.get_health_status.return_value = {'status': 'healthy'}
    
    # Configure client manager methods
    client_manager.get_client.side_effect = lambda service: {
        'gmail': mock_gmail_client,
        'drive': mock_drive_client,
        'twitter': mock_twitter_client,
        'linkedin': mock_linkedin_client
    }.get(service)
    
    client_manager.check_authentication.return_value = True
    client_manager.get_overall_health.return_value = {
        'overall_health': 'healthy',
        'services': {'total': 4, 'healthy': 4, 'authenticated': 4}
    }
    
    return client_manager


@pytest.fixture(scope="function")
def mock_cache_manager():
    """Create a mock cache manager."""
    cache_manager = MagicMock()
    cache_manager.get.return_value = None
    cache_manager.set.return_value = True
    cache_manager.delete.return_value = True
    cache_manager.search_cache.return_value = []
    return cache_manager


@pytest.fixture(scope="function")
def mock_encryption_manager():
    """Create a mock encryption manager."""
    encryption_manager = MagicMock()
    encryption_manager.encrypt_data.side_effect = lambda data: f"encrypted_{data}"
    encryption_manager.decrypt_data.side_effect = lambda data: data.replace("encrypted_", "")
    encryption_manager.hash_data.side_effect = lambda data: f"hash_{hash(data)}"
    return encryption_manager


@pytest.fixture(scope="function")
def sample_emails():
    """Generate sample email data for testing."""
    return [
        {
            'id': 'email_1',
            'subject': 'Project Alpha Update',
            'body': 'Please review the attached proposal and let me know your thoughts by Friday. This is urgent.',
            'snippet': 'Please review the attached proposal...',
            'from': 'john.doe@example.com',
            'from_name': 'John Doe',
            'to': ['user@example.com'],
            'cc': [],
            'bcc': [],
            'date': datetime.now() - timedelta(days=1),
            'direction': 'received',
            'thread_id': 'thread_1',
            'labels': ['INBOX', 'IMPORTANT'],
            'attachments': ['proposal.pdf']
        },
        {
            'id': 'email_2',
            'subject': 'Re: Project Alpha Update',
            'body': 'Thanks for the update. I\'ll review the proposal and get back to you tomorrow.',
            'snippet': 'Thanks for the update...',
            'from': 'user@example.com',
            'from_name': 'Test User',
            'to': ['john.doe@example.com'],
            'cc': [],
            'bcc': [],
            'date': datetime.now() - timedelta(hours=6),
            'direction': 'sent',
            'thread_id': 'thread_1',
            'labels': ['SENT'],
            'attachments': []
        },
        {
            'id': 'email_3',
            'subject': 'Meeting follow-up - Action Items',
            'body': 'Action items from our meeting:\n1. Complete user research by next week\n2. Schedule next review meeting\n3. Update project timeline',
            'snippet': 'Action items from our meeting...',
            'from': 'user@example.com',
            'from_name': 'Test User',
            'to': ['team@example.com'],
            'cc': ['manager@example.com'],
            'bcc': [],
            'date': datetime.now() - timedelta(hours=2),
            'direction': 'sent',
            'thread_id': 'thread_2',
            'labels': ['SENT'],
            'attachments': []
        }
    ]


@pytest.fixture(scope="function")
def sample_social_posts():
    """Generate sample social media post data for testing."""
    return [
        {
            'id': 'tweet_1',
            'text': 'Excited to share our latest product update! 🚀 #innovation #tech #startup',
            'author': 'testuser',
            'author_id': 'user123',
            'created_at': datetime.now() - timedelta(hours=2),
            'public_metrics': {
                'like_count': 45,
                'retweet_count': 12,
                'reply_count': 8,
                'impression_count': 1500
            },
            'platform': 'twitter',
            'url': 'https://twitter.com/testuser/status/123456789'
        },
        {
            'id': 'linkedin_post_1',
            'text': 'Reflecting on the key insights from this week\'s industry conference. The future of AI in productivity tools looks incredibly promising.',
            'author': 'Test User',
            'author_id': 'testuser',
            'created_at': datetime.now() - timedelta(days=1),
            'likes': 23,
            'comments': 5,
            'shares': 3,
            'platform': 'linkedin',
            'url': 'https://linkedin.com/posts/testuser_123456789'
        }
    ]


@pytest.fixture(scope="function")
def sample_drive_files():
    """Generate sample Google Drive file data for testing."""
    return [
        {
            'id': 'file_1',
            'name': 'Project Alpha Proposal.pdf',
            'description': 'Detailed proposal for Project Alpha initiative',
            'mimeType': 'application/pdf',
            'size': 2048576,  # 2MB
            'createdTime': datetime.now() - timedelta(days=3),
            'modifiedTime': datetime.now() - timedelta(days=1),
            'owners': [{'displayName': 'John Doe', 'emailAddress': 'john.doe@example.com'}],
            'shared': True,
            'webViewLink': 'https://drive.google.com/file/d/file_1/view',
            'webContentLink': 'https://drive.google.com/file/d/file_1/download'
        },
        {
            'id': 'file_2',
            'name': 'Meeting Notes - Weekly Review',
            'description': 'Notes from weekly team review meeting',
            'mimeType': 'application/vnd.google-apps.document',
            'size': 45632,  # 45KB
            'createdTime': datetime.now() - timedelta(days=1),
            'modifiedTime': datetime.now() - timedelta(hours=3),
            'owners': [{'displayName': 'Test User', 'emailAddress': 'user@example.com'}],
            'shared': False,
            'webViewLink': 'https://docs.google.com/document/d/file_2/edit',
        }
    ]


@pytest.fixture(scope="function")
def sample_time_range():
    """Generate a sample time range for testing."""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    return TimeRange(start=start_time, end=end_time)


@pytest.fixture(scope="function")
def sample_project_data():
    """Generate sample project data for testing."""
    return {
        'id': 'project_alpha',
        'name': 'Project Alpha',
        'description': 'Revolutionary new product development initiative',
        'status': 'active',
        'priority': 'high',
        'deadline': datetime.now() + timedelta(days=30),
        'created_at': datetime.now() - timedelta(days=60),
        'updated_at': datetime.now() - timedelta(days=1),
        'tags': ['product', 'development', 'innovation'],
        'team_members': ['john.doe@example.com', 'jane.smith@example.com'],
        'progress': 0.65,
        'milestones': [
            {
                'name': 'Research Phase',
                'completed': True,
                'completed_at': datetime.now() - timedelta(days=45)
            },
            {
                'name': 'Prototype Development',
                'completed': True,
                'completed_at': datetime.now() - timedelta(days=20)
            },
            {
                'name': 'Testing Phase',
                'completed': False,
                'due_date': datetime.now() + timedelta(days=10)
            }
        ]
    }


@pytest.fixture(scope="function")
def mock_nlp_processor():
    """Create a mock NLP processor."""
    processor = AsyncMock()
    
    # Mock text analysis results
    processor.analyze_text.return_value = MagicMock(
        word_count=25,
        sentence_count=3,
        sentiment_score=0.2,
        sentiment_label='positive',
        urgency_level='medium',
        entities=[
            {'text': 'Project Alpha', 'label': 'PROJECT', 'confidence': 0.9},
            {'text': 'Friday', 'label': 'DATE', 'confidence': 0.8}
        ],
        keywords=[('project', 0.8), ('proposal', 0.7), ('review', 0.6)]
    )
    
    processor._analyze_sentiment.return_value = (0.2, 'positive')
    processor._extract_keywords.return_value = [('project', 0.8), ('proposal', 0.7)]
    processor._extract_entities.return_value = [
        {'text': 'Project Alpha', 'label': 'PROJECT', 'confidence': 0.9}
    ]
    
    return processor


@pytest.fixture(scope="function")
def mock_analytics_engine():
    """Create a mock analytics engine."""
    engine = MagicMock()
    
    engine.analyze_correlations.return_value = MagicMock(
        correlations={'metric1': {'metric2': 0.85}},
        significant_correlations=[('metric1', 'metric2', 0.85)],
        analysis_date=datetime.now()
    )
    
    engine.detect_trends.return_value = [
        {'metric': 'email_volume', 'trend': 'increasing', 'confidence': 0.8},
        {'metric': 'response_time', 'trend': 'stable', 'confidence': 0.6}
    ]
    
    return engine


@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external service calls to prevent actual API calls during testing."""
    with patch('httpx.AsyncClient') as mock_client:
        # Mock HTTP responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'success'}
        mock_response.text = '{"status": "success"}'
        
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.put.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.delete.return_value = mock_response
        
        yield mock_client


@pytest.fixture(scope="function")
def caplog_structlog(caplog):
    """Capture structlog messages in pytest caplog."""
    import logging
    logging.getLogger().addHandler(caplog.handler)
    return caplog


# Utility functions for tests

def create_mock_mcp_client():
    """Create a mock MCP client for testing MCP protocol compliance."""
    client = MagicMock()
    client.initialize.return_value = None
    client.list_tools.return_value = []
    client.call_tool.return_value = []
    client.list_resources.return_value = []
    return client


def assert_valid_email_structure(email_data: Dict[str, Any]):
    """Assert that email data has the expected structure."""
    required_fields = ['id', 'subject', 'from', 'to', 'date', 'direction']
    for field in required_fields:
        assert field in email_data, f"Missing required field: {field}"
    
    assert isinstance(email_data['to'], list), "Email 'to' field must be a list"
    assert isinstance(email_data['date'], datetime), "Email 'date' field must be a datetime"
    assert email_data['direction'] in ['sent', 'received'], "Invalid email direction"


def assert_valid_social_post_structure(post_data: Dict[str, Any]):
    """Assert that social media post data has the expected structure."""
    required_fields = ['id', 'text', 'author', 'created_at', 'platform']
    for field in required_fields:
        assert field in post_data, f"Missing required field: {field}"
    
    assert isinstance(post_data['created_at'], datetime), "Post 'created_at' field must be a datetime"
    assert post_data['platform'] in ['twitter', 'linkedin', 'facebook', 'instagram'], "Invalid platform"


def assert_valid_search_result_structure(result: Dict[str, Any]):
    """Assert that search result has the expected structure."""
    required_fields = ['id', 'type', 'title', 'snippet', 'relevance_score', 'timestamp', 'source']
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    assert isinstance(result['relevance_score'], (int, float)), "Relevance score must be numeric"
    assert isinstance(result['timestamp'], datetime), "Timestamp must be datetime"
    assert 0 <= result['relevance_score'] <= 1, "Relevance score must be between 0 and 1"


# Async test utilities

async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
    """Wait for a condition to become true with timeout."""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        await asyncio.sleep(interval)
    return False


# Performance testing utilities

class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
    
    def assert_duration_less_than(self, max_seconds: float):
        """Assert that the measured duration is less than the specified maximum."""
        assert self.duration is not None, "Timer was not used in context manager"
        assert self.duration < max_seconds, f"Execution took {self.duration}s, expected < {max_seconds}s"


# Security testing utilities

def generate_test_credentials():
    """Generate test credentials for authentication testing."""
    return {
        'client_id': 'test_client_id_123',
        'client_secret': 'test_client_secret_456',
        'access_token': 'test_access_token_789',
        'refresh_token': 'test_refresh_token_abc',
        'expires_at': datetime.now() + timedelta(hours=1)
    }


def assert_data_is_encrypted(data: str):
    """Assert that data appears to be encrypted."""
    # Simple check - encrypted data should not contain common readable patterns
    readable_patterns = ['email', 'password', 'token', 'secret', 'user']
    data_lower = data.lower()
    for pattern in readable_patterns:
        assert pattern not in data_lower, f"Data appears to contain readable pattern: {pattern}"


def assert_secure_headers_present(headers: Dict[str, str]):
    """Assert that security headers are present."""
    security_headers = [
        'X-Content-Type-Options',
        'X-Frame-Options', 
        'X-XSS-Protection',
        'Strict-Transport-Security'
    ]
    for header in security_headers:
        assert header in headers, f"Missing security header: {header}"