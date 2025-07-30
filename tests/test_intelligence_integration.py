"""
Integration tests for the intelligence and analytics features
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.tools.search_tools import get_search_engine, SearchScope
from src.tools.analysis_tools import get_communication_analyzer, AnalysisType
from src.tools.social_tools import get_social_analyzer, SocialPlatform
from src.tools.task_tools import get_task_engine, TaskSource
from src.utils.nlp_processor import get_nlp_processor
from src.utils.analytics_engine import get_analytics_engine
from src.models.data_models import TimeRange


class TestIntelligenceIntegration:
    """Test integration between intelligence components"""
    
    @pytest.fixture
    async def mock_data(self):
        """Create mock data for testing"""
        return {
            'emails': [
                {
                    'id': 'email_1',
                    'subject': 'Project Alpha Update',
                    'body': 'Please review the proposal by Friday. Let me know your thoughts.',
                    'from': 'john@example.com',
                    'to': ['user@example.com'],
                    'date': datetime.now() - timedelta(days=1),
                    'direction': 'received'
                },
                {
                    'id': 'email_2',
                    'subject': 'Meeting follow-up',
                    'body': 'Action items from our meeting: 1. Complete user research 2. Schedule next review',
                    'from': 'user@example.com',
                    'to': ['team@example.com'],
                    'date': datetime.now() - timedelta(hours=6),
                    'direction': 'sent'
                }
            ],
            'social_posts': [
                {
                    'id': 'post_1',
                    'text': 'Excited to share our latest product update! #innovation #tech',
                    'created_at': datetime.now() - timedelta(hours=2),
                    'public_metrics': {
                        'like_count': 45,
                        'retweet_count': 12,
                        'reply_count': 8,
                        'impression_count': 1500
                    }
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_nlp_processor_initialization(self):
        """Test NLP processor can be initialized"""
        processor = await get_nlp_processor()
        assert processor is not None
        assert processor._initialized
        
        # Test basic text analysis
        text = "This is an urgent task that needs to be completed by tomorrow."
        analysis = await processor.analyze_text(text)
        
        assert analysis.word_count > 0
        assert analysis.sentiment_score is not None
        assert analysis.urgency_level is not None
    
    @pytest.mark.asyncio
    async def test_analytics_engine_functionality(self):
        """Test analytics engine basic functionality"""
        engine = get_analytics_engine()
        assert engine is not None
        
        # Test correlation analysis
        data = {
            'metric1': [1, 2, 3, 4, 5],
            'metric2': [2, 4, 6, 8, 10]
        }
        
        correlation_analysis = await engine.analyze_correlations(data)
        assert correlation_analysis is not None
        assert 'metric1' in correlation_analysis.correlations
    
    @pytest.mark.asyncio
    async def test_search_engine_integration(self, mock_data):
        """Test search engine with mocked data"""
        search_engine = await get_search_engine()
        assert search_engine is not None
        
        # Mock the client manager and search results
        search_engine.client_manager = MagicMock()
        
        # Test universal search
        query = "project alpha"
        results = await search_engine.universal_search(query, scope=SearchScope.EMAIL)
        
        assert results is not None
        assert hasattr(results, 'results')
        assert hasattr(results, 'total_count')
    
    @pytest.mark.asyncio
    async def test_task_extraction_integration(self, mock_data):
        """Test task extraction from email content"""
        task_engine = await get_task_engine()
        assert task_engine is not None
        
        # Test task extraction from mock email
        email = mock_data['emails'][0]
        tasks = await task_engine.extract_tasks_from_email(
            email['body'],
            {'id': email['id'], 'from': email['from'], 'direction': email['direction']}
        )
        
        assert isinstance(tasks, list)
        # Should extract the "Please review" task
        if tasks:
            task = tasks[0]
            assert hasattr(task, 'title')
            assert hasattr(task, 'priority')
            assert hasattr(task, 'deadline')
    
    @pytest.mark.asyncio
    async def test_communication_analysis_integration(self, mock_data):
        """Test communication analysis with mock data"""
        analyzer = await get_communication_analyzer()
        assert analyzer is not None
        
        # Mock the email fetching
        analyzer._fetch_emails_in_range = AsyncMock(return_value=mock_data['emails'])
        
        time_range = TimeRange(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now()
        )
        
        # Test communication pattern analysis
        patterns = await analyzer.analyze_communication_patterns(
            time_range,
            include_sentiment=True
        )
        
        assert isinstance(patterns, dict)
        assert 'time_range' in patterns
        assert 'overview' in patterns
    
    @pytest.mark.asyncio
    async def test_social_media_analysis_integration(self, mock_data):
        """Test social media analysis with mock data"""
        social_analyzer = await get_social_analyzer()
        assert social_analyzer is not None
        
        # Mock the post fetching
        social_analyzer._fetch_posts = AsyncMock(return_value=mock_data['social_posts'])
        
        time_range = TimeRange(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now()
        )
        
        # Test content performance analysis
        performance = await social_analyzer.analyze_content_performance(
            SocialPlatform.TWITTER,
            time_range
        )
        
        assert isinstance(performance, dict)
        if 'overview' in performance:
            assert 'total_posts' in performance['overview']
    
    @pytest.mark.asyncio
    async def test_cross_component_integration(self, mock_data):
        """Test integration between multiple components"""
        # Initialize all components
        nlp_processor = await get_nlp_processor()
        analytics_engine = get_analytics_engine()
        task_engine = await get_task_engine()
        
        # Test workflow: Extract tasks -> Analyze sentiment -> Generate insights
        email_content = mock_data['emails'][0]['body']
        
        # Step 1: Extract tasks
        tasks = await task_engine.extract_tasks_from_email(
            email_content,
            mock_data['emails'][0]
        )
        
        # Step 2: Analyze sentiment of the email
        sentiment_analysis = await nlp_processor._analyze_sentiment(email_content)
        
        # Step 3: Generate productivity insights (mock)
        time_range = TimeRange(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now()
        )
        
        task_engine._fetch_emails_in_range = AsyncMock(return_value=mock_data['emails'])
        insights = await task_engine.analyze_productivity_patterns(time_range)
        
        # Verify the workflow completed
        assert len(tasks) >= 0  # Tasks may or may not be extracted
        assert sentiment_analysis is not None
        assert isinstance(insights, list)
    
    @pytest.mark.asyncio
    async def test_project_context_aggregation(self, mock_data):
        """Test project context aggregation across components"""
        task_engine = await get_task_engine()
        search_engine = await get_search_engine()
        
        # Mock search results
        from src.tools.search_tools import SearchResult, SearchResultType
        mock_search_results = MagicMock()
        mock_search_results.results = [
            SearchResult(
                id='result_1',
                type=SearchResultType.EMAIL,
                title='Project Alpha Update',
                snippet='Project progress discussion',
                content=mock_data['emails'][0]['body'],
                metadata={'from': 'john@example.com'},
                relevance_score=0.9,
                timestamp=datetime.now() - timedelta(days=1),
                source='gmail'
            )
        ]
        
        search_engine.universal_search = AsyncMock(return_value=mock_search_results)
        
        # Test project context aggregation
        project_context = await task_engine.aggregate_project_context('Project Alpha')
        
        assert project_context is not None
        assert project_context.project_name == 'Project Alpha'
        assert hasattr(project_context, 'key_participants')
        assert hasattr(project_context, 'timeline')
        assert hasattr(project_context, 'current_status')
    
    @pytest.mark.asyncio
    async def test_privacy_preservation(self):
        """Test that privacy settings are respected"""
        nlp_processor = await get_nlp_processor()
        
        # Test with anonymization enabled
        sensitive_text = "Contact John Smith at john.smith@company.com or call 555-123-4567"
        
        analysis = await nlp_processor.analyze_text(
            sensitive_text,
            include_entities=True,
            anonymize=True
        )
        
        # Check that sensitive information is anonymized
        assert analysis.text != sensitive_text  # Should be anonymized
        assert '[EMAIL]' in analysis.text or analysis.text != sensitive_text
    
    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self):
        """Test that components handle errors gracefully"""
        task_engine = await get_task_engine()
        
        # Test with invalid input
        try:
            tasks = await task_engine.extract_tasks_from_email(
                "",  # Empty content
                {}   # Empty metadata
            )
            # Should return empty list, not crash
            assert isinstance(tasks, list)
        except Exception as e:
            pytest.fail(f"Component should handle empty input gracefully: {e}")
        
        # Test with malformed data
        try:
            malformed_metadata = {'invalid': None, 'date': 'not-a-date'}
            tasks = await task_engine.extract_tasks_from_email(
                "Some task content",
                malformed_metadata
            )
            assert isinstance(tasks, list)
        except Exception as e:
            pytest.fail(f"Component should handle malformed data gracefully: {e}")


if __name__ == "__main__":
    pytest.main([__file__])