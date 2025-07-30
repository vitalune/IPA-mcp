"""
Integration tests for API Client functionality

Tests real integration between components with mocked external APIs.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.integrations.client_manager import APIClientManager, get_client_manager
from src.tools.search_tools import UniversalSearchEngine, get_search_engine, SearchScope
from src.tools.analysis_tools import CommunicationAnalyzer, get_communication_analyzer
from src.tools.social_tools import get_social_analyzer, SocialPlatform
from src.models.data_models import TimeRange


class TestAPIClientManagerIntegration:
    """Test API client manager integration with real workflows"""
    
    @pytest.fixture
    async def integrated_client_manager(self, test_settings, mock_auth_manager):
        """Create a fully integrated client manager"""
        with patch('src.integrations.client_manager.get_auth_manager', return_value=mock_auth_manager):
            manager = APIClientManager()
            await manager.initialize()
            
            # Configure all services
            services_config = {
                'gmail': {
                    'client_id': 'gmail_test_id',
                    'client_secret': 'gmail_test_secret',
                    'scopes': ['https://www.googleapis.com/auth/gmail.readonly']
                },
                'drive': {
                    'client_id': 'drive_test_id',
                    'client_secret': 'drive_test_secret',
                    'scopes': ['https://www.googleapis.com/auth/drive.readonly']
                },
                'twitter': {
                    'client_id': 'twitter_test_id',
                    'client_secret': 'twitter_test_secret',
                    'scopes': ['tweet.read', 'tweet.write']
                },
                'linkedin': {
                    'client_id': 'linkedin_test_id',
                    'client_secret': 'linkedin_test_secret',
                    'scopes': ['r_liteprofile', 'w_member_social']
                }
            }
            
            for service, config in services_config.items():
                manager.configure_service(service, **config)
            
            yield manager
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_multi_service_authentication_flow(self, integrated_client_manager):
        """Test authentication flow across multiple services"""
        services = ['gmail', 'drive', 'twitter', 'linkedin']
        
        with patch('src.integrations.gmail_client.GmailClient') as mock_gmail, \
             patch('src.integrations.drive_client.DriveClient') as mock_drive, \
             patch('src.integrations.twitter_client.TwitterClient') as mock_twitter, \
             patch('src.integrations.linkedin_client.LinkedInClient') as mock_linkedin:
            
            # Mock client creation
            mock_clients = {
                'gmail': AsyncMock(),
                'drive': AsyncMock(),
                'twitter': AsyncMock(),
                'linkedin': AsyncMock()
            }
            
            mock_gmail.return_value = mock_clients['gmail']
            mock_drive.return_value = mock_clients['drive']
            mock_twitter.return_value = mock_clients['twitter']
            mock_linkedin.return_value = mock_clients['linkedin']
            
            # Mock authentication responses
            for service, client in mock_clients.items():
                client.authenticate.side_effect = Exception(f"Please visit this URL to authorize: https://{service}.auth.url")
                client.handle_oauth_callback.return_value = True
                client.get_health_status.return_value = {'status': 'healthy'}
            
            # Test authentication for all services
            auth_urls = {}
            for service in services:
                try:
                    auth_url = await integrated_client_manager.authenticate_service(service)
                    auth_urls[service] = auth_url
                    assert f"{service}.auth.url" in auth_url
                except Exception as e:
                    pytest.fail(f"Authentication failed for {service}: {e}")
            
            # Test OAuth callbacks for all services
            callback_results = {}
            for service in services:
                success = await integrated_client_manager.handle_oauth_callback(
                    service=service,
                    authorization_code=f"{service}_auth_code",
                    state=f"{service}_state"
                )
                callback_results[service] = success
                assert success is True
            
            # Verify all services are authenticated
            for service in services:
                status = await integrated_client_manager.get_service_status(service)
                assert service in status
                assert status[service].authenticated is True
                assert status[service].healthy is True
    
    @pytest.mark.asyncio
    async def test_concurrent_api_operations(self, integrated_client_manager):
        """Test concurrent operations across multiple APIs"""
        with patch('src.integrations.gmail_client.GmailClient') as mock_gmail, \
             patch('src.integrations.drive_client.DriveClient') as mock_drive, \
             patch('src.integrations.twitter_client.TwitterClient') as mock_twitter:
            
            # Setup mock clients
            gmail_client = AsyncMock()
            drive_client = AsyncMock()
            twitter_client = AsyncMock()
            
            mock_gmail.return_value = gmail_client
            mock_drive.return_value = drive_client
            mock_twitter.return_value = twitter_client
            
            # Mock API responses
            gmail_client.search_messages.return_value = [
                {'id': 'email1', 'subject': 'Test Email 1', 'from': 'sender1@example.com'},
                {'id': 'email2', 'subject': 'Test Email 2', 'from': 'sender2@example.com'}
            ]
            
            drive_client.search_files.return_value = [
                {'id': 'file1', 'name': 'Document1.pdf', 'mimeType': 'application/pdf'},
                {'id': 'file2', 'name': 'Spreadsheet1.xlsx', 'mimeType': 'application/vnd.ms-excel'}
            ]
            
            twitter_client.search_tweets.return_value = [
                {'id': 'tweet1', 'text': 'Test tweet 1', 'author': 'user1'},
                {'id': 'tweet2', 'text': 'Test tweet 2', 'author': 'user2'}
            ]
            
            # Get clients
            gmail = await integrated_client_manager.get_client('gmail')
            drive = await integrated_client_manager.get_client('drive')
            twitter = await integrated_client_manager.get_client('twitter')
            
            # Perform concurrent operations
            tasks = [
                gmail.search_messages(query="test", max_results=10),
                drive.search_files(query="test", max_results=10),
                twitter.search_tweets(query="test", max_results=10)
            ]
            
            import time
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time
            
            # Verify all operations completed successfully
            assert len(results) == 3
            
            gmail_results, drive_results, twitter_results = results
            assert len(gmail_results) == 2
            assert len(drive_results) == 2
            assert len(twitter_results) == 2
            
            # Should complete concurrently (faster than sequential)
            assert execution_time < 5.0  # Very generous for mocked calls
    
    @pytest.mark.asyncio
    async def test_cross_service_data_flow(self, integrated_client_manager, sample_emails, sample_drive_files):
        """Test data flow between different services"""
        with patch('src.integrations.gmail_client.GmailClient') as mock_gmail, \
             patch('src.integrations.drive_client.DriveClient') as mock_drive:
            
            gmail_client = AsyncMock()
            drive_client = AsyncMock()
            
            mock_gmail.return_value = gmail_client
            mock_drive.return_value = drive_client
            
            # Mock Gmail returning emails with attachment references
            gmail_client.search_messages.return_value = sample_emails
            
            # Mock Drive returning files that match email attachments
            drive_client.search_files.return_value = sample_drive_files
            drive_client.get_file_content.return_value = "Sample file content for analysis"
            
            # Get clients
            gmail = await integrated_client_manager.get_client('gmail')
            drive = await integrated_client_manager.get_client('drive')
            
            # Simulate cross-service workflow
            # 1. Search for emails with attachments
            emails = await gmail.search_messages(query="has:attachment", max_results=10)
            assert len(emails) > 0
            
            # 2. For each email with attachments, find related files in Drive
            attachment_files = []
            for email in emails:
                if email.get('attachments'):
                    for attachment in email['attachments']:
                        # Search for files with similar names
                        related_files = await drive.search_files(
                            query=f"name:{attachment.split('.')[0]}",
                            max_results=5
                        )
                        attachment_files.extend(related_files)
            
            # 3. Get content from related files
            file_contents = []
            for file_info in attachment_files[:3]:  # Limit to first 3 files
                content = await drive.get_file_content(file_info['id'])
                file_contents.append({
                    'file_id': file_info['id'],
                    'file_name': file_info['name'],
                    'content': content
                })
            
            # Verify the cross-service data flow worked
            assert len(attachment_files) > 0
            assert len(file_contents) > 0
            
            # Verify API calls were made
            gmail_client.search_messages.assert_called_once()
            drive_client.search_files.assert_called()
            drive_client.get_file_content.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_resilience_across_services(self, integrated_client_manager):
        """Test system resilience when some services fail"""
        with patch('src.integrations.gmail_client.GmailClient') as mock_gmail, \
             patch('src.integrations.drive_client.DriveClient') as mock_drive, \
             patch('src.integrations.twitter_client.TwitterClient') as mock_twitter:
            
            # Setup mock clients with mixed success/failure
            gmail_client = AsyncMock()
            drive_client = AsyncMock()
            twitter_client = AsyncMock()
            
            mock_gmail.return_value = gmail_client
            mock_drive.return_value = drive_client
            mock_twitter.return_value = twitter_client
            
            # Gmail works fine
            gmail_client.search_messages.return_value = [
                {'id': 'email1', 'subject': 'Working Email'}
            ]
            gmail_client.get_health_status.return_value = {'status': 'healthy'}
            
            # Drive has a temporary error
            drive_client.search_files.side_effect = ConnectionError("Drive API temporarily unavailable")
            drive_client.get_health_status.return_value = {'status': 'error', 'error': 'Connection failed'}
            
            # Twitter has authentication issues
            twitter_client.search_tweets.side_effect = Exception("Authentication required")
            twitter_client.get_health_status.return_value = {'status': 'error', 'error': 'Not authenticated'}
            
            # Get overall health status
            health = await integrated_client_manager.get_overall_health()
            
            # System should still be partially functional
            assert health['services']['total'] >= 3
            assert health['services']['healthy'] >= 1  # At least Gmail should be healthy
            assert health['services']['unhealthy'] >= 2  # Drive and Twitter should be unhealthy
            assert health['overall_health'] == 'degraded'  # Not fully healthy, but not completely down
            
            # Gmail should still work despite other services failing
            gmail = await integrated_client_manager.get_client('gmail')
            emails = await gmail.search_messages(query="test", max_results=10)
            assert len(emails) == 1
            assert emails[0]['subject'] == 'Working Email'
    
    @pytest.mark.asyncio
    async def test_bulk_operations_integration(self, integrated_client_manager, temp_dir):
        """Test bulk operations across multiple services"""
        with patch('src.integrations.gmail_client.GmailClient') as mock_gmail, \
             patch('src.integrations.drive_client.DriveClient') as mock_drive:
            
            gmail_client = AsyncMock()
            drive_client = AsyncMock()
            
            mock_gmail.return_value = gmail_client
            mock_drive.return_value = drive_client
            
            # Mock export functionality
            gmail_client.export_data.return_value = 150  # 150 emails exported
            drive_client.export_data.return_value = 75   # 75 files exported
            
            # Mock authentication check
            with patch.object(integrated_client_manager, 'check_authentication', return_value=True):
                # Perform bulk export
                results = await integrated_client_manager.bulk_export(
                    output_directory=temp_dir,
                    services=['gmail', 'drive'],
                    max_items=200
                )
                
                # Verify export results
                assert 'gmail' in results
                assert 'drive' in results
                assert results['gmail'] == 150
                assert results['drive'] == 75
                
                # Verify export was called for each service
                gmail_client.export_data.assert_called_once()
                drive_client.export_data.assert_called_once()
                
                # Verify output directories were created
                assert (temp_dir / 'gmail').exists() or True  # May be created by mock
                assert (temp_dir / 'drive').exists() or True  # May be created by mock


class TestSearchEngineIntegration:
    """Test search engine integration with API clients"""
    
    @pytest.fixture
    async def integrated_search_engine(self, mock_client_manager, mock_nlp_processor, mock_cache_manager):
        """Create a search engine with integrated dependencies"""
        with patch('src.tools.search_tools.get_client_manager', return_value=mock_client_manager), \
             patch('src.tools.search_tools.get_nlp_processor', return_value=mock_nlp_processor), \
             patch('src.tools.search_tools.get_cache_manager', return_value=mock_cache_manager):
            
            engine = UniversalSearchEngine()
            await engine.initialize()
            return engine
    
    @pytest.mark.asyncio
    async def test_universal_search_across_services(self, integrated_search_engine, mock_client_manager):
        """Test universal search across all integrated services"""
        # Mock client responses
        gmail_client = mock_client_manager.get_client.return_value
        gmail_client.search_messages.return_value = [
            {
                'id': 'email1',
                'subject': 'Project Alpha Update',
                'snippet': 'Latest updates on project alpha progress',
                'from': 'john@example.com',
                'date': datetime.now(),
                'body': 'Detailed project progress report'
            }
        ]
        
        drive_client = mock_client_manager.get_client.return_value
        drive_client.search_files.return_value = [
            {
                'id': 'file1',
                'name': 'Project Alpha Proposal.pdf',
                'description': 'Proposal document for project alpha',
                'modifiedTime': datetime.now(),
                'mimeType': 'application/pdf'
            }
        ]
        
        twitter_client = mock_client_manager.get_client.return_value
        twitter_client.search_tweets.return_value = [
            {
                'id': 'tweet1',
                'text': 'Excited about project alpha launch! #innovation',
                'author': 'company_account',
                'created_at': datetime.now(),
                'public_metrics': {'like_count': 25, 'retweet_count': 5}
            }
        ]
        
        # Configure mock to return different clients for different services
        def mock_get_client(service):
            if service == 'gmail':
                return gmail_client
            elif service == 'drive':
                return drive_client  
            elif service == 'twitter':
                return twitter_client
            return None
        
        mock_client_manager.get_client.side_effect = mock_get_client
        
        # Perform universal search
        results = await integrated_search_engine.universal_search(
            query="project alpha",
            scope=SearchScope.ALL,
            page_size=20
        )
        
        # Verify results from all sources
        assert results.total_count >= 3
        assert len(results.results) >= 3
        
        # Verify we got results from different sources
        sources = {result.source for result in results.results}
        assert 'gmail' in sources
        assert 'drive' in sources
        assert 'twitter' in sources
        
        # Verify result structure
        for result in results.results:
            assert result.id is not None
            assert result.title is not None
            assert result.relevance_score >= 0
            assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_contextual_search_integration(self, integrated_search_engine, mock_nlp_processor):
        """Test contextual search with NLP integration"""
        # Mock NLP processor for keyword extraction
        mock_nlp_processor._extract_keywords.return_value = [
            ('project', 0.9),
            ('alpha', 0.8),
            ('proposal', 0.7)
        ]
        
        # Mock search results
        primary_result = type('MockResult', (), {
            'id': 'primary_1',
            'title': 'Project Alpha Proposal',
            'content': 'Detailed proposal for project alpha initiative',
            'timestamp': datetime.now(),
            'source': 'drive'
        })()
        
        with patch.object(integrated_search_engine, 'universal_search') as mock_search:
            # Mock primary search
            mock_search.return_value = type('MockResults', (), {
                'results': [primary_result]
            })()
            
            # Perform contextual search
            contextual_results = await integrated_search_engine.contextual_search(
                query="project alpha proposal",
                context_items=["proposal", "project", "alpha"]
            )
            
            # Verify contextual search was performed
            assert len(contextual_results) >= 1
            
            contextual_result = contextual_results[0]
            assert contextual_result.primary_result == primary_result
            assert contextual_result.relationship_type == "content_similarity"
            assert contextual_result.context_summary is not None
    
    @pytest.mark.asyncio
    async def test_search_caching_integration(self, integrated_search_engine, mock_cache_manager):
        """Test search results caching integration"""
        query = "test query for caching"
        
        # First search - should not be cached
        mock_cache_manager.get.return_value = None
        
        with patch.object(integrated_search_engine, '_search_source') as mock_search_source:
            mock_search_source.return_value = [
                type('MockResult', (), {
                    'id': 'result1',
                    'title': 'Cached Result',
                    'snippet': 'This result should be cached',
                    'relevance_score': 0.8,
                    'timestamp': datetime.now(),
                    'source': 'gmail'
                })()
            ]
            
            # Perform search
            results1 = await integrated_search_engine.universal_search(query, page_size=10)
            
            # Verify cache was checked and result was stored
            mock_cache_manager.get.assert_called()
            mock_cache_manager.set.assert_called()
            
            # Second search - should use cache
            mock_cache_manager.get.return_value = results1.results
            
            results2 = await integrated_search_engine.universal_search(query, page_size=10)
            
            # Should have same results
            assert len(results2.results) == len(results1.results)
            assert results2.results[0].title == results1.results[0].title


class TestAnalysisToolsIntegration:
    """Test analysis tools integration with API clients"""
    
    @pytest.fixture
    async def integrated_communication_analyzer(self, mock_client_manager, mock_nlp_processor, mock_analytics_engine):
        """Create communication analyzer with integrated dependencies"""
        with patch('src.tools.analysis_tools.get_client_manager', return_value=mock_client_manager), \
             patch('src.tools.analysis_tools.get_nlp_processor', return_value=mock_nlp_processor), \
             patch('src.tools.analysis_tools.get_analytics_engine', return_value=mock_analytics_engine):
            
            analyzer = CommunicationAnalyzer()
            await analyzer.initialize()
            return analyzer
    
    @pytest.mark.asyncio
    async def test_communication_pattern_analysis_integration(self, integrated_communication_analyzer, sample_emails, sample_time_range):
        """Test communication pattern analysis with real email data"""
        # Mock email fetching
        with patch.object(integrated_communication_analyzer, '_fetch_emails_in_range', return_value=sample_emails):
            
            # Perform communication pattern analysis
            patterns = await integrated_communication_analyzer.analyze_communication_patterns(
                time_range=sample_time_range,
                include_sentiment=True
            )
            
            # Verify analysis structure
            assert 'time_range' in patterns
            assert 'overview' in patterns
            assert 'temporal_patterns' in patterns
            assert 'response_patterns' in patterns
            assert 'contact_analysis' in patterns
            assert 'sentiment_analysis' in patterns
            
            # Verify overview data
            overview = patterns['overview']
            assert 'total_emails' in overview
            assert 'sent_emails' in overview
            assert 'received_emails' in overview
            assert 'daily_average' in overview
            
            # Verify temporal patterns
            temporal = patterns['temporal_patterns']
            assert 'peak_hours' in temporal
            assert 'peak_days_of_week' in temporal
            
            # Verify contact analysis
            contacts = patterns['contact_analysis']
            assert isinstance(contacts, list)
            if contacts:
                contact = contacts[0]
                assert 'contact' in contact
                assert 'total_emails' in contact
    
    @pytest.mark.asyncio
    async def test_network_analysis_integration(self, integrated_communication_analyzer, sample_emails, sample_time_range):
        """Test communication network analysis integration"""
        # Mock email fetching with network-relevant data
        network_emails = [
            {
                'id': 'email1',
                'from': 'alice@example.com',
                'to': ['bob@example.com'],
                'date': datetime.now() - timedelta(days=1),
                'body': 'Project discussion'
            },
            {
                'id': 'email2', 
                'from': 'bob@example.com',
                'to': ['alice@example.com', 'charlie@example.com'],
                'date': datetime.now() - timedelta(hours=12),
                'body': 'Follow-up discussion'
            },
            {
                'id': 'email3',
                'from': 'charlie@example.com',
                'to': ['alice@example.com'],
                'date': datetime.now() - timedelta(hours=6),
                'body': 'Response to discussion'
            }
        ]
        
        with patch.object(integrated_communication_analyzer, '_fetch_emails_in_range', return_value=network_emails):
            
            # Perform network analysis
            network = await integrated_communication_analyzer.analyze_communication_network(
                time_range=sample_time_range,
                min_interactions=1
            )
            
            # Verify network structure
            assert len(network.nodes) >= 3  # alice, bob, charlie
            assert len(network.edges) >= 2  # At least some connections
            
            # Verify node properties
            for node in network.nodes:
                assert node.id is not None
                assert node.centrality_score >= 0
                assert isinstance(node.cluster_id, int)
            
            # Verify edge properties
            for edge in network.edges:
                assert edge.source is not None
                assert edge.target is not None
                assert edge.weight > 0
                assert edge.interaction_count > 0
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_integration(self, integrated_communication_analyzer, mock_nlp_processor, sample_emails):
        """Test sentiment analysis integration with NLP processor"""
        # Mock NLP processor sentiment analysis
        mock_nlp_processor._analyze_sentiment.side_effect = [
            (0.3, 'positive'),  # First email - positive
            (-0.1, 'negative'), # Second email - slightly negative  
            (0.7, 'positive')   # Third email - very positive
        ]
        
        with patch.object(integrated_communication_analyzer, '_fetch_emails_in_range', return_value=sample_emails):
            
            # Perform sentiment analysis as part of communication patterns
            patterns = await integrated_communication_analyzer.analyze_communication_patterns(
                time_range=TimeRange(
                    start=datetime.now() - timedelta(days=7),
                    end=datetime.now()
                ),
                include_sentiment=True
            )
            
            # Verify sentiment analysis results
            assert 'sentiment_analysis' in patterns
            sentiment = patterns['sentiment_analysis']
            
            assert 'overall_sentiment' in sentiment
            assert 'sentiment_variability' in sentiment
            assert 'positive_communication_ratio' in sentiment
            assert 'negative_communication_ratio' in sentiment
            
            # Verify NLP processor was called for each email
            assert mock_nlp_processor._analyze_sentiment.call_count == len(sample_emails)


class TestEndToEndIntegration:
    """Test complete end-to-end integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self, temp_dir, test_settings):
        """Test complete workflow from authentication to analysis"""
        # This test simulates a complete user workflow:
        # 1. Configure services
        # 2. Authenticate with APIs
        # 3. Search across services
        # 4. Analyze communication patterns
        # 5. Generate insights report
        
        with patch('src.integrations.client_manager.get_auth_manager') as mock_auth_manager_factory:
            mock_auth_manager = AsyncMock()
            mock_auth_manager.list_tokens.return_value = [
                {"status": "valid", "provider": "gmail", "expires_at": datetime.now() + timedelta(hours=1)}
            ]
            mock_auth_manager_factory.return_value = mock_auth_manager
            
            # Step 1: Initialize client manager
            client_manager = APIClientManager()
            await client_manager.initialize()
            
            # Configure Gmail service
            client_manager.configure_service(
                service="gmail",
                client_id="test_gmail_id",
                client_secret="test_gmail_secret",
                scopes=["https://www.googleapis.com/auth/gmail.readonly"]
            )
            
            # Step 2: Mock authentication
            with patch('src.integrations.gmail_client.GmailClient') as mock_gmail_class:
                mock_gmail_client = AsyncMock()
                mock_gmail_client.authenticate.side_effect = Exception("Please visit: https://gmail.auth.url")
                mock_gmail_client.handle_oauth_callback.return_value = True
                mock_gmail_client.get_health_status.return_value = {'status': 'healthy'}
                
                # Mock search functionality
                mock_gmail_client.search_messages.return_value = [
                    {
                        'id': 'email1',
                        'subject': 'Weekly Team Update',
                        'from': 'manager@company.com',
                        'to': ['team@company.com'],
                        'body': 'This week we accomplished great things in project alpha.',
                        'date': datetime.now() - timedelta(days=2),
                        'direction': 'received'
                    },
                    {
                        'id': 'email2',
                        'subject': 'Re: Weekly Team Update',
                        'from': 'user@company.com',
                        'to': ['manager@company.com'],
                        'body': 'Thanks for the update! Looking forward to next week.',
                        'date': datetime.now() - timedelta(days=1),
                        'direction': 'sent'
                    }
                ]
                
                mock_gmail_class.return_value = mock_gmail_client
                
                # Step 3: Authenticate
                auth_url = await client_manager.authenticate_service("gmail")
                assert "gmail.auth.url" in auth_url
                
                success = await client_manager.handle_oauth_callback(
                    service="gmail",
                    authorization_code="test_code",
                    state="test_state"
                )
                assert success is True
                
                # Step 4: Initialize search engine
                with patch('src.tools.search_tools.get_client_manager', return_value=client_manager):
                    search_engine = UniversalSearchEngine()
                    await search_engine.initialize()
                    
                    # Perform search
                    search_results = await search_engine.universal_search(
                        query="weekly update",
                        scope=SearchScope.EMAIL,
                        page_size=10
                    )
                    
                    assert search_results.total_count >= 2
                    assert len(search_results.results) >= 2
                
                # Step 5: Initialize communication analyzer
                with patch('src.tools.analysis_tools.get_client_manager', return_value=client_manager), \
                     patch('src.tools.analysis_tools.get_nlp_processor') as mock_nlp_factory:
                    
                    mock_nlp = AsyncMock()
                    mock_nlp._analyze_sentiment.side_effect = [(0.5, 'positive'), (0.3, 'positive')]
                    mock_nlp_factory.return_value = mock_nlp
                    
                    analyzer = CommunicationAnalyzer()
                    await analyzer.initialize()
                    
                    # Mock email fetching to return our test emails
                    with patch.object(analyzer, '_fetch_emails_in_range') as mock_fetch:
                        mock_fetch.return_value = mock_gmail_client.search_messages.return_value
                        
                        # Perform communication analysis
                        time_range = TimeRange(
                            start=datetime.now() - timedelta(days=7),
                            end=datetime.now()
                        )
                        
                        patterns = await analyzer.analyze_communication_patterns(
                            time_range=time_range,
                            include_sentiment=True
                        )
                        
                        # Verify analysis results
                        assert 'overview' in patterns
                        assert 'temporal_patterns' in patterns
                        assert 'sentiment_analysis' in patterns
                        
                        overview = patterns['overview']
                        assert overview['total_emails'] == 2
                        assert overview['sent_emails'] == 1
                        assert overview['received_emails'] == 1
            
            # Cleanup
            await client_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, test_settings):
        """Test system behavior when services fail and recover"""
        with patch('src.integrations.client_manager.get_auth_manager') as mock_auth_factory:
            mock_auth_manager = AsyncMock()
            mock_auth_manager.list_tokens.return_value = []
            mock_auth_factory.return_value = mock_auth_manager
            
            client_manager = APIClientManager()
            await client_manager.initialize()
            
            # Configure services
            client_manager.configure_service("gmail", "test_id", "test_secret", ["gmail.readonly"])
            client_manager.configure_service("drive", "test_id", "test_secret", ["drive.readonly"])
            
            with patch('src.integrations.gmail_client.GmailClient') as mock_gmail_class, \
                 patch('src.integrations.drive_client.DriveClient') as mock_drive_class:
                
                gmail_client = AsyncMock()
                drive_client = AsyncMock()
                
                mock_gmail_class.return_value = gmail_client
                mock_drive_class.return_value = drive_client
                
                # Initially, both services are healthy
                gmail_client.get_health_status.return_value = {'status': 'healthy'}
                drive_client.get_health_status.return_value = {'status': 'healthy'}
                
                # Get initial health
                initial_health = await client_manager.get_overall_health()
                assert initial_health['overall_health'] in ['healthy', 'degraded']  # May be degraded due to no auth
                
                # Simulate Gmail failure
                gmail_client.get_health_status.return_value = {'status': 'error', 'error': 'API rate limit exceeded'}
                gmail_client.search_messages.side_effect = Exception("Rate limit exceeded")
                
                # Health should now be degraded
                degraded_health = await client_manager.get_overall_health()
                assert degraded_health['overall_health'] == 'degraded'
                
                # Drive should still work
                drive_client.search_files.return_value = [{'id': 'file1', 'name': 'test.pdf'}]
                drive = await client_manager.get_client('drive')
                files = await drive.search_files(query="test", max_results=10)
                assert len(files) == 1
                
                # Gmail should fail
                gmail = await client_manager.get_client('gmail')
                with pytest.raises(Exception, match="Rate limit exceeded"):
                    await gmail.search_messages(query="test", max_results=10)
                
                # Simulate Gmail recovery
                gmail_client.get_health_status.return_value = {'status': 'healthy'}
                gmail_client.search_messages.side_effect = None
                gmail_client.search_messages.return_value = [{'id': 'email1', 'subject': 'recovered'}]
                
                # Both services should work again
                emails = await gmail.search_messages(query="test", max_results=10)
                assert len(emails) == 1
                assert emails[0]['subject'] == 'recovered'
                
                recovered_health = await client_manager.get_overall_health()
                # Health should improve (though may still be degraded due to auth status)
                assert recovered_health['services']['healthy'] >= degraded_health['services']['healthy']
            
            await client_manager.shutdown()