"""
Unit tests for API Integrations

Tests all API clients with comprehensive mocking to avoid external API calls.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.integrations.client_manager import APIClientManager, ClientConfig, ServiceStatus
from src.integrations.gmail_client import GmailClient
from src.integrations.drive_client import DriveClient
from src.integrations.twitter_client import TwitterClient
from src.integrations.linkedin_client import LinkedInClient
from src.config.auth import AuthProvider


class TestAPIClientManager:
    """Test the central API client manager"""
    
    @pytest.fixture
    async def client_manager(self, test_settings, mock_auth_manager):
        """Create a test client manager"""
        with patch('src.integrations.client_manager.get_auth_manager', return_value=mock_auth_manager):
            manager = APIClientManager()
            await manager.initialize()
            yield manager
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_client_manager_initialization(self, client_manager):
        """Test client manager initializes properly"""
        assert client_manager is not None
        assert client_manager._clients == {}
        assert client_manager._client_configs == {}
        assert client_manager._service_status == {}
    
    def test_configure_service(self, client_manager):
        """Test service configuration"""
        client_manager.configure_service(
            service="gmail",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        
        assert "gmail" in client_manager._client_configs
        config = client_manager._client_configs["gmail"]
        assert config.client_id == "test_client_id"
        assert config.client_secret == "test_client_secret"
        assert config.scopes == ["https://www.googleapis.com/auth/gmail.readonly"]
        assert config.enabled is True
    
    @pytest.mark.asyncio
    async def test_create_gmail_client(self, client_manager):
        """Test Gmail client creation"""
        client_manager.configure_service(
            service="gmail",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        
        with patch('src.integrations.client_manager.GmailClient') as mock_gmail_class:
            mock_client = AsyncMock()
            mock_gmail_class.return_value = mock_client
            
            client = await client_manager.create_client("gmail")
            
            assert client is not None
            assert client == mock_client
            assert "gmail" in client_manager._clients
            assert "gmail" in client_manager._service_status
    
    @pytest.mark.asyncio
    async def test_create_drive_client(self, client_manager):
        """Test Drive client creation"""
        client_manager.configure_service(
            service="drive",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        
        with patch('src.integrations.client_manager.DriveClient') as mock_drive_class:
            mock_client = AsyncMock()
            mock_drive_class.return_value = mock_client
            
            client = await client_manager.create_client("drive")
            
            assert client is not None
            assert client == mock_client
            assert "drive" in client_manager._clients
    
    @pytest.mark.asyncio
    async def test_create_twitter_client(self, client_manager):
        """Test Twitter client creation"""
        client_manager.configure_service(
            service="twitter",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["tweet.read", "tweet.write"]
        )
        
        with patch('src.integrations.client_manager.TwitterClient') as mock_twitter_class:
            mock_client = AsyncMock()
            mock_twitter_class.return_value = mock_client
            
            client = await client_manager.create_client("twitter")
            
            assert client is not None
            assert client == mock_client
            assert "twitter" in client_manager._clients
    
    @pytest.mark.asyncio
    async def test_create_linkedin_client(self, client_manager):
        """Test LinkedIn client creation"""
        client_manager.configure_service(
            service="linkedin",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["r_liteprofile", "w_member_social"]
        )
        
        with patch('src.integrations.client_manager.LinkedInClient') as mock_linkedin_class:
            mock_client = AsyncMock()
            mock_linkedin_class.return_value = mock_client
            
            client = await client_manager.create_client("linkedin")
            
            assert client is not None
            assert client == mock_client
            assert "linkedin" in client_manager._clients
    
    @pytest.mark.asyncio
    async def test_get_client_creates_if_not_exists(self, client_manager):
        """Test that get_client creates client if it doesn't exist"""
        client_manager.configure_service(
            service="gmail",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        
        with patch('src.integrations.client_manager.GmailClient') as mock_gmail_class:
            mock_client = AsyncMock()
            mock_gmail_class.return_value = mock_client
            
            # First call should create the client
            client1 = await client_manager.get_client("gmail")
            # Second call should return the same client
            client2 = await client_manager.get_client("gmail")
            
            assert client1 == client2
            assert client1 == mock_client
            # Should only create once
            mock_gmail_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authentication_flow(self, client_manager, mock_auth_manager):
        """Test authentication flow"""
        client_manager.configure_service(
            service="gmail",
            client_id="test_client_id",
            client_secret="test_client_secret"
        )
        
        mock_client = AsyncMock()
        mock_client.authenticate.side_effect = Exception("Please visit this URL to authorize: https://auth.url")
        
        with patch('src.integrations.client_manager.GmailClient', return_value=mock_client):
            auth_url = await client_manager.authenticate_service("gmail")
            
            assert auth_url == "https://auth.url"
            mock_client.authenticate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_oauth_callback_handling(self, client_manager):
        """Test OAuth callback handling"""
        client_manager.configure_service(
            service="gmail",
            client_id="test_client_id",
            client_secret="test_client_secret"
        )
        
        mock_client = AsyncMock()
        mock_client.handle_oauth_callback.return_value = True
        
        with patch('src.integrations.client_manager.GmailClient', return_value=mock_client):
            success = await client_manager.handle_oauth_callback(
                service="gmail",
                authorization_code="test_code",
                state="test_state"
            )
            
            assert success is True
            mock_client.handle_oauth_callback.assert_called_once_with("test_code", "test_state")
            
            # Check that service status was updated
            assert "gmail" in client_manager._service_status
            status = client_manager._service_status["gmail"]
            assert status.authenticated is True
            assert status.healthy is True
    
    @pytest.mark.asyncio
    async def test_check_authentication(self, client_manager, mock_auth_manager):
        """Test authentication checking"""
        mock_auth_manager.list_tokens.return_value = [
            {"status": "valid", "provider": "gmail", "expires_at": datetime.now() + timedelta(hours=1)}
        ]
        
        is_authenticated = await client_manager.check_authentication("gmail")
        
        assert is_authenticated is True
        mock_auth_manager.list_tokens.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_refresh_tokens(self, client_manager):
        """Test token refresh functionality"""
        client_manager.configure_service(
            service="gmail",
            client_id="test_client_id",
            client_secret="test_client_secret"
        )
        
        mock_client = AsyncMock()
        mock_client.refresh_token.return_value = True
        
        with patch('src.integrations.client_manager.GmailClient', return_value=mock_client):
            client_manager._clients["gmail"] = mock_client
            
            results = await client_manager.refresh_tokens("gmail")
            
            assert results == {"gmail": True}
            mock_client.refresh_token.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_service_status(self, client_manager):
        """Test service status retrieval"""
        # Add some mock status
        status = ServiceStatus(
            service_name="gmail",
            authenticated=True,
            healthy=True,
            last_check=datetime.now()
        )
        client_manager._service_status["gmail"] = status
        
        # Get specific service status
        single_status = await client_manager.get_service_status("gmail")
        assert "gmail" in single_status
        assert single_status["gmail"] == status
        
        # Get all service statuses
        all_statuses = await client_manager.get_service_status()
        assert "gmail" in all_statuses
        assert all_statuses["gmail"] == status
    
    @pytest.mark.asyncio
    async def test_get_overall_health(self, client_manager, mock_auth_manager):
        """Test overall health status"""
        # Mock status for multiple services
        statuses = {
            "gmail": ServiceStatus("gmail", True, True, datetime.now()),
            "drive": ServiceStatus("drive", True, False, datetime.now(), "Connection timeout"),
            "twitter": ServiceStatus("twitter", False, False, datetime.now(), "Not authenticated")
        }
        client_manager._service_status = statuses
        
        with patch.object(client_manager.rate_limiter, 'get_all_status', return_value={}):
            with patch.object(client_manager.client_registry, 'get_all_health_status', return_value={}):
                health = await client_manager.get_overall_health()
                
                assert health["services"]["total"] == 3
                assert health["services"]["healthy"] == 1
                assert health["services"]["authenticated"] == 2
                assert health["services"]["unhealthy"] == 2
                assert health["overall_health"] == "degraded"
    
    @pytest.mark.asyncio
    async def test_bulk_export(self, client_manager, temp_dir):
        """Test bulk data export functionality"""
        client_manager.configure_service(
            service="gmail",
            client_id="test_client_id",
            client_secret="test_client_secret"
        )
        
        mock_client = AsyncMock()
        mock_client.export_data.return_value = 150  # Mock export count
        
        with patch('src.integrations.client_manager.GmailClient', return_value=mock_client):
            client_manager._clients["gmail"] = mock_client
            
            with patch.object(client_manager, 'check_authentication', return_value=True):
                results = await client_manager.bulk_export(
                    output_directory=temp_dir,
                    services=["gmail"],
                    max_items=1000
                )
                
                assert results == {"gmail": 150}
                mock_client.export_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_sync(self, client_manager):
        """Test data synchronization between services"""
        # Configure both services
        for service in ["twitter", "linkedin"]:
            client_manager.configure_service(
                service=service,
                client_id="test_client_id",
                client_secret="test_client_secret"
            )
        
        mock_twitter_client = AsyncMock()
        mock_twitter_client.get_home_timeline.return_value = {
            'tweets': [
                MagicMock(id='123', text='Test tweet content')
            ]
        }
        
        mock_linkedin_client = AsyncMock()
        mock_linkedin_client.create_post.return_value = {'id': 'linkedin_post_123'}
        
        with patch('src.integrations.client_manager.TwitterClient', return_value=mock_twitter_client):
            with patch('src.integrations.client_manager.LinkedInClient', return_value=mock_linkedin_client):
                client_manager._clients["twitter"] = mock_twitter_client
                client_manager._clients["linkedin"] = mock_linkedin_client
                
                with patch.object(client_manager, 'check_authentication', return_value=True):
                    synced_count = await client_manager.sync_data(
                        source_service="twitter",
                        target_service="linkedin",
                        data_type="posts",
                        max_items=10
                    )
                    
                    assert synced_count == 1
                    mock_twitter_client.get_home_timeline.assert_called_once()
                    mock_linkedin_client.create_post.assert_called_once()


class TestGmailClientMocking:
    """Test Gmail client with comprehensive mocking"""
    
    @pytest.fixture
    def mock_gmail_client(self):
        """Create a mock Gmail client"""
        with patch('src.integrations.gmail_client.build') as mock_build:
            mock_service = MagicMock()
            mock_build.return_value = mock_service
            
            client = GmailClient(
                client_id="test_client_id",
                client_secret="test_client_secret",
                scopes=["https://www.googleapis.com/auth/gmail.readonly"]
            )
            
            client._service = mock_service
            client._authenticated = True
            
            return client, mock_service
    
    @pytest.mark.asyncio
    async def test_gmail_search_messages(self, mock_gmail_client):
        """Test Gmail message search functionality"""
        client, mock_service = mock_gmail_client
        
        # Mock the API response
        mock_service.users().messages().list().execute.return_value = {
            'messages': [
                {'id': '123', 'threadId': 'thread1'},
                {'id': '456', 'threadId': 'thread2'}
            ]
        }
        
        mock_service.users().messages().get().execute.side_effect = [
            {
                'id': '123',
                'snippet': 'Test message 1',
                'payload': {
                    'headers': [
                        {'name': 'Subject', 'value': 'Test Subject 1'},
                        {'name': 'From', 'value': 'sender1@example.com'},
                        {'name': 'To', 'value': 'recipient@example.com'}
                    ]
                },
                'internalDate': '1640995200000'  # Unix timestamp in milliseconds
            },
            {
                'id': '456',
                'snippet': 'Test message 2',
                'payload': {
                    'headers': [
                        {'name': 'Subject', 'value': 'Test Subject 2'},
                        {'name': 'From', 'value': 'sender2@example.com'},
                        {'name': 'To', 'value': 'recipient@example.com'}
                    ]
                },
                'internalDate': '1640995260000'
            }
        ]
        
        messages = await client.search_messages(query="test query", max_results=10)
        
        assert len(messages) == 2
        assert messages[0]['subject'] == 'Test Subject 1'
        assert messages[1]['subject'] == 'Test Subject 2'
        
        # Verify API calls
        mock_service.users().messages().list.assert_called_once()
        assert mock_service.users().messages().get().execute.call_count == 2
    
    @pytest.mark.asyncio
    async def test_gmail_send_message(self, mock_gmail_client):
        """Test Gmail message sending functionality"""
        client, mock_service = mock_gmail_client
        
        mock_service.users().messages().send().execute.return_value = {
            'id': 'sent_message_123',
            'labelIds': ['SENT']
        }
        
        result = await client.send_message(
            to=["recipient@example.com"],
            subject="Test Subject",
            body="Test message body"
        )
        
        assert result['id'] == 'sent_message_123'
        mock_service.users().messages().send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_gmail_get_labels(self, mock_gmail_client):
        """Test Gmail labels retrieval"""
        client, mock_service = mock_gmail_client
        
        mock_service.users().labels().list().execute.return_value = {
            'labels': [
                {'id': 'INBOX', 'name': 'INBOX', 'type': 'system'},
                {'id': 'SENT', 'name': 'SENT', 'type': 'system'},
                {'id': 'custom_label_1', 'name': 'Work', 'type': 'user'}
            ]
        }
        
        labels = await client.get_labels()
        
        assert len(labels) == 3
        assert any(label['name'] == 'INBOX' for label in labels)
        assert any(label['name'] == 'Work' for label in labels)


class TestDriveClientMocking:
    """Test Drive client with comprehensive mocking"""
    
    @pytest.fixture
    def mock_drive_client(self):
        """Create a mock Drive client"""
        with patch('src.integrations.drive_client.build') as mock_build:
            mock_service = MagicMock()
            mock_build.return_value = mock_service
            
            client = DriveClient(
                client_id="test_client_id",
                client_secret="test_client_secret",
                scopes=["https://www.googleapis.com/auth/drive.readonly"]
            )
            
            client._service = mock_service
            client._authenticated = True
            
            return client, mock_service
    
    @pytest.mark.asyncio
    async def test_drive_search_files(self, mock_drive_client):
        """Test Drive file search functionality"""
        client, mock_service = mock_drive_client
        
        mock_service.files().list().execute.return_value = {
            'files': [
                {
                    'id': 'file1',
                    'name': 'Document1.pdf',
                    'mimeType': 'application/pdf',
                    'size': '1024000',
                    'modifiedTime': '2023-01-01T12:00:00.000Z'
                },
                {
                    'id': 'file2',
                    'name': 'Spreadsheet1.xlsx',
                    'mimeType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'size': '512000',
                    'modifiedTime': '2023-01-02T12:00:00.000Z'
                }
            ]
        }
        
        files = await client.search_files(query="test query", max_results=10)
        
        assert len(files) == 2
        assert files[0]['name'] == 'Document1.pdf'
        assert files[1]['name'] == 'Spreadsheet1.xlsx'
        
        mock_service.files().list.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_drive_get_file_content(self, mock_drive_client):
        """Test Drive file content retrieval"""
        client, mock_service = mock_drive_client
        
        mock_service.files().get().execute.return_value = {
            'id': 'file1',
            'name': 'test.txt',
            'mimeType': 'text/plain'
        }
        
        mock_service.files().get_media().execute.return_value = b"Test file content"
        
        content = await client.get_file_content("file1")
        
        assert content == "Test file content"
        mock_service.files().get.assert_called_once()
        mock_service.files().get_media.assert_called_once()


class TestTwitterClientMocking:
    """Test Twitter client with comprehensive mocking"""
    
    @pytest.fixture
    def mock_twitter_client(self):
        """Create a mock Twitter client"""
        client = TwitterClient(
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["tweet.read", "tweet.write"]
        )
        client._authenticated = True
        return client
    
    @pytest.mark.asyncio
    async def test_twitter_search_tweets(self, mock_twitter_client):
        """Test Twitter tweet search functionality"""
        client = mock_twitter_client
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                'data': [
                    {
                        'id': '123456789',
                        'text': 'This is a test tweet #testing',
                        'created_at': '2023-01-01T12:00:00.000Z',
                        'author_id': 'user123',
                        'public_metrics': {
                            'like_count': 10,
                            'retweet_count': 5,
                            'reply_count': 2
                        }
                    }
                ],
                'meta': {
                    'result_count': 1
                }
            }
            
            tweets = await client.search_tweets(query="test query", max_results=10)
            
            assert len(tweets) == 1
            assert tweets[0]['text'] == 'This is a test tweet #testing'
            assert tweets[0]['public_metrics']['like_count'] == 10
    
    @pytest.mark.asyncio
    async def test_twitter_post_tweet(self, mock_twitter_client):
        """Test Twitter tweet posting functionality"""
        client = mock_twitter_client
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                'data': {
                    'id': '987654321',
                    'text': 'Posted test tweet'
                }
            }
            
            result = await client.post_tweet(text="Posted test tweet")
            
            assert result['id'] == '987654321'
            assert result['text'] == 'Posted test tweet'
    
    @pytest.mark.asyncio
    async def test_twitter_get_user_timeline(self, mock_twitter_client):
        """Test Twitter user timeline retrieval"""
        client = mock_twitter_client
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                'data': [
                    {
                        'id': '111',
                        'text': 'First tweet',
                        'created_at': '2023-01-01T12:00:00.000Z'
                    },
                    {
                        'id': '222',
                        'text': 'Second tweet',
                        'created_at': '2023-01-01T13:00:00.000Z'
                    }
                ]
            }
            
            timeline = await client.get_home_timeline(max_results=10)
            
            assert len(timeline['tweets']) == 2
            assert timeline['tweets'][0]['text'] == 'First tweet'


class TestLinkedInClientMocking:
    """Test LinkedIn client with comprehensive mocking"""
    
    @pytest.fixture
    def mock_linkedin_client(self):
        """Create a mock LinkedIn client"""
        client = LinkedInClient(
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["r_liteprofile", "w_member_social"]
        )
        client._authenticated = True
        return client
    
    @pytest.mark.asyncio
    async def test_linkedin_create_post(self, mock_linkedin_client):
        """Test LinkedIn post creation functionality"""
        client = mock_linkedin_client
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                'id': 'linkedin_post_123'
            }
            
            result = await client.create_post(
                text="Test LinkedIn post",
                visibility="PUBLIC"
            )
            
            assert result['id'] == 'linkedin_post_123'
    
    @pytest.mark.asyncio
    async def test_linkedin_get_profile(self, mock_linkedin_client):
        """Test LinkedIn profile retrieval"""
        client = mock_linkedin_client
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                'id': 'user123',
                'firstName': {'localized': {'en_US': 'John'}},
                'lastName': {'localized': {'en_US': 'Doe'}},
                'profilePicture': {
                    'displayImage': 'urn:li:digitalmediaAsset:image123'
                }
            }
            
            profile = await client.get_profile()
            
            assert profile['id'] == 'user123'
            assert profile['firstName']['localized']['en_US'] == 'John'


class TestAPIClientErrorHandling:
    """Test error handling across all API clients"""
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self):
        """Test handling of authentication errors"""
        client = GmailClient(
            client_id="invalid_client_id",
            client_secret="invalid_client_secret",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        
        with patch('src.integrations.gmail_client.build') as mock_build:
            mock_build.side_effect = Exception("Authentication failed")
            
            with pytest.raises(Exception) as exc_info:
                await client.authenticate("http://localhost:8080/callback")
            
            assert "Authentication failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_gmail_client):
        """Test handling of rate limit errors"""
        client, mock_service = mock_gmail_client
        
        # Mock rate limit error
        from googleapiclient.errors import HttpError
        error_response = MagicMock()
        error_response.status = 429
        error_response.reason = "Rate Limit Exceeded"
        
        mock_service.users().messages().list().execute.side_effect = HttpError(
            resp=error_response,
            content=b'{"error": {"code": 429, "message": "Rate limit exceeded"}}'
        )
        
        with pytest.raises(Exception) as exc_info:
            await client.search_messages(query="test", max_results=10)
        
        # Should handle rate limit appropriately
        assert "429" in str(exc_info.value) or "rate limit" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, mock_twitter_client):
        """Test handling of network errors"""
        client = mock_twitter_client
        
        with patch.object(client, '_make_request') as mock_request:
            mock_request.side_effect = ConnectionError("Network unreachable")
            
            with pytest.raises(ConnectionError):
                await client.search_tweets(query="test", max_results=10)
    
    @pytest.mark.asyncio
    async def test_invalid_response_handling(self, mock_drive_client):
        """Test handling of invalid API responses"""
        client, mock_service = mock_drive_client
        
        # Mock invalid response structure
        mock_service.files().list().execute.return_value = {
            'invalid_key': 'invalid_data'
            # Missing 'files' key
        }
        
        files = await client.search_files(query="test", max_results=10)
        
        # Should handle gracefully and return empty list or appropriate default
        assert isinstance(files, list)


class TestAPIClientPerformance:
    """Test performance characteristics of API clients"""
    
    @pytest.mark.asyncio
    async def test_concurrent_api_calls(self, mock_gmail_client):
        """Test concurrent API calls performance"""
        client, mock_service = mock_gmail_client
        
        # Mock fast responses
        mock_service.users().messages().list().execute.return_value = {
            'messages': [{'id': '123', 'threadId': 'thread1'}]
        }
        mock_service.users().messages().get().execute.return_value = {
            'id': '123',
            'snippet': 'Test message',
            'payload': {'headers': []}
        }
        
        import time
        start_time = time.time()
        
        # Make multiple concurrent calls
        tasks = [
            client.search_messages(query=f"query_{i}", max_results=1)
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        
        # Should complete all calls
        assert len(results) == 5
        for result in results:
            assert isinstance(result, list)
        
        # Should be reasonably fast (allowing for mocking overhead)
        assert execution_time < 5.0  # 5 seconds is very generous for mocked calls
    
    @pytest.mark.asyncio
    async def test_client_resource_cleanup(self, mock_client_manager):
        """Test that clients properly clean up resources"""
        manager = APIClientManager()
        await manager.initialize()
        
        # Add some mock clients
        manager._clients = {
            'gmail': MagicMock(),
            'drive': MagicMock(),
            'twitter': MagicMock()
        }
        
        # Shutdown should clean up all clients
        await manager.shutdown()
        
        # Verify cleanup tasks were initiated
        assert manager._health_check_task.cancelled()
        assert manager._token_refresh_task.cancelled()