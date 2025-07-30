"""
Unit tests for Security Features

Tests encryption, authentication, privacy controls, and audit logging.
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from typing import Dict, Any

from src.config.settings import Settings, Environment, EncryptionSettings, SecuritySettings, PrivacySettings
from src.config.auth import AuthProvider, AuthManager, TokenInfo
from src.utils.encryption import EncryptionManager, DataEncryption
from src.utils.audit import AuditLogger, AuditEvent, EventType


class TestEncryptionSettings:
    """Test encryption configuration and validation"""
    
    def test_encryption_settings_defaults(self):
        """Test encryption settings have secure defaults"""
        settings = EncryptionSettings()
        
        assert settings.pbkdf2_iterations >= 50000
        assert settings.argon2_time_cost >= 2
        assert settings.argon2_memory_cost >= 1024
        assert settings.token_expiry_hours >= 1
        assert settings.refresh_token_expiry_days >= 1
    
    def test_encryption_settings_validation(self):
        """Test encryption settings validation"""
        # Test minimum values
        settings = EncryptionSettings(
            pbkdf2_iterations=100000,
            argon2_time_cost=2,
            argon2_memory_cost=65536
        )
        
        assert settings.pbkdf2_iterations == 100000
        assert settings.argon2_time_cost == 2
        assert settings.argon2_memory_cost == 65536
    
    def test_master_key_generation(self):
        """Test master key is properly generated"""
        settings = EncryptionSettings()
        
        master_key = settings.master_key.get_secret_value()
        assert len(master_key) > 0
        assert isinstance(master_key, str)
        
        # Should be different each time
        settings2 = EncryptionSettings()
        master_key2 = settings2.master_key.get_secret_value()
        assert master_key != master_key2


class TestSecuritySettings:
    """Test security configuration and validation"""
    
    def test_security_settings_defaults(self):
        """Test security settings have reasonable defaults"""
        settings = SecuritySettings()
        
        assert settings.session_timeout_minutes >= 5
        assert settings.max_login_attempts >= 3
        assert settings.lockout_duration_minutes >= 5
        assert settings.api_rate_limit_per_minute >= 10
        assert settings.max_file_size_mb >= 1
        assert len(settings.allowed_file_types) > 0
    
    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration"""
        settings = SecuritySettings(
            api_rate_limit_per_minute=120,
            cache_rate_limit_per_hour=1000
        )
        
        assert settings.api_rate_limit_per_minute == 120
        assert settings.cache_rate_limit_per_hour == 1000
    
    def test_cors_configuration(self):
        """Test CORS configuration"""
        settings = SecuritySettings(
            allowed_origins=["https://example.com", "https://app.example.com"]
        )
        
        assert "https://example.com" in settings.allowed_origins
        assert "https://app.example.com" in settings.allowed_origins
    
    def test_file_upload_security(self):
        """Test file upload security settings"""
        settings = SecuritySettings(
            max_file_size_mb=5,
            allowed_file_types=[".pdf", ".txt", ".json"]
        )
        
        assert settings.max_file_size_mb == 5
        assert ".pdf" in settings.allowed_file_types
        assert ".exe" not in settings.allowed_file_types


class TestPrivacySettings:
    """Test privacy configuration and controls"""
    
    def test_privacy_settings_defaults(self):
        """Test privacy settings default to privacy-protective values"""
        settings = PrivacySettings()
        
        assert settings.anonymize_logs is True  # Privacy by default
        assert settings.require_explicit_consent is True
        assert settings.enable_data_export is True  # User rights
        assert settings.enable_data_deletion is True  # User rights
        assert settings.cache_retention_days >= 1
        assert settings.log_retention_days >= 30
    
    def test_data_retention_configuration(self):
        """Test data retention configuration"""
        settings = PrivacySettings(
            cache_retention_days=14,
            log_retention_days=180,
            audit_retention_years=3
        )
        
        assert settings.cache_retention_days == 14
        assert settings.log_retention_days == 180
        assert settings.audit_retention_years == 3
    
    def test_geographic_restrictions(self):
        """Test geographic restriction configuration"""
        settings = PrivacySettings(
            allowed_countries=["US", "CA", "GB"],
            blocked_countries=["XX", "YY"]
        )
        
        assert "US" in settings.allowed_countries
        assert "XX" in settings.blocked_countries
    
    def test_processing_purposes(self):
        """Test data processing purposes configuration"""
        settings = PrivacySettings()
        
        assert "personal_knowledge_management" in settings.processing_purposes
        assert len(settings.processing_purposes) > 0


class TestEncryptionManager:
    """Test encryption functionality"""
    
    @pytest.fixture
    def encryption_manager(self, test_settings):
        """Create an encryption manager for testing"""
        return EncryptionManager(test_settings.encryption)
    
    def test_encryption_manager_initialization(self, encryption_manager):
        """Test encryption manager initializes properly"""
        assert encryption_manager is not None
        assert encryption_manager.settings is not None
    
    def test_data_encryption_decryption(self, encryption_manager):
        """Test data encryption and decryption"""
        original_data = "This is sensitive test data that needs encryption"
        
        # Encrypt data
        encrypted_data = encryption_manager.encrypt_data(original_data)
        
        assert encrypted_data != original_data
        assert len(encrypted_data) > 0
        assert isinstance(encrypted_data, str)
        
        # Decrypt data
        decrypted_data = encryption_manager.decrypt_data(encrypted_data)
        
        assert decrypted_data == original_data
    
    def test_different_data_produces_different_ciphertext(self, encryption_manager):
        """Test that different data produces different ciphertext"""
        data1 = "First piece of data"
        data2 = "Second piece of data"
        
        encrypted1 = encryption_manager.encrypt_data(data1)
        encrypted2 = encryption_manager.encrypt_data(data2)
        
        assert encrypted1 != encrypted2
    
    def test_same_data_produces_different_ciphertext(self, encryption_manager):
        """Test that same data produces different ciphertext (due to randomness)"""
        data = "Same data encrypted twice"
        
        encrypted1 = encryption_manager.encrypt_data(data)
        encrypted2 = encryption_manager.encrypt_data(data)
        
        # Should be different due to random IV/salt
        assert encrypted1 != encrypted2
        
        # But both should decrypt to the same original data
        assert encryption_manager.decrypt_data(encrypted1) == data
        assert encryption_manager.decrypt_data(encrypted2) == data
    
    def test_password_hashing(self, encryption_manager):
        """Test password hashing functionality"""
        password = "secure_test_password_123"
        
        hashed_password = encryption_manager.hash_password(password)
        
        assert hashed_password != password
        assert len(hashed_password) > 0
        assert isinstance(hashed_password, str)
        
        # Should be able to verify the password
        assert encryption_manager.verify_password(password, hashed_password) is True
        assert encryption_manager.verify_password("wrong_password", hashed_password) is False
    
    def test_key_derivation(self, encryption_manager):
        """Test key derivation functionality"""
        password = "test_password"
        salt = "test_salt"
        
        key1 = encryption_manager.derive_key(password, salt)
        key2 = encryption_manager.derive_key(password, salt)
        
        # Same password and salt should produce same key
        assert key1 == key2
        
        # Different salt should produce different key
        key3 = encryption_manager.derive_key(password, "different_salt")
        assert key1 != key3
    
    def test_secure_random_generation(self, encryption_manager):
        """Test secure random data generation"""
        random1 = encryption_manager.generate_secure_random(32)
        random2 = encryption_manager.generate_secure_random(32)
        
        assert len(random1) == 32
        assert len(random2) == 32
        assert random1 != random2
        assert isinstance(random1, bytes)
    
    def test_encryption_with_invalid_data(self, encryption_manager):
        """Test encryption with invalid data"""
        # Should handle None gracefully
        with pytest.raises((TypeError, ValueError)):
            encryption_manager.encrypt_data(None)
        
        # Should handle empty string
        encrypted_empty = encryption_manager.encrypt_data("")
        decrypted_empty = encryption_manager.decrypt_data(encrypted_empty)
        assert decrypted_empty == ""
    
    def test_decryption_with_invalid_data(self, encryption_manager):
        """Test decryption with invalid data"""
        # Should raise exception for invalid ciphertext
        with pytest.raises(Exception):
            encryption_manager.decrypt_data("invalid_ciphertext")
        
        with pytest.raises((TypeError, ValueError)):
            encryption_manager.decrypt_data(None)


class TestAuthManager:
    """Test authentication manager functionality"""
    
    @pytest.fixture
    async def auth_manager(self, test_settings, temp_dir):
        """Create an auth manager for testing"""
        settings = test_settings
        settings.database.data_directory = temp_dir
        
        manager = AuthManager(settings)
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_auth_manager_initialization(self, auth_manager):
        """Test auth manager initializes properly"""
        assert auth_manager is not None
        assert auth_manager.settings is not None
        assert auth_manager._initialized is True
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_credentials(self, auth_manager):
        """Test storing and retrieving credentials"""
        credentials = {
            'access_token': 'test_access_token_123',
            'refresh_token': 'test_refresh_token_456',
            'expires_at': datetime.now() + timedelta(hours=1),
            'token_type': 'Bearer',
            'scope': ['read', 'write']
        }
        
        # Store credentials
        await auth_manager.store_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user",
            credentials=credentials
        )
        
        # Retrieve credentials
        retrieved = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user"
        )
        
        assert retrieved is not None
        assert retrieved['access_token'] == 'test_access_token_123'
        assert retrieved['refresh_token'] == 'test_refresh_token_456'
        assert retrieved['token_type'] == 'Bearer'
    
    @pytest.mark.asyncio
    async def test_token_validation(self, auth_manager):
        """Test token validation functionality"""
        # Store valid token
        credentials = {
            'access_token': 'valid_token',
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        await auth_manager.store_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user",
            credentials=credentials
        )
        
        # Validate token
        is_valid = await auth_manager.validate_token(
            provider=AuthProvider.GMAIL,
            user_id="test_user",
            token="valid_token"
        )
        
        assert is_valid is True
        
        # Test invalid token
        is_invalid = await auth_manager.validate_token(
            provider=AuthProvider.GMAIL,
            user_id="test_user",
            token="invalid_token"
        )
        
        assert is_invalid is False
    
    @pytest.mark.asyncio
    async def test_token_expiration_handling(self, auth_manager):
        """Test handling of expired tokens"""
        # Store expired token
        expired_credentials = {
            'access_token': 'expired_token',
            'expires_at': datetime.now() - timedelta(hours=1)  # Expired
        }
        
        await auth_manager.store_credentials(
            provider=AuthProvider.TWITTER,
            user_id="test_user",
            credentials=expired_credentials
        )
        
        # Should detect as expired
        is_valid = await auth_manager.validate_token(
            provider=AuthProvider.TWITTER,
            user_id="test_user",
            token="expired_token"
        )
        
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_token_refresh(self, auth_manager):
        """Test token refresh functionality"""
        old_credentials = {
            'access_token': 'old_access_token',
            'refresh_token': 'refresh_token_123',
            'expires_at': datetime.now() - timedelta(minutes=5)  # Recently expired
        }
        
        await auth_manager.store_credentials(
            provider=AuthProvider.LINKEDIN,
            user_id="test_user",
            credentials=old_credentials
        )
        
        new_credentials = {
            'access_token': 'new_access_token',
            'refresh_token': 'new_refresh_token',
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        # Simulate token refresh
        success = await auth_manager.refresh_token(
            provider=AuthProvider.LINKEDIN,
            user_id="test_user",
            new_credentials=new_credentials
        )
        
        assert success is True
        
        # Verify new credentials are stored
        retrieved = await auth_manager.get_user_credentials(
            provider=AuthProvider.LINKEDIN,
            user_id="test_user"
        )
        
        assert retrieved['access_token'] == 'new_access_token'
    
    @pytest.mark.asyncio
    async def test_list_tokens(self, auth_manager):
        """Test listing stored tokens"""
        # Store multiple tokens
        providers_and_tokens = [
            (AuthProvider.GMAIL, 'gmail_token'),
            (AuthProvider.DRIVE, 'drive_token'),
            (AuthProvider.TWITTER, 'twitter_token')
        ]
        
        for provider, token in providers_and_tokens:
            credentials = {
                'access_token': token,
                'expires_at': datetime.now() + timedelta(hours=1)
            }
            await auth_manager.store_credentials(
                provider=provider,
                user_id="test_user",
                credentials=credentials
            )
        
        # List all tokens
        tokens = await auth_manager.list_tokens()
        
        assert len(tokens) >= 3
        token_providers = [token['provider'] for token in tokens]
        assert 'gmail' in token_providers
        assert 'drive' in token_providers
        assert 'twitter' in token_providers
    
    @pytest.mark.asyncio
    async def test_delete_credentials(self, auth_manager):
        """Test deleting stored credentials"""
        credentials = {
            'access_token': 'token_to_delete',
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        await auth_manager.store_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user",
            credentials=credentials
        )
        
        # Verify it exists
        retrieved = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user"
        )
        assert retrieved is not None
        
        # Delete it
        success = await auth_manager.delete_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user"
        )
        assert success is True
        
        # Verify it's gone
        retrieved_after = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user"
        )
        assert retrieved_after is None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_tokens(self, auth_manager):
        """Test cleanup of expired tokens"""
        # Store mix of valid and expired tokens
        valid_credentials = {
            'access_token': 'valid_token',
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        expired_credentials = {
            'access_token': 'expired_token',
            'expires_at': datetime.now() - timedelta(hours=1)
        }
        
        await auth_manager.store_credentials(
            provider=AuthProvider.GMAIL,
            user_id="valid_user",
            credentials=valid_credentials
        )
        
        await auth_manager.store_credentials(
            provider=AuthProvider.TWITTER,
            user_id="expired_user",
            credentials=expired_credentials
        )
        
        # Clean up expired tokens
        cleaned_count = await auth_manager.cleanup_expired_tokens()
        
        assert cleaned_count >= 1
        
        # Verify valid token still exists
        valid_still_exists = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id="valid_user"
        )
        assert valid_still_exists is not None
        
        # Verify expired token is gone
        expired_gone = await auth_manager.get_user_credentials(
            provider=AuthProvider.TWITTER,
            user_id="expired_user"
        )
        assert expired_gone is None


class TestAuditLogger:
    """Test audit logging functionality"""
    
    @pytest.fixture
    def audit_logger(self, test_settings, temp_dir):
        """Create an audit logger for testing"""
        settings = test_settings
        settings.database.logs_directory = temp_dir
        settings.audit.audit_enabled = True
        
        return AuditLogger(settings)
    
    def test_audit_logger_initialization(self, audit_logger):
        """Test audit logger initializes properly"""
        assert audit_logger is not None
        assert audit_logger.settings is not None
        assert audit_logger.settings.audit.audit_enabled is True
    
    def test_log_authentication_event(self, audit_logger):
        """Test logging authentication events"""
        event = AuditEvent(
            event_type=EventType.AUTHENTICATION,
            user_id="test_user",
            action="login_success",
            resource="gmail_api",
            timestamp=datetime.now(),
            ip_address="192.168.1.100",
            user_agent="Test User Agent",
            details={"provider": "gmail", "method": "oauth2"}
        )
        
        # Should not raise any exceptions
        audit_logger.log_event(event)
        
        # Verify log file was created and contains the event
        log_files = list(audit_logger.settings.database.logs_directory.glob("audit_*.log"))
        assert len(log_files) > 0
        
        # Read the log file and verify content
        log_content = log_files[0].read_text()
        assert "login_success" in log_content
        assert "test_user" in log_content
        assert "gmail_api" in log_content
    
    def test_log_data_access_event(self, audit_logger):
        """Test logging data access events"""
        event = AuditEvent(
            event_type=EventType.DATA_ACCESS,
            user_id="test_user",
            action="email_search",
            resource="gmail_messages",
            timestamp=datetime.now(),
            details={
                "query": "project alpha",
                "results_count": 15,
                "time_range": "last_30_days"
            }
        )
        
        audit_logger.log_event(event)
        
        # Verify event was logged
        log_files = list(audit_logger.settings.database.logs_directory.glob("audit_*.log"))
        assert len(log_files) > 0
        
        log_content = log_files[0].read_text()
        assert "email_search" in log_content
        assert "gmail_messages" in log_content
    
    def test_log_configuration_change_event(self, audit_logger):
        """Test logging configuration change events"""
        event = AuditEvent(
            event_type=EventType.CONFIGURATION_CHANGE,
            user_id="admin_user",
            action="privacy_setting_changed",
            resource="privacy_settings",
            timestamp=datetime.now(),
            details={
                "setting": "anonymize_logs",
                "old_value": False,
                "new_value": True
            }
        )
        
        audit_logger.log_event(event)
        
        log_files = list(audit_logger.settings.database.logs_directory.glob("audit_*.log"))
        log_content = log_files[0].read_text()
        
        assert "privacy_setting_changed" in log_content
        assert "anonymize_logs" in log_content
    
    def test_log_security_event(self, audit_logger):
        """Test logging security events"""
        event = AuditEvent(
            event_type=EventType.SECURITY_EVENT,
            user_id="potential_attacker",
            action="failed_login_attempt",
            resource="authentication",
            timestamp=datetime.now(),
            ip_address="10.0.0.1",
            details={
                "reason": "invalid_credentials",
                "attempt_count": 5,
                "lockout_triggered": True
            }
        )
        
        audit_logger.log_event(event)
        
        log_files = list(audit_logger.settings.database.logs_directory.glob("audit_*.log"))
        log_content = log_files[0].read_text()
        
        assert "failed_login_attempt" in log_content
        assert "lockout_triggered" in log_content
    
    def test_audit_log_rotation(self, audit_logger, temp_dir):
        """Test audit log file rotation"""
        # Configure small max file size to trigger rotation
        audit_logger.settings.audit.audit_max_file_size_mb = 0.001  # Very small for testing
        
        # Log many events to trigger rotation
        for i in range(100):
            event = AuditEvent(
                event_type=EventType.DATA_ACCESS,
                user_id=f"user_{i}",
                action="test_action",
                resource="test_resource",
                timestamp=datetime.now(),
                details={"iteration": i, "data": "x" * 1000}  # Make it large
            )
            audit_logger.log_event(event)
        
        # Should have multiple log files due to rotation
        log_files = list(temp_dir.glob("audit_*.log"))
        assert len(log_files) >= 1  # At least one file should exist
    
    def test_audit_log_with_privacy_mode(self, audit_logger):
        """Test audit logging with privacy mode enabled"""
        # Enable privacy mode
        audit_logger.settings.privacy.anonymize_logs = True
        
        event = AuditEvent(
            event_type=EventType.DATA_ACCESS,
            user_id="sensitive_user@example.com",
            action="email_access",
            resource="gmail_messages",
            timestamp=datetime.now(),
            ip_address="192.168.1.100",
            details={
                "email_from": "secret@company.com",
                "email_subject": "Confidential Information"
            }
        )
        
        audit_logger.log_event(event)
        
        # Verify sensitive information is anonymized
        log_files = list(audit_logger.settings.database.logs_directory.glob("audit_*.log"))
        log_content = log_files[0].read_text()
        
        # Original sensitive data should not appear in logs
        assert "sensitive_user@example.com" not in log_content
        assert "secret@company.com" not in log_content
        assert "192.168.1.100" not in log_content
        
        # But action and resource should still be logged
        assert "email_access" in log_content
        assert "gmail_messages" in log_content
    
    def test_audit_log_integrity_check(self, audit_logger):
        """Test audit log integrity verification"""
        # Log some events
        events = [
            AuditEvent(
                event_type=EventType.AUTHENTICATION,
                user_id="user1",
                action="login",
                resource="system",
                timestamp=datetime.now()
            ),
            AuditEvent(
                event_type=EventType.DATA_ACCESS,
                user_id="user1",
                action="search",
                resource="emails",
                timestamp=datetime.now()
            )
        ]
        
        for event in events:
            audit_logger.log_event(event)
        
        # Verify integrity
        integrity_ok = audit_logger.verify_log_integrity()
        assert integrity_ok is True
        
        # Corrupt the log file
        log_files = list(audit_logger.settings.database.logs_directory.glob("audit_*.log"))
        if log_files:
            log_file = log_files[0]
            corrupted_content = log_file.read_text() + "\nCORRUPTED LINE"
            log_file.write_text(corrupted_content)
            
            # Integrity check should now fail
            integrity_after_corruption = audit_logger.verify_log_integrity()
            assert integrity_after_corruption is False


class TestSecurityIntegration:
    """Test integration between security components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_flow(self, test_settings, temp_dir):
        """Test complete security flow from authentication to audit"""
        # Initialize all security components
        test_settings.database.data_directory = temp_dir
        test_settings.database.logs_directory = temp_dir
        
        encryption_manager = EncryptionManager(test_settings.encryption)
        auth_manager = AuthManager(test_settings)
        await auth_manager.initialize()
        audit_logger = AuditLogger(test_settings)
        
        # Step 1: Store encrypted credentials
        sensitive_credentials = {
            'access_token': 'very_sensitive_access_token_12345',
            'refresh_token': 'very_sensitive_refresh_token_67890',
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        await auth_manager.store_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user",
            credentials=sensitive_credentials
        )
        
        # Step 2: Log authentication event
        auth_event = AuditEvent(
            event_type=EventType.AUTHENTICATION,
            user_id="test_user",
            action="credentials_stored",
            resource="gmail_api",
            timestamp=datetime.now(),
            details={"provider": "gmail"}
        )
        audit_logger.log_event(auth_event)
        
        # Step 3: Retrieve and verify credentials
        retrieved_credentials = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user"
        )
        
        assert retrieved_credentials is not None
        assert retrieved_credentials['access_token'] == sensitive_credentials['access_token']
        
        # Step 4: Log data access event
        access_event = AuditEvent(
            event_type=EventType.DATA_ACCESS,
            user_id="test_user",
            action="credentials_retrieved",
            resource="gmail_api",
            timestamp=datetime.now(),
            details={"success": True}
        )
        audit_logger.log_event(access_event)
        
        # Step 5: Verify audit trail exists
        log_files = list(temp_dir.glob("audit_*.log"))
        assert len(log_files) > 0
        
        log_content = log_files[0].read_text()
        assert "credentials_stored" in log_content
        assert "credentials_retrieved" in log_content
        
        # Step 6: Clean up
        await auth_manager.delete_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user"
        )
        
        # Verify cleanup worked
        after_cleanup = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user"
        )
        assert after_cleanup is None
    
    def test_production_security_validation(self):
        """Test that production environment enforces strong security"""
        # Production settings should enforce strong security
        with pytest.raises(ValueError, match="Master encryption key must be set"):
            Settings(
                environment=Environment.PRODUCTION,
                encryption__master_key=""  # Empty key should fail
            )
        
        with pytest.raises(ValueError, match="Audit logging must be enabled"):
            Settings(
                environment=Environment.PRODUCTION,
                encryption__master_key="secure_production_key_32_bytes_test",
                audit__audit_enabled=False  # Disabled audit should fail
            )
        
        with pytest.raises(ValueError, match="Session timeout too long"):
            Settings(
                environment=Environment.PRODUCTION,
                encryption__master_key="secure_production_key_32_bytes_test",
                security__session_timeout_minutes=300  # Too long should fail
            )
    
    def test_security_headers_generation(self, test_settings):
        """Test security headers are properly generated"""
        headers = test_settings.get_security_headers()
        
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Referrer-Policy'
        ]
        
        for header in required_headers:
            assert header in headers
            assert headers[header] is not None
        
        # Production should include CSP
        prod_settings = Settings(
            environment=Environment.PRODUCTION,
            encryption__master_key="production_key_32_bytes_for_test"
        )
        prod_headers = prod_settings.get_security_headers()
        assert 'Content-Security-Policy' in prod_headers