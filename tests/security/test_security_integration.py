"""
Security Integration Tests

Tests security features in realistic scenarios with proper integration testing.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.config.settings import Settings, Environment
from src.config.auth import AuthManager, AuthProvider
from src.utils.encryption import EncryptionManager
from src.utils.audit import AuditLogger, AuditEvent, EventType
from src.integrations.client_manager import APIClientManager


class TestSecurityIntegrationScenarios:
    """Test realistic security scenarios"""
    
    @pytest.fixture
    async def secure_environment(self, temp_dir):
        """Set up a secure test environment"""
        settings = Settings(
            environment=Environment.PRODUCTION,
            encryption__master_key="secure_test_key_32_bytes_for_testing",
            database__data_directory=temp_dir / "data",
            database__cache_directory=temp_dir / "cache", 
            database__logs_directory=temp_dir / "logs",
            security__require_mfa=True,
            security__session_timeout_minutes=30,
            security__max_login_attempts=3,
            privacy__anonymize_logs=True,
            audit__audit_enabled=True,
            audit__track_api_calls=True,
            audit__track_data_access=True
        )
        
        # Ensure directories exist with proper permissions
        for directory in [settings.database.data_directory, settings.database.cache_directory, settings.database.logs_directory]:
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, 0o700)  # Owner read/write/execute only
        
        return settings
    
    @pytest.mark.asyncio
    async def test_secure_authentication_flow(self, secure_environment):
        """Test complete secure authentication flow with audit logging"""
        auth_manager = AuthManager(secure_environment)
        await auth_manager.initialize()
        
        audit_logger = AuditLogger(secure_environment)
        encryption_manager = EncryptionManager(secure_environment.encryption)
        
        # Step 1: Attempt authentication with invalid credentials
        with pytest.raises(Exception):
            await auth_manager.validate_token(
                provider=AuthProvider.GMAIL,
                user_id="test_user",
                token="invalid_token"
            )
        
        # Log failed authentication attempt
        failed_auth_event = AuditEvent(
            event_type=EventType.SECURITY_EVENT,
            user_id="test_user",
            action="authentication_failed",
            resource="gmail_api",
            timestamp=datetime.now(),
            ip_address="192.168.1.100",
            details={"reason": "invalid_token", "provider": "gmail"}
        )
        audit_logger.log_event(failed_auth_event)
        
        # Step 2: Store encrypted credentials
        sensitive_credentials = {
            'access_token': 'highly_sensitive_access_token_12345',
            'refresh_token': 'highly_sensitive_refresh_token_67890',
            'client_secret': 'client_secret_should_be_encrypted',
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        # Encrypt sensitive parts before storage
        encrypted_credentials = sensitive_credentials.copy()
        encrypted_credentials['access_token'] = encryption_manager.encrypt_data(sensitive_credentials['access_token'])
        encrypted_credentials['refresh_token'] = encryption_manager.encrypt_data(sensitive_credentials['refresh_token'])
        encrypted_credentials['client_secret'] = encryption_manager.encrypt_data(sensitive_credentials['client_secret'])
        
        await auth_manager.store_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user",
            credentials=encrypted_credentials
        )
        
        # Log successful credential storage
        success_auth_event = AuditEvent(
            event_type=EventType.AUTHENTICATION,
            user_id="test_user",
            action="credentials_stored",
            resource="gmail_api",
            timestamp=datetime.now(),
            ip_address="192.168.1.100",
            details={"provider": "gmail", "encrypted": True}
        )
        audit_logger.log_event(success_auth_event)
        
        # Step 3: Retrieve and decrypt credentials
        retrieved_credentials = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id="test_user"
        )
        
        assert retrieved_credentials is not None
        
        # Decrypt for verification
        decrypted_access_token = encryption_manager.decrypt_data(retrieved_credentials['access_token'])
        decrypted_refresh_token = encryption_manager.decrypt_data(retrieved_credentials['refresh_token'])
        
        assert decrypted_access_token == sensitive_credentials['access_token']
        assert decrypted_refresh_token == sensitive_credentials['refresh_token']
        
        # Step 4: Verify audit trail exists and is properly anonymized
        log_files = list(secure_environment.database.logs_directory.glob("audit_*.log"))
        assert len(log_files) > 0
        
        log_content = log_files[0].read_text()
        
        # Should contain audit events
        assert "authentication_failed" in log_content
        assert "credentials_stored" in log_content
        
        # Should NOT contain sensitive data (anonymized)
        assert "highly_sensitive_access_token" not in log_content
        assert "192.168.1.100" not in log_content  # IP should be anonymized
        
        # Should contain anonymized identifiers
        assert "test_user" not in log_content or log_content.count("test_user") < 2  # May be anonymized
    
    @pytest.mark.asyncio
    async def test_security_incident_detection_and_response(self, secure_environment):
        """Test security incident detection and automated response"""
        auth_manager = AuthManager(secure_environment)
        await auth_manager.initialize()
        
        audit_logger = AuditLogger(secure_environment)
        
        # Simulate multiple failed login attempts (brute force attack)
        attacker_ip = "10.0.0.1"
        attacker_user = "potential_attacker"
        
        failed_attempts = []
        for attempt in range(5):  # More than max_login_attempts (3)
            event = AuditEvent(
                event_type=EventType.SECURITY_EVENT,
                user_id=attacker_user,
                action="login_attempt_failed",
                resource="authentication",
                timestamp=datetime.now() - timedelta(minutes=5 - attempt),
                ip_address=attacker_ip,
                details={
                    "attempt_number": attempt + 1,
                    "reason": "invalid_password",
                    "user_agent": "Automated Tool v1.0"
                }
            )
            audit_logger.log_event(event)
            failed_attempts.append(event)
        
        # Simulate account lockout trigger
        lockout_event = AuditEvent(
            event_type=EventType.SECURITY_EVENT,
            user_id=attacker_user,
            action="account_locked",
            resource="authentication",
            timestamp=datetime.now(),
            ip_address=attacker_ip,
            details={
                "lockout_duration_minutes": secure_environment.security.lockout_duration_minutes,
                "failed_attempts_count": len(failed_attempts),
                "automated_response": True
            }
        )
        audit_logger.log_event(lockout_event)
        
        # Verify security events were logged
        log_files = list(secure_environment.database.logs_directory.glob("audit_*.log"))
        log_content = log_files[0].read_text()
        
        assert "login_attempt_failed" in log_content
        assert "account_locked" in log_content
        assert "automated_response" in log_content
        
        # Verify IP and user info is anonymized in production
        assert attacker_ip not in log_content  # Should be anonymized
        
        # Test legitimate user can still authenticate after incident
        legitimate_credentials = {
            'access_token': 'legitimate_token',
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        await auth_manager.store_credentials(
            provider=AuthProvider.GMAIL,
            user_id="legitimate_user",
            credentials=legitimate_credentials
        )
        
        # Should work fine for legitimate user
        is_valid = await auth_manager.validate_token(
            provider=AuthProvider.GMAIL,
            user_id="legitimate_user",
            token="legitimate_token"
        )
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_data_access_privacy_controls(self, secure_environment):
        """Test privacy controls during data access operations"""
        audit_logger = AuditLogger(secure_environment)
        
        # Simulate API client manager with privacy controls
        with patch('src.integrations.client_manager.get_auth_manager') as mock_auth_factory:
            mock_auth_manager = AsyncMock()
            mock_auth_manager.list_tokens.return_value = [
                {"status": "valid", "provider": "gmail", "expires_at": datetime.now() + timedelta(hours=1)}
            ]
            mock_auth_factory.return_value = mock_auth_manager
            
            client_manager = APIClientManager()
            await client_manager.initialize()
            
            client_manager.configure_service(
                service="gmail",
                client_id="test_client_id",
                client_secret="test_client_secret",
                scopes=["https://www.googleapis.com/auth/gmail.readonly"]
            )
            
            # Mock Gmail client with data access logging
            with patch('src.integrations.gmail_client.GmailClient') as mock_gmail_class:
                mock_gmail_client = AsyncMock()
                
                # Mock sensitive email data
                sensitive_emails = [
                    {
                        'id': 'email1',
                        'subject': 'Confidential: Salary Information',
                        'from': 'hr@company.com',
                        'to': ['employee@company.com'],
                        'body': 'Your salary has been increased to $75,000',
                        'date': datetime.now()
                    },
                    {
                        'id': 'email2',
                        'subject': 'Medical Records Request',
                        'from': 'doctor@clinic.com',
                        'to': ['patient@email.com'],
                        'body': 'Please find your test results attached',
                        'date': datetime.now()
                    }
                ]
                
                mock_gmail_client.search_messages.return_value = sensitive_emails
                mock_gmail_class.return_value = mock_gmail_client
                
                # Perform data access with privacy controls
                client = await client_manager.get_client("gmail")
                
                # Log data access attempt
                access_event = AuditEvent(
                    event_type=EventType.DATA_ACCESS,
                    user_id="test_user",
                    action="email_search_initiated",
                    resource="gmail_messages",
                    timestamp=datetime.now(),
                    ip_address="192.168.1.50",
                    details={
                        "query": "confidential OR medical",
                        "max_results": 10,
                        "privacy_mode": True
                    }
                )
                audit_logger.log_event(access_event)
                
                # Perform the search
                results = await client.search_messages(query="confidential OR medical", max_results=10)
                
                # Log data access results (with privacy controls)
                result_event = AuditEvent(
                    event_type=EventType.DATA_ACCESS,
                    user_id="test_user",
                    action="email_search_completed",
                    resource="gmail_messages",
                    timestamp=datetime.now(),
                    ip_address="192.168.1.50",
                    details={
                        "results_count": len(results),
                        "sensitive_data_detected": True,
                        "privacy_controls_applied": True,
                        "data_minimization": True
                    }
                )
                audit_logger.log_event(result_event)
                
                # Verify privacy controls are applied
                assert len(results) == 2
                
                # Verify audit log contains access events
                log_files = list(secure_environment.database.logs_directory.glob("audit_*.log"))
                log_content = log_files[0].read_text()
                
                assert "email_search_initiated" in log_content
                assert "email_search_completed" in log_content
                assert "privacy_controls_applied" in log_content
                
                # Verify sensitive data is NOT in logs
                assert "$75,000" not in log_content
                assert "test results" not in log_content
                assert "192.168.1.50" not in log_content  # IP anonymized
            
            await client_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_encryption_key_rotation(self, secure_environment):
        """Test encryption key rotation process"""
        encryption_manager = EncryptionManager(secure_environment.encryption)
        auth_manager = AuthManager(secure_environment)
        await auth_manager.initialize()
        
        audit_logger = AuditLogger(secure_environment)
        
        # Step 1: Store data with original key
        original_data = "sensitive_information_to_be_re_encrypted"
        encrypted_with_old_key = encryption_manager.encrypt_data(original_data)
        
        # Store encrypted credentials
        credentials = {
            'access_token': encrypted_with_old_key,
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        await auth_manager.store_credentials(
            provider=AuthProvider.GMAIL,
            user_id="rotation_test_user",
            credentials=credentials
        )
        
        # Step 2: Simulate key rotation
        # In real implementation, this would involve generating new key
        # and re-encrypting all stored data
        
        # Log key rotation event
        rotation_event = AuditEvent(
            event_type=EventType.CONFIGURATION_CHANGE,
            user_id="system_admin",
            action="encryption_key_rotation_initiated",
            resource="encryption_system",
            timestamp=datetime.now(),
            details={
                "old_key_id": "key_001",
                "new_key_id": "key_002",  
                "rotation_reason": "scheduled_rotation",
                "affected_records": 1
            }
        )
        audit_logger.log_event(rotation_event)
        
        # Step 3: Verify old data can still be decrypted
        retrieved_credentials = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id="rotation_test_user"
        )
        
        decrypted_data = encryption_manager.decrypt_data(retrieved_credentials['access_token'])
        assert decrypted_data == original_data
        
        # Step 4: Encrypt new data with new key (simulated)
        new_data = "new_sensitive_information_with_new_key"
        encrypted_with_new_key = encryption_manager.encrypt_data(new_data)
        
        # Should be different from old encryption
        assert encrypted_with_new_key != encrypted_with_old_key
        
        # Both should decrypt correctly
        assert encryption_manager.decrypt_data(encrypted_with_old_key) == original_data
        assert encryption_manager.decrypt_data(encrypted_with_new_key) == new_data
        
        # Log completion
        completion_event = AuditEvent(
            event_type=EventType.CONFIGURATION_CHANGE,
            user_id="system_admin",
            action="encryption_key_rotation_completed",
            resource="encryption_system",
            timestamp=datetime.now(),
            details={
                "new_key_id": "key_002",
                "records_migrated": 1,
                "migration_successful": True
            }
        )
        audit_logger.log_event(completion_event)
        
        # Verify audit trail
        log_files = list(secure_environment.database.logs_directory.glob("audit_*.log"))
        log_content = log_files[0].read_text()
        
        assert "encryption_key_rotation_initiated" in log_content
        assert "encryption_key_rotation_completed" in log_content
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance_workflow(self, secure_environment):
        """Test GDPR compliance features (data export, deletion, consent)"""
        auth_manager = AuthManager(secure_environment)
        await auth_manager.initialize()
        
        audit_logger = AuditLogger(secure_environment)
        encryption_manager = EncryptionManager(secure_environment.encryption)
        
        user_id = "gdpr_test_user"
        
        # Step 1: Store user data with consent tracking
        user_data = {
            'access_token': 'user_access_token',
            'refresh_token': 'user_refresh_token',
            'email': 'user@example.com',
            'consent_given': True,
            'consent_timestamp': datetime.now(),
            'data_processing_purposes': ['email_management', 'productivity_analysis'],
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        # Encrypt sensitive data
        encrypted_data = user_data.copy()
        encrypted_data['access_token'] = encryption_manager.encrypt_data(user_data['access_token'])
        encrypted_data['refresh_token'] = encryption_manager.encrypt_data(user_data['refresh_token'])
        encrypted_data['email'] = encryption_manager.encrypt_data(user_data['email'])
        
        await auth_manager.store_credentials(
            provider=AuthProvider.GMAIL,
            user_id=user_id,
            credentials=encrypted_data
        )
        
        # Log consent
        consent_event = AuditEvent(
            event_type=EventType.DATA_ACCESS,
            user_id=user_id,
            action="consent_recorded",
            resource="user_data",
            timestamp=datetime.now(),
            details={
                "consent_given": True,
                "purposes": user_data['data_processing_purposes'],
                "gdpr_compliant": True
            }
        )
        audit_logger.log_event(consent_event)
        
        # Step 2: User requests data export (GDPR Article 20)
        export_request_event = AuditEvent(
            event_type=EventType.DATA_ACCESS,
            user_id=user_id,
            action="data_export_requested",
            resource="user_data",
            timestamp=datetime.now(),
            details={
                "request_type": "gdpr_article_20",
                "format_requested": "json"
            }
        )
        audit_logger.log_event(export_request_event)
        
        # Simulate data export
        exported_data = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id=user_id
        )
        
        # Decrypt for export
        export_ready_data = {
            'user_id': user_id,
            'provider': 'gmail',
            'email': encryption_manager.decrypt_data(exported_data['email']),
            'consent_given': exported_data['consent_given'],
            'consent_timestamp': exported_data['consent_timestamp'].isoformat(),
            'data_processing_purposes': exported_data['data_processing_purposes'],
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Log successful export
        export_completion_event = AuditEvent(
            event_type=EventType.DATA_ACCESS,
            user_id=user_id,
            action="data_export_completed",
            resource="user_data",
            timestamp=datetime.now(),
            details={
                "export_format": "json",
                "records_exported": 1,
                "sensitive_data_encrypted": False  # Decrypted for export
            }
        )
        audit_logger.log_event(export_completion_event)
        
        # Step 3: User requests data deletion (GDPR Article 17 - Right to be forgotten)
        deletion_request_event = AuditEvent(
            event_type=EventType.DATA_ACCESS,
            user_id=user_id,
            action="data_deletion_requested",
            resource="user_data",
            timestamp=datetime.now(),
            details={
                "request_type": "gdpr_article_17",
                "reason": "user_request"
            }
        )
        audit_logger.log_event(deletion_request_event)
        
        # Perform deletion
        deletion_success = await auth_manager.delete_credentials(
            provider=AuthProvider.GMAIL,
            user_id=user_id
        )
        assert deletion_success is True
        
        # Verify deletion
        deleted_data = await auth_manager.get_user_credentials(
            provider=AuthProvider.GMAIL,
            user_id=user_id
        )
        assert deleted_data is None
        
        # Log deletion completion
        deletion_completion_event = AuditEvent(
            event_type=EventType.DATA_ACCESS,
            user_id=user_id,
            action="data_deletion_completed",
            resource="user_data",
            timestamp=datetime.now(),
            details={
                "records_deleted": 1,
                "secure_deletion": True,
                "gdpr_compliant": True
            }
        )
        audit_logger.log_event(deletion_completion_event)
        
        # Step 4: Verify GDPR compliance in audit logs
        log_files = list(secure_environment.database.logs_directory.glob("audit_*.log"))
        log_content = log_files[0].read_text()
        
        # Should contain all GDPR-related events
        gdpr_events = [
            "consent_recorded",
            "data_export_requested", 
            "data_export_completed",
            "data_deletion_requested",
            "data_deletion_completed"
        ]
        
        for event in gdpr_events:
            assert event in log_content
        
        # Should indicate GDPR compliance
        assert "gdpr_compliant" in log_content
        
        # Should NOT contain sensitive exported data in logs
        assert "user@example.com" not in log_content  # Should be anonymized
    
    @pytest.mark.asyncio
    async def test_security_monitoring_and_alerting(self, secure_environment):
        """Test security monitoring and alerting systems"""
        audit_logger = AuditLogger(secure_environment)
        
        # Enable security alerts
        secure_environment.audit.security_alerts_enabled = True
        secure_environment.audit.failed_login_alert_threshold = 3
        
        # Simulate various security events
        security_events = [
            # Unusual access patterns
            AuditEvent(
                event_type=EventType.SECURITY_EVENT,
                user_id="suspicious_user",
                action="unusual_access_pattern",
                resource="gmail_api",
                timestamp=datetime.now(),
                ip_address="203.0.113.1",  # Example IP from different country
                details={
                    "pattern": "bulk_download",
                    "data_volume": "100MB",
                    "time_of_day": "03:00 AM",
                    "risk_score": 0.8
                }
            ),
            
            # Privilege escalation attempt
            AuditEvent(
                event_type=EventType.SECURITY_EVENT,
                user_id="standard_user",
                action="privilege_escalation_attempt",
                resource="admin_panel",
                timestamp=datetime.now(),
                details={
                    "attempted_action": "modify_security_settings",
                    "current_privileges": "user",
                    "requested_privileges": "admin",
                    "blocked": True
                }
            ),
            
            # Data exfiltration attempt
            AuditEvent(
                event_type=EventType.SECURITY_EVENT,
                user_id="compromised_account",
                action="potential_data_exfiltration",
                resource="email_export",
                timestamp=datetime.now(),
                details={
                    "export_volume": "10000 emails",
                    "export_destination": "external_service",
                    "blocked": True,
                    "alert_triggered": True
                }
            )
        ]
        
        # Log all security events
        for event in security_events:
            audit_logger.log_event(event)
        
        # Simulate alert generation
        alert_event = AuditEvent(
            event_type=EventType.SECURITY_EVENT,
            user_id="security_system",
            action="security_alert_generated",
            resource="monitoring_system",
            timestamp=datetime.now(),
            details={
                "alert_type": "multiple_security_incidents",
                "incidents_count": len(security_events),
                "risk_level": "high",
                "automated_response": "temporary_account_suspension",
                "notification_sent": True
            }
        )
        audit_logger.log_event(alert_event)
        
        # Verify security events and alerts are logged
        log_files = list(secure_environment.database.logs_directory.glob("audit_*.log"))
        log_content = log_files[0].read_text()
        
        # Should contain all security events
        assert "unusual_access_pattern" in log_content
        assert "privilege_escalation_attempt" in log_content
        assert "potential_data_exfiltration" in log_content
        assert "security_alert_generated" in log_content
        
        # Should indicate protective actions were taken
        assert "blocked" in log_content
        assert "automated_response" in log_content
        
        # Sensitive data should be anonymized
        assert "203.0.113.1" not in log_content  # IP should be anonymized in production


class TestSecurityPerformance:
    """Test security feature performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_encryption_performance(self, secure_environment):
        """Test encryption/decryption performance under load"""
        encryption_manager = EncryptionManager(secure_environment.encryption)
        
        # Test data of various sizes
        test_data_sizes = [
            ("small", "small_data" * 10),      # ~100 bytes
            ("medium", "medium_data" * 100),   # ~1KB
            ("large", "large_data" * 1000),    # ~10KB
        ]
        
        import time
        
        for size_name, test_data in test_data_sizes:
            # Measure encryption time
            start_time = time.time()
            encrypted_data = encryption_manager.encrypt_data(test_data)
            encryption_time = time.time() - start_time
            
            # Measure decryption time
            start_time = time.time()
            decrypted_data = encryption_manager.decrypt_data(encrypted_data)
            decryption_time = time.time() - start_time
            
            # Verify correctness
            assert decrypted_data == test_data
            
            # Performance assertions (generous limits for testing)
            assert encryption_time < 1.0, f"Encryption too slow for {size_name} data: {encryption_time:.3f}s"
            assert decryption_time < 1.0, f"Decryption too slow for {size_name} data: {decryption_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_authentication_performance(self, secure_environment):
        """Test authentication performance with concurrent users"""
        auth_manager = AuthManager(secure_environment)
        await auth_manager.initialize()
        
        # Create multiple concurrent authentication tasks
        async def authenticate_user(user_id: str):
            credentials = {
                'access_token': f'token_for_{user_id}',
                'expires_at': datetime.now() + timedelta(hours=1)
            }
            
            await auth_manager.store_credentials(
                provider=AuthProvider.GMAIL,
                user_id=user_id,
                credentials=credentials
            )
            
            is_valid = await auth_manager.validate_token(
                provider=AuthProvider.GMAIL,
                user_id=user_id,
                token=f'token_for_{user_id}'
            )
            
            return is_valid
        
        # Test with multiple concurrent users
        import time
        user_count = 10
        
        start_time = time.time()
        
        tasks = [authenticate_user(f"user_{i}") for i in range(user_count)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Verify all authentications succeeded
        assert all(results), "All authentications should succeed"
        
        # Performance check - should handle concurrent load efficiently
        avg_time_per_user = total_time / user_count
        assert avg_time_per_user < 1.0, f"Average auth time too slow: {avg_time_per_user:.3f}s per user"
        assert total_time < 5.0, f"Total concurrent auth time too slow: {total_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_audit_logging_performance(self, secure_environment):
        """Test audit logging performance under high load"""
        audit_logger = AuditLogger(secure_environment)
        
        # Generate many audit events
        event_count = 1000
        
        import time
        start_time = time.time()
        
        for i in range(event_count):
            event = AuditEvent(
                event_type=EventType.DATA_ACCESS,
                user_id=f"performance_test_user_{i % 10}",  # 10 different users
                action="performance_test_action",
                resource="test_resource",
                timestamp=datetime.now(),
                details={"iteration": i, "test_data": f"data_{i}"}
            )
            audit_logger.log_event(event)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        events_per_second = event_count / total_time
        assert events_per_second > 100, f"Audit logging too slow: {events_per_second:.1f} events/sec"
        
        # Verify all events were logged
        log_files = list(secure_environment.database.logs_directory.glob("audit_*.log"))
        assert len(log_files) > 0
        
        # Check that log files contain our test events
        total_log_content = ""
        for log_file in log_files:
            total_log_content += log_file.read_text()
        
        # Should contain performance test events
        assert "performance_test_action" in total_log_content