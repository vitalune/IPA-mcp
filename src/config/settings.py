"""
Secure Configuration Management System

This module provides comprehensive configuration management with security best practices:
- Environment-based configuration loading
- Secure credential management
- Configuration validation and sanitization
- Privacy control settings
- Audit logging configuration
"""

import os
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from enum import Enum

from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic_settings import SettingsConfigDict


class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EncryptionSettings(BaseSettings):
    """Encryption and cryptographic settings"""
    
    # Master encryption key (should be 32 bytes base64 encoded)
    master_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        description="Master encryption key for data at rest"
    )
    
    # Key derivation settings
    pbkdf2_iterations: int = Field(
        default=100000,
        ge=50000,
        description="PBKDF2 iteration count for key derivation"
    )
    
    # Salt for key derivation (16 bytes base64 encoded)
    key_salt: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(16)),
        description="Salt for key derivation"
    )
    
    # Argon2 parameters for password hashing
    argon2_time_cost: int = Field(default=2, ge=1, description="Argon2 time cost parameter")
    argon2_memory_cost: int = Field(default=65536, ge=1024, description="Argon2 memory cost in KB")
    argon2_parallelism: int = Field(default=1, ge=1, description="Argon2 parallelism parameter")
    
    # Token settings
    token_expiry_hours: int = Field(default=24, ge=1, le=720, description="OAuth token expiry in hours")
    refresh_token_expiry_days: int = Field(default=30, ge=1, le=365, description="Refresh token expiry in days")


class SecuritySettings(BaseSettings):
    """Security and access control settings"""
    
    # Authentication settings
    require_mfa: bool = Field(default=False, description="Require multi-factor authentication")
    session_timeout_minutes: int = Field(default=60, ge=5, le=480, description="Session timeout in minutes")
    max_login_attempts: int = Field(default=5, ge=3, le=10, description="Maximum login attempts before lockout")
    lockout_duration_minutes: int = Field(default=15, ge=5, le=60, description="Account lockout duration")
    
    # Rate limiting
    api_rate_limit_per_minute: int = Field(default=60, ge=10, le=1000, description="API requests per minute")
    cache_rate_limit_per_hour: int = Field(default=500, ge=50, le=5000, description="Cache operations per hour")
    
    # CORS and network security
    allowed_origins: List[str] = Field(default=["http://localhost:3000"], description="Allowed CORS origins")
    trusted_proxies: List[str] = Field(default=[], description="Trusted proxy IP addresses")
    
    # Content security
    max_file_size_mb: int = Field(default=10, ge=1, le=100, description="Maximum file size in MB")
    allowed_file_types: List[str] = Field(
        default=[".txt", ".pdf", ".doc", ".docx", ".json"],
        description="Allowed file extensions"
    )


class PrivacySettings(BaseSettings):
    """Privacy and data protection settings"""
    
    # Data minimization
    collect_analytics: bool = Field(default=False, description="Enable analytics data collection")
    anonymize_logs: bool = Field(default=True, description="Anonymize IP addresses in logs")
    
    # Data retention
    cache_retention_days: int = Field(default=7, ge=1, le=365, description="Cache data retention in days")
    log_retention_days: int = Field(default=90, ge=30, le=730, description="Log retention in days")
    audit_retention_years: int = Field(default=2, ge=1, le=7, description="Audit log retention in years")
    
    # User consent and rights
    require_explicit_consent: bool = Field(default=True, description="Require explicit user consent")
    enable_data_export: bool = Field(default=True, description="Enable user data export")
    enable_data_deletion: bool = Field(default=True, description="Enable user data deletion")
    
    # Geographic restrictions
    allowed_countries: List[str] = Field(default=[], description="Allowed country codes (empty = all)")
    blocked_countries: List[str] = Field(default=[], description="Blocked country codes")
    
    # Data processing purposes
    processing_purposes: List[str] = Field(
        default=["personal_knowledge_management", "data_analysis", "search_optimization"],
        description="Legal purposes for data processing"
    )


class DatabaseSettings(BaseSettings):
    """Database and storage settings"""
    
    # Local storage paths
    data_directory: Path = Field(default=Path.home() / ".ipa_mcp", description="Data storage directory")
    cache_directory: Path = Field(default=Path.home() / ".ipa_mcp" / "cache", description="Cache directory")
    logs_directory: Path = Field(default=Path.home() / ".ipa_mcp" / "logs", description="Logs directory")
    
    # Database configuration
    encrypt_database: bool = Field(default=True, description="Encrypt local database files")
    database_backup_enabled: bool = Field(default=True, description="Enable automatic database backups")
    backup_retention_count: int = Field(default=5, ge=1, le=20, description="Number of backups to retain")
    
    # Cache settings
    max_cache_size_mb: int = Field(default=100, ge=10, le=1000, description="Maximum cache size in MB")
    cache_compression_enabled: bool = Field(default=True, description="Enable cache compression")


class APICredentialSettings(BaseSettings):
    """API credential management settings"""
    
    # OAuth2 settings
    oauth_redirect_uri: str = Field(default="http://localhost:8080/oauth/callback", description="OAuth redirect URI")
    oauth_state_expiry_minutes: int = Field(default=10, ge=5, le=30, description="OAuth state expiry in minutes")
    
    # Service-specific settings
    gmail_scopes: List[str] = Field(
        default=["https://www.googleapis.com/auth/gmail.readonly"],
        description="Gmail API scopes"
    )
    drive_scopes: List[str] = Field(
        default=["https://www.googleapis.com/auth/drive.readonly"],
        description="Google Drive API scopes"
    )
    
    # Credential storage
    encrypt_stored_tokens: bool = Field(default=True, description="Encrypt stored OAuth tokens")
    token_rotation_enabled: bool = Field(default=True, description="Enable automatic token rotation")


class AuditSettings(BaseSettings):
    """Audit logging and monitoring settings"""
    
    # Audit logging
    audit_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_file_rotation: bool = Field(default=True, description="Enable audit file rotation")
    audit_max_file_size_mb: int = Field(default=50, ge=10, le=500, description="Maximum audit file size")
    
    # Event tracking
    track_api_calls: bool = Field(default=True, description="Track API calls in audit logs")
    track_data_access: bool = Field(default=True, description="Track data access in audit logs")
    track_configuration_changes: bool = Field(default=True, description="Track config changes")
    
    # Alert settings
    security_alerts_enabled: bool = Field(default=True, description="Enable security alerts")
    failed_login_alert_threshold: int = Field(default=3, ge=1, le=10, description="Failed login alert threshold")
    
    # Log integrity
    audit_log_signing: bool = Field(default=True, description="Enable audit log digital signing")
    log_integrity_check_interval_hours: int = Field(default=24, ge=1, le=168, description="Log integrity check interval")


class Settings(BaseSettings):
    """Main application settings with comprehensive security configuration"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="forbid"  # Prevent unknown configuration parameters
    )
    
    # Core application settings
    app_name: str = Field(default="IPA-MCP-Server", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Logging configuration
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # Server settings
    host: str = Field(default="127.0.0.1", description="Server host address")
    port: int = Field(default=8080, ge=1024, le=65535, description="Server port")
    
    # Security and privacy configuration
    encryption: EncryptionSettings = Field(default_factory=EncryptionSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api_credentials: APICredentialSettings = Field(default_factory=APICredentialSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    
    @validator("environment", pre=True, always=True)
    def validate_environment(cls, v):
        """Validate and normalize environment setting"""
        if isinstance(v, str):
            v = v.lower()
            if v in ["dev", "develop"]:
                return Environment.DEVELOPMENT
            elif v in ["stage", "stag"]:
                return Environment.STAGING
            elif v in ["prod", "production"]:
                return Environment.PRODUCTION
        return v
    
    @validator("debug", always=True)
    def validate_debug_in_production(cls, v, values):
        """Ensure debug is disabled in production"""
        if values.get("environment") == Environment.PRODUCTION and v:
            raise ValueError("Debug mode cannot be enabled in production environment")
        return v
    
    def __init__(self, **kwargs):
        """Initialize settings with security validations"""
        super().__init__(**kwargs)
        self._ensure_directories_exist()
        self._validate_security_settings()
    
    def _ensure_directories_exist(self):
        """Ensure required directories exist with proper permissions"""
        directories = [
            self.database.data_directory,
            self.database.cache_directory,
            self.database.logs_directory,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions (owner read/write/execute only)
            os.chmod(directory, 0o700)
    
    def _validate_security_settings(self):
        """Validate security configuration consistency"""
        # Ensure production environment has strong security settings
        if self.environment == Environment.PRODUCTION:
            if not self.encryption.master_key.get_secret_value():
                raise ValueError("Master encryption key must be set in production")
            if not self.audit.audit_enabled:
                raise ValueError("Audit logging must be enabled in production")
            if self.security.session_timeout_minutes > 120:
                raise ValueError("Session timeout too long for production environment")
    
    def get_storage_path(self, filename: str) -> Path:
        """Get secure storage path for a file"""
        return self.database.data_directory / filename
    
    def get_cache_path(self, filename: str) -> Path:
        """Get cache storage path for a file"""
        return self.database.cache_directory / filename
    
    def get_log_path(self, filename: str) -> Path:
        """Get log storage path for a file"""
        return self.database.logs_directory / filename
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get HTTP security headers"""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }
        
        if self.is_production():
            headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https:; "
                "font-src 'self'; "
                "object-src 'none'; "
                "media-src 'self'; "
                "frame-src 'none';"
            )
        
        return headers


# Global settings instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton"""
    global settings
    if settings is None:
        settings = Settings()
    return settings


def reload_settings() -> Settings:
    """Reload settings from configuration"""
    global settings
    settings = Settings()
    return settings