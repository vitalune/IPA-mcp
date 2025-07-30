"""
Configuration Package

This package contains comprehensive configuration management and security settings
for the Personal Knowledge Assistant MCP Server.

Features:
- Secure configuration management with environment-based settings
- OAuth2 token storage and management with encryption
- Privacy-compliant data handling and retention policies
- Audit logging and security monitoring
- Multi-service authentication state management

Security Implementation:
- All sensitive data encrypted at rest using AES-256-GCM
- PBKDF2/Argon2 key derivation for password security
- Comprehensive audit trails for compliance
- Privacy by design with configurable data retention
- Production-ready security controls and validation
"""

from .settings import (
    Settings,
    Environment,
    LogLevel,
    EncryptionSettings,
    SecuritySettings,
    PrivacySettings,
    DatabaseSettings,
    APICredentialSettings,
    AuditSettings,
    get_settings,
    reload_settings
)

from .auth import (
    AuthProvider,
    TokenType,
    AuthStatus,
    SecureToken,
    TokenMetadata,
    AuthenticationManager,
    get_auth_manager
)

__all__ = [
    # Settings classes
    "Settings",
    "Environment", 
    "LogLevel",
    "EncryptionSettings",
    "SecuritySettings",
    "PrivacySettings",
    "DatabaseSettings",
    "APICredentialSettings",
    "AuditSettings",
    
    # Settings functions
    "get_settings",
    "reload_settings",
    
    # Authentication classes
    "AuthProvider",
    "TokenType", 
    "AuthStatus",
    "SecureToken",
    "TokenMetadata",
    "AuthenticationManager",
    
    # Authentication functions
    "get_auth_manager"
]