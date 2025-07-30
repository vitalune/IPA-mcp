"""
Utilities Package

This package contains shared utility functions and helpers used across the application.

Security and Privacy Utilities:
- Comprehensive encryption utilities with AES-256-GCM and ChaCha20-Poly1305
- Secure caching layer with encryption and audit trails
- Key derivation functions using PBKDF2, Scrypt, and Argon2
- File-based encrypted storage with integrity verification
- Secure random number generation and salt management
- Privacy-compliant data handling with configurable retention

Features:
- Production-ready encryption with multiple algorithms
- Secure cache with memory and file backends
- Automatic data cleanup and retention policies
- Comprehensive audit logging for all operations
- Thread-safe implementations with async support
"""

from .encryption import (
    EncryptionAlgorithm,
    KeyDerivationFunction,
    EncryptionMetadata,
    SecureRandom,
    KeyDerivation,
    SymmetricEncryption,
    AsymmetricEncryption,
    SecureStorage,
    EncryptionManager,
    get_encryption_manager
)

from .cache import (
    CacheBackend,
    CacheAccessType,
    CacheEntry,
    CacheAuditEntry,
    LRUCache,
    SecureCache,
    get_secure_cache
)

__all__ = [
    # Encryption classes and enums
    "EncryptionAlgorithm",
    "KeyDerivationFunction", 
    "EncryptionMetadata",
    "SecureRandom",
    "KeyDerivation",
    "SymmetricEncryption",
    "AsymmetricEncryption",
    "SecureStorage",
    "EncryptionManager",
    
    # Encryption functions
    "get_encryption_manager",
    
    # Cache classes and enums
    "CacheBackend",
    "CacheAccessType",
    "CacheEntry",
    "CacheAuditEntry", 
    "LRUCache",
    "SecureCache",
    
    # Cache functions
    "get_secure_cache"
]