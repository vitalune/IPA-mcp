"""
Secure Authentication Management System

This module provides comprehensive authentication handling with security best practices:
- OAuth2 token management with encryption
- API key secure storage and rotation
- Multi-service authentication state management
- Session management for MCP connections
- Token refresh and validation
"""

import asyncio
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, Field, SecretStr

from .settings import get_settings

logger = structlog.get_logger(__name__)


class AuthProvider(str, Enum):
    """Supported authentication providers"""
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    CUSTOM = "custom"


class TokenType(str, Enum):
    """Types of authentication tokens"""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    ID_TOKEN = "id_token"
    API_KEY = "api_key"
    SESSION_TOKEN = "session_token"


class AuthStatus(str, Enum):
    """Authentication status"""
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    INVALID = "invalid"
    PENDING_REFRESH = "pending_refresh"


@dataclass
class TokenMetadata:
    """Metadata for authentication tokens"""
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int
    scopes: List[str]
    issuer: str
    subject: Optional[str]
    client_id: Optional[str]


class SecureToken(BaseModel):
    """Secure token container with encryption"""
    
    token_type: TokenType
    provider: AuthProvider
    encrypted_value: bytes
    metadata: TokenMetadata
    checksum: str
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "token_type": self.token_type.value,
            "provider": self.provider.value,
            "encrypted_value": self.encrypted_value.hex(),
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "expires_at": self.metadata.expires_at.isoformat() if self.metadata.expires_at else None,
                "last_used": self.metadata.last_used.isoformat() if self.metadata.last_used else None,
                "usage_count": self.metadata.usage_count,
                "scopes": self.metadata.scopes,
                "issuer": self.metadata.issuer,
                "subject": self.metadata.subject,
                "client_id": self.metadata.client_id,
            },
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecureToken":
        """Create from dictionary"""
        metadata = TokenMetadata(
            created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
            expires_at=datetime.fromisoformat(data["metadata"]["expires_at"]) if data["metadata"]["expires_at"] else None,
            last_used=datetime.fromisoformat(data["metadata"]["last_used"]) if data["metadata"]["last_used"] else None,
            usage_count=data["metadata"]["usage_count"],
            scopes=data["metadata"]["scopes"],
            issuer=data["metadata"]["issuer"],
            subject=data["metadata"]["subject"],
            client_id=data["metadata"]["client_id"],
        )
        
        return cls(
            token_type=TokenType(data["token_type"]),
            provider=AuthProvider(data["provider"]),
            encrypted_value=bytes.fromhex(data["encrypted_value"]),
            metadata=metadata,
            checksum=data["checksum"],
        )


class AuthenticationManager:
    """Secure authentication manager for multiple providers and token types"""
    
    def __init__(self):
        self.settings = get_settings()
        self._encryption_key = self._derive_encryption_key()
        self._token_storage: Dict[str, SecureToken] = {}
        self._session_states: Dict[str, Dict[str, Any]] = {}
        self._load_tokens()
        
        # Audit logger for authentication events
        self._audit_logger = structlog.get_logger("auth_audit")
    
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from master key and salt"""
        master_key = self.settings.encryption.master_key.get_secret_value().encode()
        salt = self.settings.encryption.key_salt.get_secret_value().encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.settings.encryption.pbkdf2_iterations,
        )
        
        return kdf.derive(master_key)
    
    def _encrypt_token(self, token_value: str) -> Tuple[bytes, str]:
        """Encrypt token value and return encrypted data with checksum"""
        aesgcm = AESGCM(self._encryption_key)
        nonce = secrets.token_bytes(12)  # 12 bytes for GCM
        
        encrypted_data = aesgcm.encrypt(nonce, token_value.encode(), None)
        full_encrypted = nonce + encrypted_data
        
        # Create checksum for integrity validation
        checksum = hashes.Hash(hashes.SHA256())
        checksum.update(full_encrypted)
        checksum_value = checksum.finalize().hex()[:16]  # First 16 chars of SHA256
        
        return full_encrypted, checksum_value
    
    def _decrypt_token(self, encrypted_data: bytes, checksum: str) -> str:
        """Decrypt token value and validate checksum"""
        # Validate checksum
        calculated_checksum = hashes.Hash(hashes.SHA256())
        calculated_checksum.update(encrypted_data)
        calculated_checksum_value = calculated_checksum.finalize().hex()[:16]
        
        if calculated_checksum_value != checksum:
            raise ValueError("Token integrity check failed - possible tampering detected")
        
        aesgcm = AESGCM(self._encryption_key)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        decrypted_data = aesgcm.decrypt(nonce, ciphertext, None)
        return decrypted_data.decode()
    
    def _get_storage_path(self) -> Path:
        """Get secure storage path for tokens"""
        return self.settings.get_storage_path("auth_tokens.encrypted")
    
    def _load_tokens(self):
        """Load encrypted tokens from secure storage"""
        storage_path = self._get_storage_path()
        
        if not storage_path.exists():
            logger.info("No existing token storage found, starting fresh")
            return
        
        try:
            with open(storage_path, 'r') as f:
                stored_data = json.load(f)
            
            for token_id, token_data in stored_data.items():
                try:
                    secure_token = SecureToken.from_dict(token_data)
                    self._token_storage[token_id] = secure_token
                except Exception as e:
                    logger.error(f"Failed to load token {token_id}", error=str(e))
            
            logger.info(f"Loaded {len(self._token_storage)} tokens from secure storage")
            
        except Exception as e:
            logger.error("Failed to load token storage", error=str(e))
            # Don't fail completely, just start with empty storage
            self._token_storage = {}
    
    def _save_tokens(self):
        """Save encrypted tokens to secure storage"""
        storage_path = self._get_storage_path()
        
        # Ensure parent directory exists
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert all tokens to dict format
            storage_data = {
                token_id: token.to_dict()
                for token_id, token in self._token_storage.items()
            }
            
            # Write to temporary file first, then rename for atomic operation
            temp_path = storage_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(storage_data, f, indent=2)
            
            # Set restrictive permissions
            temp_path.chmod(0o600)
            
            # Atomic rename
            temp_path.rename(storage_path)
            
            logger.info("Tokens saved to secure storage")
            
        except Exception as e:
            logger.error("Failed to save tokens to storage", error=str(e))
            raise
    
    def _generate_token_id(self, provider: AuthProvider, token_type: TokenType, subject: Optional[str] = None) -> str:
        """Generate unique token identifier"""
        base_id = f"{provider.value}_{token_type.value}"
        if subject:
            base_id += f"_{subject}"
        
        # Add timestamp and random component for uniqueness
        timestamp = int(time.time())
        random_suffix = secrets.token_hex(4)
        
        return f"{base_id}_{timestamp}_{random_suffix}"
    
    async def store_token(
        self,
        provider: AuthProvider,
        token_type: TokenType,
        token_value: str,
        expires_in: Optional[int] = None,
        scopes: Optional[List[str]] = None,
        subject: Optional[str] = None,
        client_id: Optional[str] = None,
        issuer: Optional[str] = None,
    ) -> str:
        """Store a token securely with encryption"""
        
        # Encrypt the token value
        encrypted_value, checksum = self._encrypt_token(token_value)
        
        # Calculate expiration time
        expires_at = None
        if expires_in:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        
        # Create token metadata
        metadata = TokenMetadata(
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            last_used=None,
            usage_count=0,
            scopes=scopes or [],
            issuer=issuer or provider.value,
            subject=subject,
            client_id=client_id,
        )
        
        # Create secure token
        secure_token = SecureToken(
            token_type=token_type,
            provider=provider,
            encrypted_value=encrypted_value,
            metadata=metadata,
            checksum=checksum,
        )
        
        # Generate unique token ID
        token_id = self._generate_token_id(provider, token_type, subject)
        
        # Store token
        self._token_storage[token_id] = secure_token
        
        # Save to persistent storage
        self._save_tokens()
        
        # Audit log
        await self._audit_log("token_stored", {
            "token_id": token_id,
            "provider": provider.value,
            "token_type": token_type.value,
            "scopes": scopes or [],
            "expires_at": expires_at.isoformat() if expires_at else None,
        })
        
        logger.info(f"Token stored securely", token_id=token_id, provider=provider.value, token_type=token_type.value)
        
        return token_id
    
    async def retrieve_token(self, token_id: str) -> Optional[str]:
        """Retrieve and decrypt a token"""
        
        if token_id not in self._token_storage:
            logger.warning(f"Token not found", token_id=token_id)
            return None
        
        secure_token = self._token_storage[token_id]
        
        # Check if token is expired
        if secure_token.metadata.expires_at and secure_token.metadata.expires_at < datetime.now(timezone.utc):
            logger.warning(f"Token expired", token_id=token_id)
            await self._audit_log("token_access_expired", {"token_id": token_id})
            return None
        
        try:
            # Decrypt token
            token_value = self._decrypt_token(secure_token.encrypted_value, secure_token.checksum)
            
            # Update usage metadata
            secure_token.metadata.last_used = datetime.now(timezone.utc)
            secure_token.metadata.usage_count += 1
            
            # Save updated metadata
            self._save_tokens()
            
            # Audit log
            await self._audit_log("token_accessed", {
                "token_id": token_id,
                "provider": secure_token.provider.value,
                "token_type": secure_token.token_type.value,
                "usage_count": secure_token.metadata.usage_count,
            })
            
            return token_value
            
        except Exception as e:
            logger.error(f"Failed to decrypt token", token_id=token_id, error=str(e))
            await self._audit_log("token_access_failed", {
                "token_id": token_id,
                "error": str(e),
            })
            return None
    
    async def revoke_token(self, token_id: str) -> bool:
        """Revoke a token and remove from storage"""
        
        if token_id not in self._token_storage:
            logger.warning(f"Token not found for revocation", token_id=token_id)
            return False
        
        secure_token = self._token_storage[token_id]
        
        # Remove token from storage
        del self._token_storage[token_id]
        
        # Save changes
        self._save_tokens()
        
        # Audit log
        await self._audit_log("token_revoked", {
            "token_id": token_id,
            "provider": secure_token.provider.value,
            "token_type": secure_token.token_type.value,
        })
        
        logger.info(f"Token revoked", token_id=token_id)
        
        return True
    
    async def list_tokens(self, provider: Optional[AuthProvider] = None, token_type: Optional[TokenType] = None) -> List[Dict[str, Any]]:
        """List stored tokens with metadata (without revealing token values)"""
        
        tokens = []
        
        for token_id, secure_token in self._token_storage.items():
            # Filter by provider if specified
            if provider and secure_token.provider != provider:
                continue
            
            # Filter by token type if specified
            if token_type and secure_token.token_type != token_type:
                continue
            
            # Check token status
            status = AuthStatus.VALID
            if secure_token.metadata.expires_at:
                if secure_token.metadata.expires_at < datetime.now(timezone.utc):
                    status = AuthStatus.EXPIRED
            
            token_info = {
                "token_id": token_id,
                "provider": secure_token.provider.value,
                "token_type": secure_token.token_type.value,
                "status": status.value,
                "created_at": secure_token.metadata.created_at.isoformat(),
                "expires_at": secure_token.metadata.expires_at.isoformat() if secure_token.metadata.expires_at else None,
                "last_used": secure_token.metadata.last_used.isoformat() if secure_token.metadata.last_used else None,
                "usage_count": secure_token.metadata.usage_count,
                "scopes": secure_token.metadata.scopes,
                "subject": secure_token.metadata.subject,
            }
            
            tokens.append(token_info)
        
        return tokens
    
    async def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from storage"""
        
        expired_token_ids = []
        current_time = datetime.now(timezone.utc)
        
        for token_id, secure_token in self._token_storage.items():
            if secure_token.metadata.expires_at and secure_token.metadata.expires_at < current_time:
                expired_token_ids.append(token_id)
        
        # Remove expired tokens
        for token_id in expired_token_ids:
            del self._token_storage[token_id]
            logger.info(f"Removed expired token", token_id=token_id)
        
        if expired_token_ids:
            self._save_tokens()
            
            # Audit log
            await self._audit_log("tokens_cleanup", {
                "expired_count": len(expired_token_ids),
                "token_ids": expired_token_ids,
            })
        
        return len(expired_token_ids)
    
    def create_session_state(self, provider: AuthProvider, redirect_uri: str, scopes: List[str]) -> str:
        """Create secure session state for OAuth flow"""
        
        state = secrets.token_urlsafe(32)
        
        session_data = {
            "provider": provider.value,
            "redirect_uri": redirect_uri,
            "scopes": scopes,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=self.settings.api_credentials.oauth_state_expiry_minutes)).isoformat(),
        }
        
        self._session_states[state] = session_data
        
        logger.info(f"Created session state", state=state[:8] + "...", provider=provider.value)
        
        return state
    
    def validate_session_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Validate session state and return session data"""
        
        if state not in self._session_states:
            logger.warning(f"Invalid session state", state=state[:8] + "...")
            return None
        
        session_data = self._session_states[state]
        
        # Check expiration
        expires_at = datetime.fromisoformat(session_data["expires_at"])
        if expires_at < datetime.now(timezone.utc):
            logger.warning(f"Expired session state", state=state[:8] + "...")
            del self._session_states[state]
            return None
        
        return session_data
    
    def consume_session_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Consume (validate and remove) session state"""
        
        session_data = self.validate_session_state(state)
        
        if session_data:
            del self._session_states[state]
            logger.info(f"Consumed session state", state=state[:8] + "...")
        
        return session_data
    
    async def rotate_tokens(self) -> int:
        """Rotate tokens that support rotation"""
        
        rotated_count = 0
        
        for token_id, secure_token in list(self._token_storage.items()):
            # Only rotate tokens that are close to expiry (within 1 hour)
            if secure_token.metadata.expires_at:
                time_to_expiry = secure_token.metadata.expires_at - datetime.now(timezone.utc)
                
                if time_to_expiry < timedelta(hours=1) and secure_token.token_type == TokenType.ACCESS_TOKEN:
                    # Look for corresponding refresh token
                    refresh_token_id = self._find_refresh_token(secure_token.provider, secure_token.metadata.subject)
                    
                    if refresh_token_id:
                        logger.info(f"Token rotation needed", token_id=token_id)
                        # This would typically involve calling the OAuth provider's refresh endpoint
                        # For now, just log the need for rotation
                        rotated_count += 1
        
        if rotated_count > 0:
            await self._audit_log("tokens_rotation_check", {
                "tokens_needing_rotation": rotated_count,
            })
        
        return rotated_count
    
    def _find_refresh_token(self, provider: AuthProvider, subject: Optional[str]) -> Optional[str]:
        """Find corresponding refresh token for a provider and subject"""
        
        for token_id, secure_token in self._token_storage.items():
            if (secure_token.provider == provider and 
                secure_token.token_type == TokenType.REFRESH_TOKEN and
                secure_token.metadata.subject == subject):
                return token_id
        
        return None
    
    async def _audit_log(self, event_type: str, event_data: Dict[str, Any]):
        """Log authentication events for audit trail"""
        
        if not self.settings.audit.audit_enabled:
            return
        
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "component": "authentication_manager",
            "data": event_data,
        }
        
        self._audit_logger.info("Auth audit event", **audit_entry)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get authentication system health status"""
        
        total_tokens = len(self._token_storage)
        expired_tokens = 0
        expiring_soon = 0
        current_time = datetime.now(timezone.utc)
        
        for secure_token in self._token_storage.values():
            if secure_token.metadata.expires_at:
                if secure_token.metadata.expires_at < current_time:
                    expired_tokens += 1
                elif secure_token.metadata.expires_at < current_time + timedelta(hours=24):
                    expiring_soon += 1
        
        return {
            "total_tokens": total_tokens,
            "expired_tokens": expired_tokens,
            "expiring_soon": expiring_soon,
            "active_sessions": len(self._session_states),
            "storage_path": str(self._get_storage_path()),
            "encryption_enabled": True,
        }


# Global authentication manager instance
auth_manager: Optional[AuthenticationManager] = None


def get_auth_manager() -> AuthenticationManager:
    """Get authentication manager singleton"""
    global auth_manager
    if auth_manager is None:
        auth_manager = AuthenticationManager()
    return auth_manager