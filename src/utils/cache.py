"""
Secure Caching Layer with Encryption and Audit Trail

This module provides comprehensive caching capabilities with security features:
- Encrypted cache storage with data integrity verification
- Configurable data retention policies and automatic cleanup
- Audit trails for all cache operations and data access
- Support for multiple cache backends (memory, file, hybrid)
- Cache invalidation mechanisms and consistency guarantees
- Privacy-compliant data handling with anonymization options
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, TypeVar, Generic
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import OrderedDict

import structlog

from ..config.settings import get_settings
from .encryption import get_encryption_manager, EncryptionManager

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class CacheBackend(str, Enum):
    """Supported cache backends"""
    MEMORY = "memory"
    FILE = "file"
    HYBRID = "hybrid"


class CacheAccessType(str, Enum):
    """Types of cache access operations"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    INVALIDATE = "invalidate"
    CLEANUP = "cleanup"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int
    last_accessed: datetime
    size_bytes: int
    tags: List[str]
    privacy_level: str
    checksum: str


@dataclass
class CacheAuditEntry:
    """Audit entry for cache operations"""
    timestamp: datetime
    operation: CacheAccessType
    cache_key: str
    user_context: Optional[str]
    ip_address: Optional[str]
    success: bool
    error_message: Optional[str]
    data_size: Optional[int]
    privacy_level: Optional[str]


class LRUCache(Generic[T]):
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache"""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: T) -> None:
        """Put item in cache"""
        with self._lock:
            if key in self._cache:
                # Update existing item
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # Remove least recently used item
                self._cache.popitem(last=False)
            
            self._cache[key] = value
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache"""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self._cache)
    
    def keys(self) -> List[str]:
        """Get all cache keys"""
        with self._lock:
            return list(self._cache.keys())


class SecureCache:
    """Secure cache with encryption, audit trails, and privacy controls"""
    
    def __init__(
        self,
        backend: CacheBackend = CacheBackend.HYBRID,
        max_memory_entries: int = 1000,
        max_file_size_mb: int = 100,
        encryption_enabled: bool = True
    ):
        self.settings = get_settings()
        self.backend = backend
        self.encryption_enabled = encryption_enabled
        self.encryption_manager = get_encryption_manager() if encryption_enabled else None
        
        # Initialize backends
        self._memory_cache: LRUCache[CacheEntry] = LRUCache(max_memory_entries)
        self._file_cache_dir = self.settings.get_cache_path("secure_cache")
        self._file_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata and statistics
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "deletes": 0,
            "cleanups": 0,
            "total_size_bytes": 0,
        }
        
        # Audit logging
        self._audit_logger = structlog.get_logger("cache_audit")
        self._audit_entries: List[CacheAuditEntry] = []
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background cleanup task"""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                self._cleanup_task = loop.create_task(self._background_cleanup())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            logger.info("No event loop available, background cleanup disabled")
    
    async def _background_cleanup(self):
        """Background task for cache cleanup and maintenance"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_expired()
                await self._rotate_audit_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Background cleanup failed", error=str(e))
    
    def _calculate_entry_size(self, value: Any) -> int:
        """Calculate size of cache entry in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value.encode() if isinstance(value, str) else value)
            else:
                return len(json.dumps(value, ensure_ascii=False).encode())
        except Exception:
            return 0
    
    def _generate_cache_key_hash(self, key: str) -> str:
        """Generate hash of cache key for file storage"""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data integrity"""
        try:
            if isinstance(data, bytes):
                content = data
            elif isinstance(data, str):
                content = data.encode()
            else:
                content = json.dumps(data, ensure_ascii=False, sort_keys=True).encode()
            
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return "invalid"
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        key_hash = self._generate_cache_key_hash(key)
        return self._file_cache_dir / f"{key_hash}.cache"
    
    async def _audit_log(
        self,
        operation: CacheAccessType,
        cache_key: str,
        success: bool,
        user_context: Optional[str] = None,
        ip_address: Optional[str] = None,
        error_message: Optional[str] = None,
        data_size: Optional[int] = None,
        privacy_level: Optional[str] = None
    ):
        """Log cache operation for audit trail"""
        
        if not self.settings.audit.audit_enabled:
            return
        
        # Anonymize IP address if required
        if ip_address and self.settings.privacy.anonymize_logs:
            ip_parts = ip_address.split('.')
            if len(ip_parts) == 4:
                ip_address = f"{ip_parts[0]}.{ip_parts[1]}.xxx.xxx"
        
        audit_entry = CacheAuditEntry(
            timestamp=datetime.now(timezone.utc),
            operation=operation,
            cache_key=cache_key[:50] + "..." if len(cache_key) > 50 else cache_key,  # Truncate long keys
            user_context=user_context,
            ip_address=ip_address,
            success=success,
            error_message=error_message,
            data_size=data_size,
            privacy_level=privacy_level
        )
        
        self._audit_entries.append(audit_entry)
        
        # Log to structured logger
        self._audit_logger.info(
            "Cache operation",
            operation=operation.value,
            cache_key=audit_entry.cache_key,
            success=success,
            data_size=data_size,
            privacy_level=privacy_level
        )
        
        # Rotate audit logs if they get too large
        if len(self._audit_entries) > 10000:
            await self._rotate_audit_logs()
    
    async def _rotate_audit_logs(self):
        """Rotate and save audit logs"""
        if not self._audit_entries:
            return
        
        try:
            # Save current audit entries to file
            audit_file = self.settings.get_log_path(f"cache_audit_{int(time.time())}.json")
            
            audit_data = [asdict(entry) for entry in self._audit_entries]
            
            if self.encryption_enabled and self.encryption_manager:
                # Encrypt audit logs
                success = self.encryption_manager.secure_storage.encrypt_file(audit_file, audit_data)
                if not success:
                    logger.error("Failed to encrypt audit logs")
            else:
                # Save unencrypted
                with open(audit_file, 'w') as f:
                    json.dump(audit_data, f, indent=2, default=str)
            
            # Clear in-memory audit entries
            self._audit_entries.clear()
            
            logger.info(f"Audit logs rotated", file=str(audit_file), entries=len(audit_data))
            
        except Exception as e:
            logger.error("Failed to rotate audit logs", error=str(e))
    
    async def get(
        self,
        key: str,
        user_context: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Optional[Any]:
        """Get value from cache"""
        
        start_time = time.time()
        
        try:
            # Try memory cache first for hybrid backend
            if self.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
                entry = self._memory_cache.get(key)
                if entry:
                    # Check expiration
                    if entry.expires_at and entry.expires_at < datetime.now(timezone.utc):
                        self._memory_cache.remove(key)
                        await self._audit_log(CacheAccessType.READ, key, False, user_context, ip_address, "Expired entry")
                        self._cache_stats["misses"] += 1
                        return None
                    
                    # Update access metadata
                    entry.access_count += 1
                    entry.last_accessed = datetime.now(timezone.utc)
                    
                    self._cache_stats["hits"] += 1
                    await self._audit_log(CacheAccessType.READ, key, True, user_context, ip_address, 
                                        data_size=entry.size_bytes, privacy_level=entry.privacy_level)
                    return entry.value
            
            # Try file cache for file or hybrid backend
            if self.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
                file_path = self._get_file_path(key)
                
                if file_path.exists():
                    try:
                        if self.encryption_enabled and self.encryption_manager:
                            # Decrypt file
                            entry_data = self.encryption_manager.secure_storage.decrypt_file(file_path, return_dict=True)
                        else:
                            # Load unencrypted
                            with open(file_path, 'r') as f:
                                entry_data = json.load(f)
                        
                        if not entry_data:
                            raise ValueError("Failed to load cache entry")
                        
                        # Reconstruct cache entry
                        entry = CacheEntry(
                            key=entry_data["key"],
                            value=entry_data["value"],
                            created_at=datetime.fromisoformat(entry_data["created_at"]),
                            expires_at=datetime.fromisoformat(entry_data["expires_at"]) if entry_data["expires_at"] else None,
                            access_count=entry_data["access_count"],
                            last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                            size_bytes=entry_data["size_bytes"],
                            tags=entry_data["tags"],
                            privacy_level=entry_data["privacy_level"],
                            checksum=entry_data["checksum"]
                        )
                        
                        # Verify checksum
                        calculated_checksum = self._calculate_checksum(entry.value)
                        if calculated_checksum != entry.checksum:
                            logger.warning(f"Cache entry checksum mismatch", key=key)
                            file_path.unlink()  # Remove corrupted entry
                            await self._audit_log(CacheAccessType.READ, key, False, user_context, ip_address, "Checksum mismatch")
                            self._cache_stats["misses"] += 1
                            return None
                        
                        # Check expiration
                        if entry.expires_at and entry.expires_at < datetime.now(timezone.utc):
                            file_path.unlink()  # Remove expired entry
                            await self._audit_log(CacheAccessType.READ, key, False, user_context, ip_address, "Expired entry")
                            self._cache_stats["misses"] += 1
                            return None
                        
                        # Update access metadata and save back
                        entry.access_count += 1
                        entry.last_accessed = datetime.now(timezone.utc)
                        
                        # Also add to memory cache if using hybrid backend
                        if self.backend == CacheBackend.HYBRID:
                            self._memory_cache.put(key, entry)
                        
                        # Update file
                        await self._save_file_entry(key, entry)
                        
                        self._cache_stats["hits"] += 1
                        await self._audit_log(CacheAccessType.READ, key, True, user_context, ip_address,
                                            data_size=entry.size_bytes, privacy_level=entry.privacy_level)
                        return entry.value
                        
                    except Exception as e:
                        logger.error(f"Failed to load cache entry from file", key=key, error=str(e))
                        # Remove corrupted file
                        if file_path.exists():
                            file_path.unlink()
            
            # Cache miss
            self._cache_stats["misses"] += 1
            await self._audit_log(CacheAccessType.READ, key, False, user_context, ip_address, "Cache miss")
            return None
            
        except Exception as e:
            logger.error(f"Cache get operation failed", key=key, error=str(e))
            await self._audit_log(CacheAccessType.READ, key, False, user_context, ip_address, str(e))
            self._cache_stats["misses"] += 1
            return None
    
    async def _save_file_entry(self, key: str, entry: CacheEntry):
        """Save cache entry to file"""
        file_path = self._get_file_path(key)
        
        entry_data = {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at.isoformat(),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed.isoformat(),
            "size_bytes": entry.size_bytes,
            "tags": entry.tags,
            "privacy_level": entry.privacy_level,
            "checksum": entry.checksum
        }
        
        if self.encryption_enabled and self.encryption_manager:
            # Encrypt and save
            success = self.encryption_manager.secure_storage.encrypt_file(file_path, entry_data)
            if not success:
                raise ValueError("Failed to encrypt cache entry")
        else:
            # Save unencrypted
            with open(file_path, 'w') as f:
                json.dump(entry_data, f, indent=2, default=str)
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        privacy_level: str = "standard",
        user_context: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """Put value in cache"""
        
        try:
            # Calculate expiration time
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            elif self.settings.privacy.cache_retention_days > 0:
                expires_at = datetime.now(timezone.utc) + timedelta(days=self.settings.privacy.cache_retention_days)
            
            # Calculate entry size and checksum
            size_bytes = self._calculate_entry_size(value)
            checksum = self._calculate_checksum(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(timezone.utc),
                expires_at=expires_at,
                access_count=0,
                last_accessed=datetime.now(timezone.utc),
                size_bytes=size_bytes,
                tags=tags or [],
                privacy_level=privacy_level,
                checksum=checksum
            )
            
            # Store in memory cache
            if self.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
                self._memory_cache.put(key, entry)
            
            # Store in file cache
            if self.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
                await self._save_file_entry(key, entry)
            
            # Update statistics
            self._cache_stats["writes"] += 1
            self._cache_stats["total_size_bytes"] += size_bytes
            
            await self._audit_log(CacheAccessType.WRITE, key, True, user_context, ip_address,
                                data_size=size_bytes, privacy_level=privacy_level)
            
            logger.debug(f"Cache entry stored", key=key, size=size_bytes, expires_at=expires_at)
            return True
            
        except Exception as e:
            logger.error(f"Cache put operation failed", key=key, error=str(e))
            await self._audit_log(CacheAccessType.WRITE, key, False, user_context, ip_address, str(e))
            return False
    
    async def delete(
        self,
        key: str,
        user_context: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """Delete value from cache"""
        
        try:
            deleted = False
            
            # Remove from memory cache
            if self.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
                if self._memory_cache.remove(key):
                    deleted = True
            
            # Remove from file cache
            if self.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
                file_path = self._get_file_path(key)
                if file_path.exists():
                    if self.encryption_enabled and self.encryption_manager:
                        # Secure delete
                        self.encryption_manager.secure_delete_file(file_path)
                    else:
                        file_path.unlink()
                    deleted = True
            
            if deleted:
                self._cache_stats["deletes"] += 1
                await self._audit_log(CacheAccessType.DELETE, key, True, user_context, ip_address)
            else:
                await self._audit_log(CacheAccessType.DELETE, key, False, user_context, ip_address, "Key not found")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete operation failed", key=key, error=str(e))
            await self._audit_log(CacheAccessType.DELETE, key, False, user_context, ip_address, str(e))
            return False
    
    async def invalidate_by_tags(
        self,
        tags: List[str],
        user_context: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> int:
        """Invalidate cache entries by tags"""
        
        invalidated_count = 0
        
        try:
            # Get all keys to check
            keys_to_check = []
            
            # Check memory cache
            if self.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
                keys_to_check.extend(self._memory_cache.keys())
            
            # Check file cache
            if self.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
                for file_path in self._file_cache_dir.glob("*.cache"):
                    key_hash = file_path.stem
                    # We can't easily reverse the hash, so we'll need to load each file
                    # This is not efficient for large caches - consider indexing by tags
                    keys_to_check.append(key_hash)
            
            # Check each entry for matching tags
            for key in keys_to_check:
                try:
                    # Get entry to check tags
                    entry = None
                    
                    if self.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
                        entry = self._memory_cache.get(key)
                    
                    if not entry and self.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
                        # Try to load from file
                        file_path = self._get_file_path(key) if len(key) > 50 else self._file_cache_dir / f"{key}.cache"
                        if file_path.exists():
                            try:
                                if self.encryption_enabled and self.encryption_manager:
                                    entry_data = self.encryption_manager.secure_storage.decrypt_file(file_path, return_dict=True)
                                else:
                                    with open(file_path, 'r') as f:
                                        entry_data = json.load(f)
                                
                                if entry_data:
                                    entry = CacheEntry(
                                        key=entry_data["key"],
                                        value=entry_data["value"],
                                        created_at=datetime.fromisoformat(entry_data["created_at"]),
                                        expires_at=datetime.fromisoformat(entry_data["expires_at"]) if entry_data["expires_at"] else None,
                                        access_count=entry_data["access_count"],
                                        last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                                        size_bytes=entry_data["size_bytes"],
                                        tags=entry_data["tags"],
                                        privacy_level=entry_data["privacy_level"],
                                        checksum=entry_data["checksum"]
                                    )
                            except Exception:
                                continue
                    
                    # Check if entry has any of the specified tags
                    if entry and any(tag in entry.tags for tag in tags):
                        await self.delete(entry.key, user_context, ip_address)
                        invalidated_count += 1
                        
                except Exception as e:
                    logger.error(f"Error checking cache entry for tag invalidation", key=key, error=str(e))
                    continue
            
            await self._audit_log(CacheAccessType.INVALIDATE, f"tags:{','.join(tags)}", True, 
                                user_context, ip_address, data_size=invalidated_count)
            
            logger.info(f"Invalidated cache entries by tags", tags=tags, count=invalidated_count)
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Cache invalidation by tags failed", tags=tags, error=str(e))
            await self._audit_log(CacheAccessType.INVALIDATE, f"tags:{','.join(tags)}", False, 
                                user_context, ip_address, error_message=str(e))
            return 0
    
    async def cleanup_expired(
        self,
        user_context: Optional[str] = None
    ) -> int:
        """Clean up expired cache entries"""
        
        cleaned_count = 0
        current_time = datetime.now(timezone.utc)
        
        try:
            # Clean memory cache
            if self.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
                keys_to_remove = []
                for key in self._memory_cache.keys():
                    entry = self._memory_cache.get(key)
                    if entry and entry.expires_at and entry.expires_at < current_time:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self._memory_cache.remove(key)
                    cleaned_count += 1
            
            # Clean file cache
            if self.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
                for file_path in self._file_cache_dir.glob("*.cache"):
                    try:
                        if self.encryption_enabled and self.encryption_manager:
                            entry_data = self.encryption_manager.secure_storage.decrypt_file(file_path, return_dict=True)
                        else:
                            with open(file_path, 'r') as f:
                                entry_data = json.load(f)
                        
                        if entry_data and entry_data.get("expires_at"):
                            expires_at = datetime.fromisoformat(entry_data["expires_at"])
                            if expires_at < current_time:
                                if self.encryption_enabled and self.encryption_manager:
                                    self.encryption_manager.secure_delete_file(file_path)
                                else:
                                    file_path.unlink()
                                cleaned_count += 1
                                
                    except Exception as e:
                        logger.error(f"Error cleaning expired cache file", file=str(file_path), error=str(e))
                        # Remove corrupted files
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except Exception:
                            pass
            
            self._cache_stats["cleanups"] += 1
            
            await self._audit_log(CacheAccessType.CLEANUP, "expired_entries", True, 
                                user_context, data_size=cleaned_count)
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up expired cache entries", count=cleaned_count)
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Cache cleanup failed", error=str(e))
            await self._audit_log(CacheAccessType.CLEANUP, "expired_entries", False, 
                                user_context, error_message=str(e))
            return 0
    
    async def clear_all(
        self,
        user_context: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """Clear all cache entries"""
        
        try:
            # Clear memory cache
            if self.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
                self._memory_cache.clear()
            
            # Clear file cache
            if self.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
                for file_path in self._file_cache_dir.glob("*.cache"):
                    if self.encryption_enabled and self.encryption_manager:
                        self.encryption_manager.secure_delete_file(file_path)
                    else:
                        file_path.unlink()
            
            # Reset statistics
            self._cache_stats = {
                "hits": 0,
                "misses": 0,
                "writes": 0,
                "deletes": 0,
                "cleanups": 0,
                "total_size_bytes": 0,
            }
            
            await self._audit_log(CacheAccessType.DELETE, "all_entries", True, user_context, ip_address)
            
            logger.info("All cache entries cleared")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear all failed", error=str(e))
            await self._audit_log(CacheAccessType.DELETE, "all_entries", False, user_context, ip_address, str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_size = self._memory_cache.size() if self.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID] else 0
        
        file_count = 0
        if self.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
            file_count = len(list(self._file_cache_dir.glob("*.cache")))
        
        total_entries = memory_size + (file_count if self.backend == CacheBackend.FILE else 0)
        
        hit_rate = 0
        if self._cache_stats["hits"] + self._cache_stats["misses"] > 0:
            hit_rate = self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"])
        
        return {
            "backend": self.backend.value,
            "total_entries": total_entries,
            "memory_entries": memory_size,
            "file_entries": file_count,
            "hit_rate": hit_rate,
            "encryption_enabled": self.encryption_enabled,
            **self._cache_stats
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get cache health status"""
        stats = self.get_stats()
        
        # Check for issues
        issues = []
        if stats["hit_rate"] < 0.5 and stats["hits"] + stats["misses"] > 100:
            issues.append("Low cache hit rate")
        
        if self.backend in [CacheBackend.FILE, CacheBackend.HYBRID]:
            if not self._file_cache_dir.exists():
                issues.append("Cache directory not accessible")
        
        return {
            "status": "healthy" if not issues else "warning",
            "issues": issues,
            "stats": stats,
            "cache_directory": str(self._file_cache_dir) if self.backend in [CacheBackend.FILE, CacheBackend.HYBRID] else None
        }
    
    async def shutdown(self):
        """Shutdown cache and cleanup resources"""
        try:
            # Cancel background cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Save audit logs
            await self._rotate_audit_logs()
            
            logger.info("Cache shutdown completed")
            
        except Exception as e:
            logger.error(f"Cache shutdown failed", error=str(e))


# Global cache instance
secure_cache: Optional[SecureCache] = None


def get_secure_cache() -> SecureCache:
    """Get secure cache singleton"""
    global secure_cache
    if secure_cache is None:
        settings = get_settings()
        secure_cache = SecureCache(
            backend=CacheBackend.HYBRID,
            max_memory_entries=1000,
            max_file_size_mb=settings.database.max_cache_size_mb,
            encryption_enabled=settings.database.encrypt_database
        )
    return secure_cache