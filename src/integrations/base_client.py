"""
Base API Client for External Service Integrations

This module provides the foundational API client class with:
- Common authentication handling
- Rate limiting and retry logic
- Error handling and logging
- Circuit breaker patterns
- Request/response encryption
- Comprehensive monitoring and auditing
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)

from ..config.auth import AuthProvider, get_auth_manager, TokenType
from ..config.settings import get_settings
from ..utils.encryption import get_encryption_manager

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class ClientStatus(str, Enum):
    """API client status states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"
    MAINTENANCE = "maintenance"


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window" 
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    backoff_multiplier: float = 2.0
    max_backoff_seconds: int = 300


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    timeout_seconds: int = 60
    success_threshold: int = 3
    enabled: bool = True


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    initial_wait: float = 1.0
    max_wait: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass 
class APIMetrics:
    """API client metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    circuit_breaker_trips: int = 0


class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from bucket"""
        async with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1) -> float:
        """Calculate wait time needed for tokens"""
        async with self._lock:
            if self.tokens >= tokens:
                return 0.0
            
            needed_tokens = tokens - self.tokens
            wait_time = needed_tokens / self.refill_rate
            return min(wait_time, 300.0)  # Max 5 minutes wait


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = ClientStatus.HEALTHY
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if not self.config.enabled:
            return await func(*args, **kwargs)
        
        async with self._lock:
            # Check if circuit should close
            if self.state == ClientStatus.CIRCUIT_OPEN:
                if (self.last_failure_time and 
                    datetime.now(timezone.utc) - self.last_failure_time > 
                    timedelta(seconds=self.config.timeout_seconds)):
                    self.state = ClientStatus.DEGRADED
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self):
        """Record successful request"""
        async with self._lock:
            if self.state == ClientStatus.DEGRADED:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = ClientStatus.HEALTHY
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == ClientStatus.HEALTHY:
                self.failure_count = 0
    
    async def _record_failure(self):
        """Record failed request"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = ClientStatus.CIRCUIT_OPEN
                self.success_count = 0


class BaseAPIClient(ABC, Generic[T]):
    """Base class for all API clients with common functionality"""
    
    def __init__(
        self,
        provider: AuthProvider,
        base_url: str,
        rate_limit_config: Optional[RateLimitConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        timeout: float = 30.0,
        encrypt_requests: bool = False,
        encrypt_responses: bool = False,
    ):
        self.provider = provider
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.encrypt_requests = encrypt_requests
        self.encrypt_responses = encrypt_responses
        
        # Configuration
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.retry_config = retry_config or RetryConfig()
        
        # Dependencies
        self.settings = get_settings()
        self.auth_manager = get_auth_manager()
        self.encryption_manager = get_encryption_manager()
        
        # Rate limiting
        self.rate_limiter = TokenBucket(
            capacity=self.rate_limit_config.burst_size,
            refill_rate=self.rate_limit_config.requests_per_minute / 60.0
        )
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_config)
        
        # Metrics
        self.metrics = APIMetrics()
        
        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        
        # Audit logger
        self._audit_logger = structlog.get_logger(f"api_audit.{provider.value}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self._get_default_headers(),
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20,
                    keepalive_expiry=30
                )
            )
    
    async def close(self):
        """Close HTTP client and cleanup resources"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default HTTP headers"""
        return {
            "User-Agent": f"{self.settings.app_name}/{self.settings.app_version}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    
    @abstractmethod
    async def authenticate(self, **kwargs) -> bool:
        """Authenticate with the API service"""
        pass
    
    @abstractmethod  
    async def refresh_token(self) -> bool:
        """Refresh authentication token"""
        pass
    
    @abstractmethod
    async def get_rate_limits(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        pass
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        # Find access token for this provider
        tokens = await self.auth_manager.list_tokens(
            provider=self.provider,
            token_type=TokenType.ACCESS_TOKEN
        )
        
        if not tokens:
            raise Exception(f"No access token found for {self.provider.value}")
        
        # Get the most recent valid token
        valid_tokens = [t for t in tokens if t["status"] == "valid"]
        if not valid_tokens:
            raise Exception(f"No valid access token found for {self.provider.value}")
        
        token_id = valid_tokens[0]["token_id"]
        token_value = await self.auth_manager.retrieve_token(token_id)
        
        if not token_value:
            raise Exception(f"Failed to retrieve access token for {self.provider.value}")
        
        return self._format_auth_header(token_value)
    
    @abstractmethod
    def _format_auth_header(self, token: str) -> Dict[str, str]:
        """Format authentication header for this provider"""
        pass
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        authenticated: bool = True,
        encrypt_request: Optional[bool] = None,
        encrypt_response: Optional[bool] = None,
    ) -> httpx.Response:
        """Make HTTP request with all protections"""
        
        await self._ensure_client()
        
        # Build full URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Prepare headers
        request_headers = self._get_default_headers()
        if headers:
            request_headers.update(headers)
        
        if authenticated:
            auth_headers = await self._get_auth_headers()
            request_headers.update(auth_headers)
        
        # Handle request encryption
        if encrypt_request or (encrypt_request is None and self.encrypt_requests):
            if data:
                encrypted_data = self.encryption_manager.encrypt_sensitive_data(
                    data, f"{self.provider.value}_request"
                )
                data = {"encrypted_payload": encrypted_data}
        
        # Rate limiting
        await self._enforce_rate_limit()
        
        # Make request with circuit breaker and retry logic
        response = await self.circuit_breaker.call(
            self._make_raw_request,
            method=method,
            url=url,
            json=data,
            params=params,
            headers=request_headers
        )
        
        # Handle response encryption
        if encrypt_response or (encrypt_response is None and self.encrypt_responses):
            # This would need to be implemented based on service requirements
            pass
        
        # Update metrics
        await self._update_metrics(response)
        
        # Audit logging
        await self._audit_log_request(method, endpoint, response.status_code)
        
        return response
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=before_sleep_log(logger, "WARNING")
    )
    async def _make_raw_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """Make raw HTTP request with retry logic"""
        
        start_time = time.time()
        
        try:
            response = await self._client.request(method, url, **kwargs)
            
            # Handle rate limiting responses
            if response.status_code == 429:
                retry_after = self._parse_retry_after(response)
                if retry_after:
                    await asyncio.sleep(retry_after)
                    self.metrics.rate_limited_requests += 1
                    raise httpx.HTTPStatusError(
                        f"Rate limited, retry after {retry_after}s",
                        request=response.request,
                        response=response
                    )
            
            # Handle authentication errors
            if response.status_code == 401:
                logger.warning("Authentication error, attempting token refresh")
                if await self.refresh_token():
                    # Retry with new token would happen via retry decorator
                    raise httpx.HTTPStatusError(
                        "Authentication failed",
                        request=response.request,
                        response=response
                    )
            
            response.raise_for_status()
            return response
            
        except Exception as e:
            logger.error(f"Request failed", method=method, url=url, error=str(e))
            raise
        finally:
            # Update response time metric
            response_time = time.time() - start_time
            if self.metrics.average_response_time == 0:
                self.metrics.average_response_time = response_time
            else:
                # Exponential moving average
                self.metrics.average_response_time = (
                    0.9 * self.metrics.average_response_time + 0.1 * response_time
                )
    
    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """Parse Retry-After header"""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                # Could be HTTP date format
                try:
                    from email.utils import parsedate_to_datetime
                    retry_time = parsedate_to_datetime(retry_after)
                    return (retry_time - datetime.now(timezone.utc)).total_seconds()
                except Exception:
                    pass
        
        # Check for X-RateLimit-Reset header (common in many APIs)
        rate_limit_reset = response.headers.get("X-RateLimit-Reset")
        if rate_limit_reset:
            try:
                reset_time = int(rate_limit_reset)
                return max(0, reset_time - int(time.time()))
            except ValueError:
                pass
        
        return None
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting"""
        if not await self.rate_limiter.consume():
            wait_time = await self.rate_limiter.wait_for_tokens()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                # Try again after waiting
                if not await self.rate_limiter.consume():
                    raise Exception("Rate limit exceeded even after waiting")
    
    async def _update_metrics(self, response: httpx.Response):
        """Update API metrics"""
        self.metrics.total_requests += 1
        self.metrics.last_request_time = datetime.now(timezone.utc)
        
        if response.status_code < 400:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now(timezone.utc)
        else:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now(timezone.utc)
    
    async def _audit_log_request(self, method: str, endpoint: str, status_code: int):
        """Log API request for audit trail"""
        if not self.settings.audit.track_api_calls:
            return
        
        audit_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider.value,
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "client_status": self.circuit_breaker.state.value,
        }
        
        self._audit_logger.info("API request", **audit_data)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get client health status"""
        return {
            "provider": self.provider.value,
            "status": self.circuit_breaker.state.value,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": (
                    self.metrics.successful_requests / max(1, self.metrics.total_requests)
                ),
                "average_response_time": self.metrics.average_response_time,
                "rate_limited_requests": self.metrics.rate_limited_requests,
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
                "last_request": self.metrics.last_request_time.isoformat() if self.metrics.last_request_time else None,
                "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            }
        }
    
    async def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        async with self.circuit_breaker._lock:
            self.circuit_breaker.state = ClientStatus.HEALTHY
            self.circuit_breaker.failure_count = 0
            self.circuit_breaker.success_count = 0
            self.circuit_breaker.last_failure_time = None
        
        logger.info(f"Circuit breaker reset for {self.provider.value}")
    
    async def update_rate_limits(self, **limits):
        """Update rate limiting configuration"""
        if "requests_per_minute" in limits:
            self.rate_limit_config.requests_per_minute = limits["requests_per_minute"]
            # Update token bucket refill rate
            self.rate_limiter.refill_rate = limits["requests_per_minute"] / 60.0
        
        if "burst_size" in limits:
            self.rate_limit_config.burst_size = limits["burst_size"]
            self.rate_limiter.capacity = limits["burst_size"]
        
        logger.info(f"Rate limits updated for {self.provider.value}", **limits)


class APIClientRegistry:
    """Registry for managing API client instances"""
    
    def __init__(self):
        self._clients: Dict[str, BaseAPIClient] = {}
        self._lock = asyncio.Lock()
    
    async def register_client(self, name: str, client: BaseAPIClient):
        """Register an API client"""
        async with self._lock:
            self._clients[name] = client
    
    async def get_client(self, name: str) -> Optional[BaseAPIClient]:
        """Get registered API client"""
        async with self._lock:
            return self._clients.get(name)
    
    async def remove_client(self, name: str):
        """Remove API client from registry"""
        async with self._lock:
            if name in self._clients:
                client = self._clients[name]
                await client.close()
                del self._clients[name]
    
    async def get_all_health_status(self) -> Dict[str, Any]:
        """Get health status for all registered clients"""
        status = {}
        async with self._lock:
            for name, client in self._clients.items():
                try:
                    status[name] = await client.get_health_status()
                except Exception as e:
                    status[name] = {
                        "provider": name,
                        "status": "error",
                        "error": str(e)
                    }
        return status
    
    async def close_all(self):
        """Close all registered clients"""
        async with self._lock:
            for client in self._clients.values():
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing client", error=str(e))
            self._clients.clear()


# Global client registry
client_registry = APIClientRegistry()


def get_client_registry() -> APIClientRegistry:
    """Get the global client registry"""
    return client_registry