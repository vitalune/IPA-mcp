"""
Advanced Rate Limiting System

This module provides sophisticated rate limiting capabilities:
- Multiple rate limiting algorithms (Token Bucket, Sliding Window, Fixed Window)
- Per-service and per-endpoint rate limiting
- Distributed rate limiting support
- Adaptive rate limiting based on API responses
- Rate limit status monitoring and alerting
- Thread-safe and async-compatible implementation
"""

import asyncio
import json
import time
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict

import structlog
from ..config.settings import get_settings

logger = structlog.get_logger(__name__)


class RateLimitAlgorithm(str, Enum):
    """Supported rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


class RateLimitScope(str, Enum):
    """Rate limit scope levels"""
    GLOBAL = "global"
    SERVICE = "service"
    ENDPOINT = "endpoint"
    USER = "user"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    requests: int
    period_seconds: int
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    burst_multiplier: float = 1.5
    scope: RateLimitScope = RateLimitScope.SERVICE
    enabled: bool = True
    
    @property
    def burst_size(self) -> int:
        """Calculate burst size based on multiplier"""
        return int(self.requests * self.burst_multiplier)
    
    @property
    def refill_rate(self) -> float:
        """Calculate token refill rate per second"""
        return self.requests / self.period_seconds


@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[float] = None
    current_usage: int = 0
    rule: Optional[RateLimitRule] = None


@dataclass
class AdaptiveSettings:
    """Settings for adaptive rate limiting"""
    base_rate: int = 60
    min_rate: int = 10
    max_rate: int = 1000
    increase_factor: float = 1.1
    decrease_factor: float = 0.9
    success_threshold: int = 10
    error_threshold: int = 3
    adaptation_window_seconds: int = 300


class RateLimiterBase(ABC):
    """Abstract base class for rate limiters"""
    
    def __init__(self, rule: RateLimitRule, identifier: str):
        self.rule = rule
        self.identifier = identifier
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def is_allowed(self, tokens: int = 1) -> RateLimitStatus:
        """Check if request is allowed"""
        pass
    
    @abstractmethod
    async def reset(self):
        """Reset rate limiter state"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        pass


class TokenBucketLimiter(RateLimiterBase):
    """Token bucket rate limiter implementation"""
    
    def __init__(self, rule: RateLimitRule, identifier: str):
        super().__init__(rule, identifier)
        self.capacity = rule.burst_size
        self.tokens = self.capacity
        self.refill_rate = rule.refill_rate
        self.last_refill = time.time()
    
    async def is_allowed(self, tokens: int = 1) -> RateLimitStatus:
        """Check if tokens are available"""
        if not self.rule.enabled:
            return RateLimitStatus(
                allowed=True,
                remaining=float('inf'),
                reset_time=datetime.now(timezone.utc),
                rule=self.rule
            )
        
        async with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return RateLimitStatus(
                    allowed=True,
                    remaining=int(self.tokens),
                    reset_time=self._calculate_reset_time(),
                    current_usage=self.capacity - int(self.tokens),
                    rule=self.rule
                )
            else:
                # Calculate retry after time
                needed_tokens = tokens - self.tokens
                retry_after = needed_tokens / self.refill_rate
                
                return RateLimitStatus(
                    allowed=False,
                    remaining=int(self.tokens),
                    reset_time=self._calculate_reset_time(),
                    retry_after=retry_after,
                    current_usage=self.capacity,
                    rule=self.rule
                )
    
    def _calculate_reset_time(self) -> datetime:
        """Calculate when bucket will be full again"""
        if self.tokens >= self.capacity:
            return datetime.now(timezone.utc)
        
        tokens_needed = self.capacity - self.tokens
        seconds_to_full = tokens_needed / self.refill_rate
        return datetime.now(timezone.utc) + timedelta(seconds=seconds_to_full)
    
    async def reset(self):
        """Reset bucket to full capacity"""
        async with self._lock:
            self.tokens = self.capacity
            self.last_refill = time.time()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current bucket status"""
        return {
            "algorithm": "token_bucket",
            "capacity": self.capacity,
            "tokens": self.tokens,
            "refill_rate": self.refill_rate,
            "utilization": (self.capacity - self.tokens) / self.capacity,
            "last_refill": self.last_refill
        }


class SlidingWindowLimiter(RateLimiterBase):
    """Sliding window rate limiter implementation"""
    
    def __init__(self, rule: RateLimitRule, identifier: str):
        super().__init__(rule, identifier)
        self.requests = deque()
        self.window_size = rule.period_seconds
    
    async def is_allowed(self, tokens: int = 1) -> RateLimitStatus:
        """Check sliding window constraints"""
        if not self.rule.enabled:
            return RateLimitStatus(
                allowed=True,
                remaining=float('inf'),
                reset_time=datetime.now(timezone.utc),
                rule=self.rule
            )
        
        async with self._lock:
            now = time.time()
            window_start = now - self.window_size
            
            # Remove old requests outside window
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()
            
            current_count = len(self.requests)
            
            if current_count + tokens <= self.rule.requests:
                # Add new request timestamps
                for _ in range(tokens):
                    self.requests.append(now)
                
                return RateLimitStatus(
                    allowed=True,
                    remaining=self.rule.requests - current_count - tokens,
                    reset_time=self._calculate_reset_time(),
                    current_usage=current_count + tokens,
                    rule=self.rule
                )
            else:
                # Calculate retry after time
                if self.requests:
                    oldest_request = self.requests[0]
                    retry_after = (oldest_request + self.window_size) - now
                else:
                    retry_after = self.window_size
                
                return RateLimitStatus(
                    allowed=False,
                    remaining=max(0, self.rule.requests - current_count),
                    reset_time=self._calculate_reset_time(),
                    retry_after=max(0, retry_after),
                    current_usage=current_count,
                    rule=self.rule
                )
    
    def _calculate_reset_time(self) -> datetime:
        """Calculate when oldest request will expire"""
        if not self.requests:
            return datetime.now(timezone.utc)
        
        oldest_request = self.requests[0]
        reset_timestamp = oldest_request + self.window_size
        return datetime.fromtimestamp(reset_timestamp, timezone.utc)
    
    async def reset(self):
        """Clear all requests from window"""
        async with self._lock:
            self.requests.clear()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current window status"""
        now = time.time()
        window_start = now - self.window_size
        
        # Count requests in current window
        current_requests = sum(1 for req_time in self.requests if req_time >= window_start)
        
        return {
            "algorithm": "sliding_window",
            "window_size": self.window_size,
            "current_requests": current_requests,
            "max_requests": self.rule.requests,
            "utilization": current_requests / self.rule.requests,
            "window_start": window_start
        }


class FixedWindowLimiter(RateLimiterBase):
    """Fixed window rate limiter implementation"""
    
    def __init__(self, rule: RateLimitRule, identifier: str):
        super().__init__(rule, identifier)
        self.window_start = self._get_current_window_start()
        self.request_count = 0
    
    def _get_current_window_start(self) -> float:
        """Get start time of current fixed window"""
        now = time.time()
        return now - (now % self.rule.period_seconds)
    
    async def is_allowed(self, tokens: int = 1) -> RateLimitStatus:
        """Check fixed window constraints"""
        if not self.rule.enabled:
            return RateLimitStatus(
                allowed=True,
                remaining=float('inf'),
                reset_time=datetime.now(timezone.utc),
                rule=self.rule
            )
        
        async with self._lock:
            current_window_start = self._get_current_window_start()
            
            # Reset counter if in new window
            if current_window_start > self.window_start:
                self.window_start = current_window_start
                self.request_count = 0
            
            if self.request_count + tokens <= self.rule.requests:
                self.request_count += tokens
                
                return RateLimitStatus(
                    allowed=True,
                    remaining=self.rule.requests - self.request_count,
                    reset_time=self._calculate_reset_time(),
                    current_usage=self.request_count,
                    rule=self.rule
                )
            else:
                retry_after = (self.window_start + self.rule.period_seconds) - time.time()
                
                return RateLimitStatus(
                    allowed=False,
                    remaining=max(0, self.rule.requests - self.request_count),
                    reset_time=self._calculate_reset_time(),
                    retry_after=max(0, retry_after),
                    current_usage=self.request_count,
                    rule=self.rule
                )
    
    def _calculate_reset_time(self) -> datetime:
        """Calculate when window resets"""
        reset_timestamp = self.window_start + self.rule.period_seconds
        return datetime.fromtimestamp(reset_timestamp, timezone.utc)
    
    async def reset(self):
        """Reset window counter"""
        async with self._lock:
            self.request_count = 0
            self.window_start = self._get_current_window_start()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current window status"""
        return {
            "algorithm": "fixed_window",
            "window_start": self.window_start,
            "window_size": self.rule.period_seconds,
            "request_count": self.request_count,
            "max_requests": self.rule.requests,
            "utilization": self.request_count / self.rule.requests,
            "time_to_reset": (self.window_start + self.rule.period_seconds) - time.time()
        }


class AdaptiveRateLimiter(TokenBucketLimiter):
    """Adaptive rate limiter that adjusts based on API responses"""
    
    def __init__(self, rule: RateLimitRule, identifier: str, adaptive_settings: AdaptiveSettings):
        # Start with base rate
        adaptive_rule = RateLimitRule(
            requests=adaptive_settings.base_rate,
            period_seconds=rule.period_seconds,
            algorithm=RateLimitAlgorithm.ADAPTIVE,
            burst_multiplier=rule.burst_multiplier,
            scope=rule.scope,
            enabled=rule.enabled
        )
        super().__init__(adaptive_rule, identifier)
        
        self.adaptive_settings = adaptive_settings
        self.success_count = 0
        self.error_count = 0
        self.last_adaptation = time.time()
        self.current_rate = adaptive_settings.base_rate
    
    async def record_response(self, success: bool, status_code: Optional[int] = None):
        """Record API response for adaptation"""
        async with self._lock:
            if success:
                self.success_count += 1
                self.error_count = max(0, self.error_count - 1)  # Slowly forget errors
            else:
                self.error_count += 1
                if status_code == 429:  # Rate limited
                    # Immediately reduce rate
                    await self._adapt_rate(force_decrease=True)
                    return
            
            # Check if adaptation is needed
            await self._check_adaptation()
    
    async def _check_adaptation(self):
        """Check if rate should be adapted"""
        now = time.time()
        if now - self.last_adaptation < self.adaptive_settings.adaptation_window_seconds:
            return
        
        # Increase rate if many successes
        if self.success_count >= self.adaptive_settings.success_threshold:
            await self._adapt_rate(increase=True)
        
        # Decrease rate if many errors
        elif self.error_count >= self.adaptive_settings.error_threshold:
            await self._adapt_rate(increase=False)
        
        # Reset counters
        self.success_count = 0
        self.error_count = 0
        self.last_adaptation = now
    
    async def _adapt_rate(self, increase: bool = True, force_decrease: bool = False):
        """Adapt the rate limit"""
        old_rate = self.current_rate
        
        if force_decrease or not increase:
            self.current_rate = max(
                self.adaptive_settings.min_rate,
                int(self.current_rate * self.adaptive_settings.decrease_factor)
            )
        else:
            self.current_rate = min(
                self.adaptive_settings.max_rate,
                int(self.current_rate * self.adaptive_settings.increase_factor)
            )
        
        if self.current_rate != old_rate:
            # Update rule and bucket parameters
            self.rule.requests = self.current_rate
            self.capacity = self.rule.burst_size
            self.refill_rate = self.rule.refill_rate
            
            # Adjust current tokens proportionally
            if old_rate > 0:
                self.tokens = int(self.tokens * (self.current_rate / old_rate))
                self.tokens = min(self.tokens, self.capacity)
            
            logger.info(
                f"Adaptive rate limit adjusted",
                identifier=self.identifier,
                old_rate=old_rate,
                new_rate=self.current_rate,
                reason="force_decrease" if force_decrease else ("increase" if increase else "decrease")
            )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get adaptive limiter status"""
        base_status = await super().get_status()
        base_status.update({
            "algorithm": "adaptive",
            "current_rate": self.current_rate,
            "base_rate": self.adaptive_settings.base_rate,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "last_adaptation": self.last_adaptation,
            "adaptation_window": self.adaptive_settings.adaptation_window_seconds
        })
        return base_status


class ServiceRateLimiter:
    """Rate limiter for a specific service with multiple endpoints"""
    
    def __init__(self, service_name: str, default_rules: Dict[str, RateLimitRule]):
        self.service_name = service_name
        self.default_rules = default_rules
        self.limiters: Dict[str, RateLimiterBase] = {}
        self.settings = get_settings()
        self._lock = asyncio.Lock()
    
    def _create_limiter(self, rule: RateLimitRule, identifier: str) -> RateLimiterBase:
        """Create appropriate limiter based on algorithm"""
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return TokenBucketLimiter(rule, identifier)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return SlidingWindowLimiter(rule, identifier)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return FixedWindowLimiter(rule, identifier)
        elif rule.algorithm == RateLimitAlgorithm.ADAPTIVE:
            adaptive_settings = AdaptiveSettings()  # Could be configurable
            return AdaptiveRateLimiter(rule, identifier, adaptive_settings)
        else:
            raise ValueError(f"Unsupported rate limit algorithm: {rule.algorithm}")
    
    async def is_allowed(
        self,
        endpoint: str = "default",
        tokens: int = 1,
        user_id: Optional[str] = None
    ) -> RateLimitStatus:
        """Check if request is allowed for endpoint"""
        
        # Get or create appropriate rule
        rule = self.default_rules.get(endpoint, self.default_rules.get("default"))
        if not rule:
            # No rule configured, allow by default
            return RateLimitStatus(
                allowed=True,
                remaining=float('inf'),
                reset_time=datetime.now(timezone.utc)
            )
        
        # Create identifier based on scope
        if rule.scope == RateLimitScope.USER and user_id:
            identifier = f"{self.service_name}:{endpoint}:{user_id}"
        elif rule.scope == RateLimitScope.ENDPOINT:
            identifier = f"{self.service_name}:{endpoint}"
        else:
            identifier = f"{self.service_name}"
        
        # Get or create limiter
        async with self._lock:
            if identifier not in self.limiters:
                self.limiters[identifier] = self._create_limiter(rule, identifier)
        
        limiter = self.limiters[identifier]
        return await limiter.is_allowed(tokens)
    
    async def record_response(
        self,
        endpoint: str,
        success: bool,
        status_code: Optional[int] = None,
        user_id: Optional[str] = None
    ):
        """Record API response for adaptive limiters"""
        
        rule = self.default_rules.get(endpoint, self.default_rules.get("default"))
        if not rule or rule.algorithm != RateLimitAlgorithm.ADAPTIVE:
            return
        
        # Get identifier
        if rule.scope == RateLimitScope.USER and user_id:
            identifier = f"{self.service_name}:{endpoint}:{user_id}"
        elif rule.scope == RateLimitScope.ENDPOINT:
            identifier = f"{self.service_name}:{endpoint}"
        else:
            identifier = f"{self.service_name}"
        
        # Record response if limiter exists
        if identifier in self.limiters:
            limiter = self.limiters[identifier]
            if isinstance(limiter, AdaptiveRateLimiter):
                await limiter.record_response(success, status_code)
    
    async def update_rule(self, endpoint: str, rule: RateLimitRule):
        """Update rate limiting rule for endpoint"""
        async with self._lock:
            self.default_rules[endpoint] = rule
            # Remove existing limiters to force recreation with new rule
            to_remove = [k for k in self.limiters.keys() if endpoint in k]
            for key in to_remove:
                del self.limiters[key]
        
        logger.info(f"Rate limit rule updated", service=self.service_name, endpoint=endpoint)
    
    async def reset_limiter(self, endpoint: str = "default", user_id: Optional[str] = None):
        """Reset specific limiter"""
        rule = self.default_rules.get(endpoint, self.default_rules.get("default"))
        if not rule:
            return
        
        # Get identifier
        if rule.scope == RateLimitScope.USER and user_id:
            identifier = f"{self.service_name}:{endpoint}:{user_id}"
        elif rule.scope == RateLimitScope.ENDPOINT:
            identifier = f"{self.service_name}:{endpoint}"
        else:
            identifier = f"{self.service_name}"
        
        if identifier in self.limiters:
            await self.limiters[identifier].reset()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of all limiters for this service"""
        status = {
            "service": self.service_name,
            "rules": {k: {"requests": v.requests, "period": v.period_seconds, "algorithm": v.algorithm.value} 
                     for k, v in self.default_rules.items()},
            "limiters": {}
        }
        
        for identifier, limiter in self.limiters.items():
            try:
                status["limiters"][identifier] = await limiter.get_status()
            except Exception as e:
                status["limiters"][identifier] = {"error": str(e)}
        
        return status


class RateLimitManager:
    """Global rate limit manager for all services"""
    
    def __init__(self):
        self.services: Dict[str, ServiceRateLimiter] = {}
        self.settings = get_settings()
        self._lock = asyncio.Lock()
        
        # Load default configurations
        self._load_default_configs()
    
    def _load_default_configs(self):
        """Load default rate limiting configurations"""
        # Gmail API limits
        gmail_rules = {
            "default": RateLimitRule(
                requests=250, period_seconds=60,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET
            ),
            "search": RateLimitRule(
                requests=10, period_seconds=60,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            )
        }
        
        # Google Drive API limits  
        drive_rules = {
            "default": RateLimitRule(
                requests=100, period_seconds=60,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET
            )
        }
        
        # Twitter API limits
        twitter_rules = {
            "default": RateLimitRule(
                requests=300, period_seconds=900,  # 15 minutes
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW
            ),
            "tweets": RateLimitRule(
                requests=300, period_seconds=900,
                algorithm=RateLimitAlgorithm.ADAPTIVE
            )
        }
        
        # LinkedIn API limits (more restrictive)
        linkedin_rules = {
            "default": RateLimitRule(
                requests=100, period_seconds=3600,  # 1 hour
                algorithm=RateLimitAlgorithm.FIXED_WINDOW
            )
        }
        
        # Register default services
        self.services["gmail"] = ServiceRateLimiter("gmail", gmail_rules)
        self.services["drive"] = ServiceRateLimiter("drive", drive_rules)
        self.services["twitter"] = ServiceRateLimiter("twitter", twitter_rules)
        self.services["linkedin"] = ServiceRateLimiter("linkedin", linkedin_rules)
    
    async def get_service_limiter(self, service_name: str) -> ServiceRateLimiter:
        """Get service rate limiter, creating if necessary"""
        async with self._lock:
            if service_name not in self.services:
                # Create with basic default rule
                default_rule = RateLimitRule(
                    requests=60, period_seconds=60,
                    algorithm=RateLimitAlgorithm.TOKEN_BUCKET
                )
                self.services[service_name] = ServiceRateLimiter(
                    service_name, {"default": default_rule}
                )
        
        return self.services[service_name]
    
    async def is_allowed(
        self,
        service: str,
        endpoint: str = "default",
        tokens: int = 1,
        user_id: Optional[str] = None
    ) -> RateLimitStatus:
        """Check if request is allowed"""
        service_limiter = await self.get_service_limiter(service)
        return await service_limiter.is_allowed(endpoint, tokens, user_id)
    
    async def record_response(
        self,
        service: str,
        endpoint: str,
        success: bool,
        status_code: Optional[int] = None,
        user_id: Optional[str] = None
    ):
        """Record API response for adaptive rate limiting"""
        if service in self.services:
            await self.services[service].record_response(
                endpoint, success, status_code, user_id
            )
    
    async def update_service_rules(self, service: str, rules: Dict[str, RateLimitRule]):
        """Update all rules for a service"""
        service_limiter = await self.get_service_limiter(service)
        for endpoint, rule in rules.items():
            await service_limiter.update_rule(endpoint, rule)
    
    async def get_all_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {}
        for service_name, service_limiter in self.services.items():
            try:
                status[service_name] = await service_limiter.get_status()
            except Exception as e:
                status[service_name] = {"error": str(e)}
        return status
    
    async def reset_service(self, service: str):
        """Reset all limiters for a service"""
        if service in self.services:
            service_limiter = self.services[service]
            # Reset all limiters in the service
            for limiter in service_limiter.limiters.values():
                await limiter.reset()
    
    async def cleanup_inactive_limiters(self, max_age_seconds: int = 3600):
        """Remove limiters that haven't been used recently"""
        cutoff_time = time.time() - max_age_seconds
        
        for service_limiter in self.services.values():
            async with service_limiter._lock:
                to_remove = []
                for identifier, limiter in service_limiter.limiters.items():
                    # Check if limiter has been inactive
                    if hasattr(limiter, 'last_refill') and limiter.last_refill < cutoff_time:
                        to_remove.append(identifier)
                
                for identifier in to_remove:
                    del service_limiter.limiters[identifier]
                
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} inactive limiters", service=service_limiter.service_name)


# Global rate limit manager
rate_limit_manager: Optional[RateLimitManager] = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get global rate limit manager singleton"""
    global rate_limit_manager
    if rate_limit_manager is None:
        rate_limit_manager = RateLimitManager()
    return rate_limit_manager


async def enforce_rate_limit(
    service: str,
    endpoint: str = "default",
    tokens: int = 1,
    user_id: Optional[str] = None
) -> RateLimitStatus:
    """Convenience function to enforce rate limiting"""
    manager = get_rate_limit_manager()
    return await manager.is_allowed(service, endpoint, tokens, user_id)


async def record_api_response(
    service: str,
    endpoint: str,
    success: bool,
    status_code: Optional[int] = None,
    user_id: Optional[str] = None
):
    """Convenience function to record API responses"""
    manager = get_rate_limit_manager()
    await manager.record_response(service, endpoint, success, status_code, user_id)