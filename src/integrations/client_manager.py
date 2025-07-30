"""
API Client Manager

This module provides a centralized manager for all API clients with:
- Unified authentication flow for multiple services
- Client lifecycle management
- Health monitoring and status reporting
- Coordinated rate limiting across services
- Bulk operations and data synchronization
- Configuration management for all clients
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

import structlog

from .base_client import get_client_registry
from .gmail_client import GmailClient
from .drive_client import DriveClient  
from .twitter_client import TwitterClient
from .linkedin_client import LinkedInClient
from ..config.auth import AuthProvider, get_auth_manager
from ..config.settings import get_settings
from ..utils.rate_limiter import get_rate_limit_manager

logger = structlog.get_logger(__name__)


@dataclass
class ClientConfig:
    """Configuration for an API client"""
    client_id: str
    client_secret: str
    scopes: List[str] = field(default_factory=list)
    enabled: bool = True
    auto_refresh: bool = True
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceStatus:
    """Status information for a service"""
    service_name: str
    authenticated: bool
    healthy: bool
    last_check: datetime
    error_message: Optional[str] = None
    rate_limit_status: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class APIClientManager:
    """Centralized manager for all API clients"""
    
    def __init__(self):
        self.settings = get_settings()
        self.auth_manager = get_auth_manager()
        self.rate_limiter = get_rate_limit_manager()
        self.client_registry = get_client_registry()
        
        # Client instances
        self._clients: Dict[str, Any] = {}
        self._client_configs: Dict[str, ClientConfig] = {}
        
        # Status tracking
        self._service_status: Dict[str, ServiceStatus] = {}
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._token_refresh_task: Optional[asyncio.Task] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()
    
    async def initialize(self):
        """Initialize the client manager"""
        logger.info("Initializing API Client Manager")
        
        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._token_refresh_task = asyncio.create_task(self._token_refresh_loop())
        
        logger.info("API Client Manager initialized successfully")
    
    async def shutdown(self):
        """Shutdown the client manager"""
        logger.info("Shutting down API Client Manager")
        
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._token_refresh_task:
            self._token_refresh_task.cancel()
            try:
                await self._token_refresh_task
            except asyncio.CancelledError:
                pass
        
        # Close all clients
        await self.client_registry.close_all()
        
        logger.info("API Client Manager shutdown complete")
    
    def configure_service(
        self,
        service: str,
        client_id: str,
        client_secret: str,
        scopes: Optional[List[str]] = None,
        **kwargs
    ):
        """Configure a service for use"""
        config = ClientConfig(
            client_id=client_id,
            client_secret=client_secret,
            scopes=scopes or [],
            **kwargs
        )
        
        self._client_configs[service] = config
        logger.info(f"Configured service: {service}", scopes=scopes)
    
    async def create_client(self, service: str) -> Optional[Any]:
        """Create and register an API client"""
        if service not in self._client_configs:
            logger.error(f"Service not configured: {service}")
            return None
        
        config = self._client_configs[service]
        if not config.enabled:
            logger.info(f"Service disabled: {service}")
            return None
        
        try:
            # Create appropriate client
            if service == "gmail":
                client = GmailClient(
                    client_id=config.client_id,
                    client_secret=config.client_secret,
                    scopes=config.scopes,
                    **config.custom_settings
                )
            elif service == "drive":
                client = DriveClient(
                    client_id=config.client_id,
                    client_secret=config.client_secret,
                    scopes=config.scopes,
                    **config.custom_settings
                )
            elif service == "twitter":
                client = TwitterClient(
                    client_id=config.client_id,
                    client_secret=config.client_secret,
                    scopes=config.scopes,
                    **config.custom_settings
                )
            elif service == "linkedin":
                client = LinkedInClient(
                    client_id=config.client_id,
                    client_secret=config.client_secret,
                    scopes=config.scopes,
                    **config.custom_settings
                )
            else:
                logger.error(f"Unknown service: {service}")
                return None
            
            # Register client
            await self.client_registry.register_client(service, client)
            self._clients[service] = client
            
            # Initialize status tracking
            self._service_status[service] = ServiceStatus(
                service_name=service,
                authenticated=False,
                healthy=False,
                last_check=datetime.now(timezone.utc)
            )
            
            logger.info(f"Created client for service: {service}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create client for {service}", error=str(e))
            return None
    
    async def get_client(self, service: str) -> Optional[Any]:
        """Get an API client, creating if necessary"""
        if service in self._clients:
            return self._clients[service]
        
        return await self.create_client(service)
    
    async def authenticate_service(self, service: str, redirect_uri: str = None) -> str:
        """Start authentication flow for a service"""
        client = await self.get_client(service)
        if not client:
            raise ValueError(f"Client for {service} not available")
        
        redirect_uri = redirect_uri or "http://localhost:8080/oauth/callback"
        
        try:
            # This will raise an exception with the authorization URL
            await client.authenticate(redirect_uri)
        except Exception as e:
            # Extract auth URL from exception message
            auth_url = str(e).replace("Please visit this URL to authorize: ", "")
            logger.info(f"Authentication URL for {service}: {auth_url}")
            return auth_url
    
    async def handle_oauth_callback(
        self,
        service: str,
        authorization_code: str,
        state: str
    ) -> bool:
        """Handle OAuth callback for a service"""
        client = await self.get_client(service)
        if not client:
            return False
        
        success = await client.handle_oauth_callback(authorization_code, state)
        
        if success:
            # Update status
            if service in self._service_status:
                self._service_status[service].authenticated = True
                self._service_status[service].healthy = True
                self._service_status[service].last_check = datetime.now(timezone.utc)
        
        return success
    
    async def check_authentication(self, service: str) -> bool:
        """Check if a service is authenticated"""
        try:
            # Check if we have valid tokens
            tokens = await self.auth_manager.list_tokens(
                provider=AuthProvider(service)
            )
            
            valid_tokens = [t for t in tokens if t["status"] == "valid"]
            return len(valid_tokens) > 0
            
        except Exception as e:
            logger.error(f"Failed to check authentication for {service}", error=str(e))
            return False
    
    async def refresh_tokens(self, service: Optional[str] = None) -> Dict[str, bool]:
        """Refresh tokens for one or all services"""
        results = {}
        
        services_to_refresh = [service] if service else self._clients.keys()
        
        for svc in services_to_refresh:
            if svc not in self._clients:
                continue
            
            try:
                client = self._clients[svc]
                success = await client.refresh_token()
                results[svc] = success
                
                logger.info(f"Token refresh for {svc}: {'success' if success else 'failed'}")
                
            except Exception as e:
                logger.error(f"Failed to refresh token for {svc}", error=str(e))
                results[svc] = False
        
        return results
    
    async def get_service_status(self, service: Optional[str] = None) -> Dict[str, ServiceStatus]:
        """Get status for one or all services"""
        if service:
            return {service: self._service_status.get(service)} if service in self._service_status else {}
        
        return self._service_status.copy()
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status of all services"""
        total_services = len(self._service_status)
        healthy_services = sum(1 for status in self._service_status.values() if status.healthy)
        authenticated_services = sum(1 for status in self._service_status.values() if status.authenticated)
        
        # Get rate limit status for all services
        rate_limit_status = await self.rate_limiter.get_all_status()
        
        # Get client health from registry
        client_health = await self.client_registry.get_all_health_status()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": "healthy" if healthy_services == total_services else "degraded",
            "services": {
                "total": total_services,
                "healthy": healthy_services,
                "authenticated": authenticated_services,
                "unhealthy": total_services - healthy_services
            },
            "rate_limits": rate_limit_status,
            "client_health": client_health,
            "service_details": {
                name: {
                    "authenticated": status.authenticated,
                    "healthy": status.healthy,
                    "last_check": status.last_check.isoformat(),
                    "error_message": status.error_message
                }
                for name, status in self._service_status.items()
            }
        }
    
    async def bulk_export(
        self,
        output_directory: Path,
        services: Optional[List[str]] = None,
        data_types: Optional[Dict[str, List[str]]] = None,
        max_items: int = 1000
    ) -> Dict[str, int]:
        """Export data from multiple services"""
        logger.info(f"Starting bulk export", output_dir=str(output_directory), services=services)
        
        output_directory.mkdir(parents=True, exist_ok=True)
        results = {}
        
        services_to_export = services or list(self._clients.keys())
        
        for service in services_to_export:
            if service not in self._clients:
                logger.warning(f"Service not available: {service}")
                continue
            
            if not await self.check_authentication(service):
                logger.warning(f"Service not authenticated: {service}")
                continue
            
            try:
                client = self._clients[service]
                service_output_dir = output_directory / service
                
                # Get data types for this service
                service_data_types = None
                if data_types and service in data_types:
                    service_data_types = data_types[service]
                
                # Export data
                count = await client.export_data(
                    service_output_dir,
                    data_types=service_data_types,
                    max_items=max_items
                )
                
                results[service] = count
                logger.info(f"Exported {count} items from {service}")
                
            except Exception as e:
                logger.error(f"Failed to export from {service}", error=str(e))
                results[service] = 0
        
        total_exported = sum(results.values())
        logger.info(f"Bulk export completed", total_exported=total_exported, results=results)
        
        return results
    
    async def sync_data(
        self,
        source_service: str,
        target_service: str,
        data_type: str,
        max_items: int = 100
    ) -> int:
        """Synchronize data between services"""
        logger.info(f"Starting data sync", source=source_service, target=target_service, data_type=data_type)
        
        source_client = await self.get_client(source_service)
        target_client = await self.get_client(target_service)
        
        if not source_client or not target_client:
            logger.error("Source or target client not available")
            return 0
        
        if not await self.check_authentication(source_service):
            logger.error(f"Source service not authenticated: {source_service}")
            return 0
        
        if not await self.check_authentication(target_service):
            logger.error(f"Target service not authenticated: {target_service}")
            return 0
        
        synced_count = 0
        
        try:
            # This is a basic example - specific sync logic would depend on the services and data types
            # For now, we'll just demonstrate the pattern
            
            if data_type == "posts" and source_service == "twitter" and target_service == "linkedin":
                # Example: Sync recent tweets to LinkedIn
                tweets_result = await source_client.get_home_timeline(max_results=max_items)
                tweets = tweets_result.get('tweets', [])
                
                for tweet in tweets[:10]:  # Limit to avoid spam
                    try:
                        # Simple cross-post (would need more sophisticated logic in practice)
                        await target_client.create_post(
                            text=f"From Twitter: {tweet.text[:200]}...",
                            visibility="PUBLIC"
                        )
                        synced_count += 1
                        
                        # Add delay to respect rate limits
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Failed to sync tweet", tweet_id=tweet.id, error=str(e))
                        continue
            
            else:
                logger.warning(f"Sync not implemented for {source_service} -> {target_service} ({data_type})")
            
        except Exception as e:
            logger.error(f"Data sync failed", error=str(e))
        
        logger.info(f"Data sync completed", synced_count=synced_count)
        return synced_count
    
    async def _health_check_loop(self):
        """Background task for periodic health checks"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for service, client in self._clients.items():
                    try:
                        # Check client health
                        health = await client.get_health_status()
                        
                        # Update status
                        if service in self._service_status:
                            status = self._service_status[service]
                            status.healthy = health.get("status") != "error"
                            status.last_check = datetime.now(timezone.utc)
                            status.metrics = health.get("metrics", {})
                            
                            if not status.healthy:
                                status.error_message = health.get("error", "Unknown error")
                            else:
                                status.error_message = None
                    
                    except Exception as e:
                        logger.error(f"Health check failed for {service}", error=str(e))
                        if service in self._service_status:
                            self._service_status[service].healthy = False
                            self._service_status[service].error_message = str(e)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
    
    async def _token_refresh_loop(self):
        """Background task for periodic token refresh"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check for tokens that need refresh
                expired_count = await self.auth_manager.cleanup_expired_tokens()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired tokens")
                
                # Attempt to refresh tokens that are close to expiry
                rotation_count = await self.auth_manager.rotate_tokens()
                if rotation_count > 0:
                    logger.info(f"Identified {rotation_count} tokens needing rotation")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Token refresh loop error", error=str(e))
    
    async def save_configuration(self, config_path: Path):
        """Save client configuration to file"""
        config_data = {
            "services": {
                service: {
                    "client_id": config.client_id,
                    "scopes": config.scopes,
                    "enabled": config.enabled,
                    "auto_refresh": config.auto_refresh,
                    "custom_settings": config.custom_settings
                    # Note: client_secret is not saved for security
                }
                for service, config in self._client_configs.items()
            },
            "saved_at": datetime.now(timezone.utc).isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    async def load_configuration(self, config_path: Path):
        """Load client configuration from file"""
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for service, service_config in config_data.get("services", {}).items():
                # Client secret will need to be provided separately
                self._client_configs[service] = ClientConfig(
                    client_id=service_config["client_id"],
                    client_secret="",  # Must be set separately
                    scopes=service_config["scopes"],
                    enabled=service_config["enabled"],
                    auto_refresh=service_config["auto_refresh"],
                    custom_settings=service_config["custom_settings"]
                )
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration", config_path=str(config_path), error=str(e))


# Global client manager instance
client_manager: Optional[APIClientManager] = None


def get_client_manager() -> APIClientManager:
    """Get the global client manager singleton"""
    global client_manager
    if client_manager is None:
        client_manager = APIClientManager()
    return client_manager