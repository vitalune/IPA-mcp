"""
API Integrations Package

This package provides comprehensive API integrations for external services:
- Gmail API client for email management
- Google Drive API client for file operations  
- Twitter/X API v2 client for social media
- LinkedIn API client for professional networking
- Base client with common functionality (rate limiting, auth, etc.)
- Rate limiting utilities for API management

All clients support:
- OAuth2 authentication with secure token storage
- Comprehensive rate limiting and retry logic
- Circuit breaker patterns for reliability
- Async/await support
- Comprehensive error handling and logging
- Data export capabilities
"""

from .base_client import BaseAPIClient, get_client_registry
from .gmail_client import GmailClient, GmailMessage, GmailThread, GmailLabel
from .drive_client import DriveClient, DriveFile, DrivePermission, DriveRevision
from .twitter_client import TwitterClient, TwitterUser, TwitterTweet, TwitterMedia
from .linkedin_client import LinkedInClient, LinkedInProfile, LinkedInPost, LinkedInConnection
from .client_manager import APIClientManager, get_client_manager

__all__ = [
    # Base functionality
    "BaseAPIClient",
    "get_client_registry",
    
    # Client management
    "APIClientManager",
    "get_client_manager",
    
    # Gmail integration
    "GmailClient",
    "GmailMessage", 
    "GmailThread",
    "GmailLabel",
    
    # Google Drive integration
    "DriveClient",
    "DriveFile",
    "DrivePermission", 
    "DriveRevision",
    
    # Twitter integration
    "TwitterClient",
    "TwitterUser",
    "TwitterTweet",
    "TwitterMedia",
    
    # LinkedIn integration
    "LinkedInClient",
    "LinkedInProfile",
    "LinkedInPost",
    "LinkedInConnection",
]