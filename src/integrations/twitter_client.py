"""
Twitter/X API v2 Client Implementation

This module provides comprehensive Twitter API v2 integration with:
- OAuth2 with PKCE authentication flow
- Tweet operations (read, post, delete, retweet, like)
- User profile management and following
- Timeline operations (home, user, mentions)
- Media upload support for images and videos
- Analytics and engagement metrics
- Rate limiting for different endpoint tiers
- Support for both app-only and user context
- Real-time streaming capabilities
"""

import asyncio
import hashlib
import json
import secrets
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
import base64

import httpx
import structlog

from .base_client import BaseAPIClient, RateLimitConfig, CircuitBreakerConfig
from ..config.auth import AuthProvider, TokenType
from ..utils.rate_limiter import get_rate_limit_manager

logger = structlog.get_logger(__name__)


@dataclass
class TwitterUser:
    """Twitter user data structure"""
    id: str
    username: str
    name: str
    created_at: Optional[datetime] = None
    description: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    verified: bool = False
    protected: bool = False
    profile_image_url: Optional[str] = None
    public_metrics: Dict[str, int] = field(default_factory=dict)
    pinned_tweet_id: Optional[str] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'TwitterUser':
        """Create TwitterUser from API response"""
        created_at = None
        if 'created_at' in data:
            created_at = datetime.fromisoformat(
                data['created_at'].replace('Z', '+00:00')
            )
        
        return cls(
            id=data['id'],
            username=data['username'],
            name=data['name'],
            created_at=created_at,
            description=data.get('description'),
            location=data.get('location'),
            url=data.get('url'),
            verified=data.get('verified', False),
            protected=data.get('protected', False),
            profile_image_url=data.get('profile_image_url'),
            public_metrics=data.get('public_metrics', {}),
            pinned_tweet_id=data.get('pinned_tweet_id')
        )


@dataclass
class TwitterTweet:
    """Twitter tweet data structure"""
    id: str
    text: str
    created_at: Optional[datetime] = None
    author_id: Optional[str] = None
    conversation_id: Optional[str] = None
    in_reply_to_user_id: Optional[str] = None
    referenced_tweets: List[Dict[str, Any]] = field(default_factory=list)
    attachments: Dict[str, List[str]] = field(default_factory=dict)
    public_metrics: Dict[str, int] = field(default_factory=dict)
    organic_metrics: Dict[str, int] = field(default_factory=dict)
    promoted_metrics: Dict[str, int] = field(default_factory=dict)
    non_public_metrics: Dict[str, int] = field(default_factory=dict)
    context_annotations: List[Dict[str, Any]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    geo: Optional[Dict[str, Any]] = None
    lang: Optional[str] = None
    possibly_sensitive: bool = False
    reply_settings: Optional[str] = None
    source: Optional[str] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'TwitterTweet':
        """Create TwitterTweet from API response"""
        created_at = None
        if 'created_at' in data:
            created_at = datetime.fromisoformat(
                data['created_at'].replace('Z', '+00:00')
            )
        
        return cls(
            id=data['id'],
            text=data['text'],
            created_at=created_at,
            author_id=data.get('author_id'),
            conversation_id=data.get('conversation_id'),
            in_reply_to_user_id=data.get('in_reply_to_user_id'),
            referenced_tweets=data.get('referenced_tweets', []),
            attachments=data.get('attachments', {}),
            public_metrics=data.get('public_metrics', {}),
            organic_metrics=data.get('organic_metrics', {}),
            promoted_metrics=data.get('promoted_metrics', {}),
            non_public_metrics=data.get('non_public_metrics', {}),
            context_annotations=data.get('context_annotations', []),
            entities=data.get('entities', {}),
            geo=data.get('geo'),
            lang=data.get('lang'),
            possibly_sensitive=data.get('possibly_sensitive', False),
            reply_settings=data.get('reply_settings'),
            source=data.get('source')
        )


@dataclass
class TwitterMedia:
    """Twitter media data structure"""
    media_key: str
    type: str  # photo, video, animated_gif
    url: Optional[str] = None
    preview_image_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_ms: Optional[int] = None
    public_metrics: Dict[str, int] = field(default_factory=dict)
    non_public_metrics: Dict[str, int] = field(default_factory=dict)
    organic_metrics: Dict[str, int] = field(default_factory=dict)
    promoted_metrics: Dict[str, int] = field(default_factory=dict)
    alt_text: Optional[str] = None
    variants: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'TwitterMedia':
        """Create TwitterMedia from API response"""
        return cls(
            media_key=data['media_key'],
            type=data['type'],
            url=data.get('url'),
            preview_image_url=data.get('preview_image_url'),
            width=data.get('width'),
            height=data.get('height'),
            duration_ms=data.get('duration_ms'),
            public_metrics=data.get('public_metrics', {}),
            non_public_metrics=data.get('non_public_metrics', {}),
            organic_metrics=data.get('organic_metrics', {}),
            promoted_metrics=data.get('promoted_metrics', {}),
            alt_text=data.get('alt_text'),
            variants=data.get('variants', [])
        )


class TwitterClient(BaseAPIClient):
    """Twitter API v2 client with OAuth2 PKCE authentication"""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        scopes: Optional[List[str]] = None,
        **kwargs
    ):
        # Twitter-specific rate limiting (varies by endpoint)
        rate_limit_config = RateLimitConfig(
            requests_per_minute=300,  # Most endpoints
            requests_per_hour=18000,
            requests_per_day=432000,
            burst_size=5
        )
        
        super().__init__(
            provider=AuthProvider.TWITTER,
            base_url="https://api.twitter.com/2",
            rate_limit_config=rate_limit_config,
            **kwargs
        )
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes or [
            'tweet.read',
            'users.read',
            'bookmark.read',
            'like.read',
            'follows.read',
            'offline.access'
        ]
        
        self.rate_limiter = get_rate_limit_manager()
        
        # OAuth2 PKCE parameters
        self._code_verifier: Optional[str] = None
        self._code_challenge: Optional[str] = None
        
        # User context (filled after authentication)
        self._authenticated_user: Optional[TwitterUser] = None
    
    def _format_auth_header(self, token: str) -> Dict[str, str]:
        """Format authentication header for Twitter API"""
        return {"Authorization": f"Bearer {token}"}
    
    def _generate_pkce_params(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge"""
        # Generate code verifier (43-128 characters)
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        
        # Generate code challenge (SHA256 hash of verifier)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        return code_verifier, code_challenge
    
    async def authenticate(self, redirect_uri: str = "http://localhost:8080/oauth/callback") -> bool:
        """Authenticate with Twitter using OAuth2 with PKCE"""
        try:
            # Generate PKCE parameters
            self._code_verifier, self._code_challenge = self._generate_pkce_params()
            
            # Build authorization URL
            auth_params = {
                'response_type': 'code',
                'client_id': self.client_id,
                'redirect_uri': redirect_uri,
                'scope': ' '.join(self.scopes),
                'state': secrets.token_urlsafe(32),
                'code_challenge': self._code_challenge,
                'code_challenge_method': 'S256'
            }
            
            auth_url = "https://twitter.com/i/oauth2/authorize?" + urllib.parse.urlencode(auth_params)
            
            # Store session state
            session_data = self.auth_manager.create_session_state(
                provider=self.provider,
                redirect_uri=redirect_uri,
                scopes=self.scopes
            )
            
            logger.info(
                "Twitter OAuth flow initiated",
                auth_url=auth_url,
                state=auth_params['state'][:8] + "...",
                scopes=self.scopes
            )
            
            raise Exception(f"Please visit this URL to authorize: {auth_url}")
            
        except Exception as e:
            logger.error(f"Twitter authentication failed", error=str(e))
            return False
    
    async def handle_oauth_callback(self, authorization_code: str, state: str) -> bool:
        """Handle OAuth callback with authorization code"""
        try:
            # Validate session state
            session_data = self.auth_manager.consume_session_state(state)
            if not session_data:
                raise ValueError("Invalid or expired OAuth state")
            
            # Exchange authorization code for tokens
            token_data = {
                'grant_type': 'authorization_code',
                'client_id': self.client_id,
                'code': authorization_code,
                'redirect_uri': session_data["redirect_uri"],
                'code_verifier': self._code_verifier
            }
            
            # Basic auth for client credentials
            auth_header = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
            
            headers = {
                'Authorization': f'Basic {auth_header}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = await self._make_request(
                method="POST",
                endpoint="https://api.twitter.com/2/oauth2/token",
                data=token_data,
                headers=headers,
                authenticated=False
            )
            
            token_response = response.json()
            
            # Store tokens securely
            access_token_id = await self.auth_manager.store_token(
                provider=self.provider,
                token_type=TokenType.ACCESS_TOKEN,
                token_value=token_response['access_token'],
                expires_in=token_response.get('expires_in', 7200),  # Default 2 hours
                scopes=self.scopes,
                subject=None,  # Will be populated after getting user info
                client_id=self.client_id
            )
            
            if 'refresh_token' in token_response:
                refresh_token_id = await self.auth_manager.store_token(
                    provider=self.provider,
                    token_type=TokenType.REFRESH_TOKEN,
                    token_value=token_response['refresh_token'],
                    expires_in=None,  # Refresh tokens don't expire
                    scopes=self.scopes,
                    subject=None,
                    client_id=self.client_id
                )
            
            # Get authenticated user info
            await self._load_authenticated_user()
            
            logger.info(
                "Twitter authentication successful",
                access_token_id=access_token_id,
                user_id=self._authenticated_user.id if self._authenticated_user else None,
                scopes=self.scopes
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Twitter OAuth callback failed", error=str(e))
            return False
    
    async def refresh_token(self) -> bool:
        """Refresh Twitter access token"""
        try:
            # Find refresh token
            tokens = await self.auth_manager.list_tokens(
                provider=self.provider,
                token_type=TokenType.REFRESH_TOKEN
            )
            
            if not tokens:
                logger.error("No refresh token found for Twitter")
                return False
            
            refresh_token = await self.auth_manager.retrieve_token(tokens[0]["token_id"])
            if not refresh_token:
                logger.error("Failed to retrieve Twitter refresh token")
                return False
            
            # Refresh token request
            token_data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': self.client_id
            }
            
            auth_header = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
            
            headers = {
                'Authorization': f'Basic {auth_header}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = await self._make_request(
                method="POST",
                endpoint="https://api.twitter.com/2/oauth2/token",
                data=token_data,
                headers=headers,
                authenticated=False
            )
            
            token_response = response.json()
            
            # Store new access token
            access_token_id = await self.auth_manager.store_token(
                provider=self.provider,
                token_type=TokenType.ACCESS_TOKEN,
                token_value=token_response['access_token'],
                expires_in=token_response.get('expires_in', 7200),
                scopes=self.scopes,
                subject=tokens[0].get("subject"),
                client_id=self.client_id
            )
            
            # Store new refresh token if provided
            if 'refresh_token' in token_response:
                await self.auth_manager.store_token(
                    provider=self.provider,
                    token_type=TokenType.REFRESH_TOKEN,
                    token_value=token_response['refresh_token'],
                    expires_in=None,
                    scopes=self.scopes,
                    subject=tokens[0].get("subject"),
                    client_id=self.client_id
                )
            
            logger.info("Twitter token refreshed successfully", token_id=access_token_id)
            return True
            
        except Exception as e:
            logger.error(f"Twitter token refresh failed", error=str(e))
            return False
    
    async def _load_authenticated_user(self):
        """Load authenticated user information"""
        try:
            user_data = await self.get_me()
            self._authenticated_user = TwitterUser.from_api_response(user_data)
        except Exception as e:
            logger.warning(f"Failed to load authenticated user", error=str(e))
    
    async def get_rate_limits(self) -> Dict[str, Any]:
        """Get current rate limit status for Twitter"""
        status = await self.rate_limiter.get_all_status()
        return status.get("twitter", {})
    
    async def get_me(self) -> Dict[str, Any]:
        """Get authenticated user information"""
        status = await self.rate_limiter.is_allowed("twitter", "users")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'user.fields': 'created_at,description,location,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified'
            }
            
            response = await self._make_request(
                method="GET",
                endpoint="/users/me",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "users", True)
            
            return response.json()['data']
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "users", False)
            logger.error(f"Failed to get Twitter user info", error=str(e))
            raise
    
    async def get_user_by_username(self, username: str) -> TwitterUser:
        """Get user by username"""
        status = await self.rate_limiter.is_allowed("twitter", "users")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'user.fields': 'created_at,description,location,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified'
            }
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/users/by/username/{username}",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "users", True)
            
            return TwitterUser.from_api_response(response.json()['data'])
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "users", False)
            logger.error(f"Failed to get Twitter user", username=username, error=str(e))
            raise
    
    async def get_user_by_id(self, user_id: str) -> TwitterUser:
        """Get user by ID"""
        status = await self.rate_limiter.is_allowed("twitter", "users")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'user.fields': 'created_at,description,location,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified'
            }
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/users/{user_id}",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "users", True)
            
            return TwitterUser.from_api_response(response.json()['data'])
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "users", False)
            logger.error(f"Failed to get Twitter user", user_id=user_id, error=str(e))
            raise
    
    async def get_tweet(self, tweet_id: str, include_metrics: bool = True) -> TwitterTweet:
        """Get a specific tweet"""
        status = await self.rate_limiter.is_allowed("twitter", "tweets")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,referenced_tweets,reply_settings,source,text',
                'expansions': 'author_id,referenced_tweets.id,attachments.media_keys',
                'user.fields': 'created_at,description,location,profile_image_url,protected,public_metrics,url,username,verified',
                'media.fields': 'duration_ms,height,media_key,preview_image_url,type,url,width,alt_text'
            }
            
            if include_metrics:
                params['tweet.fields'] += ',public_metrics,organic_metrics,promoted_metrics,non_public_metrics'
                params['media.fields'] += ',public_metrics,organic_metrics,promoted_metrics,non_public_metrics'
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/tweets/{tweet_id}",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "tweets", True)
            
            return TwitterTweet.from_api_response(response.json()['data'])
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "tweets", False)
            logger.error(f"Failed to get tweet", tweet_id=tweet_id, error=str(e))
            raise
    
    async def get_user_tweets(
        self,
        user_id: str,
        max_results: int = 10,
        pagination_token: Optional[str] = None,
        exclude_replies: bool = True,
        exclude_retweets: bool = True
    ) -> Dict[str, Any]:
        """Get tweets from a specific user"""
        status = await self.rate_limiter.is_allowed("twitter", "tweets")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'max_results': min(max_results, 100),  # Twitter limit
                'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text',
                'expansions': 'attachments.media_keys,referenced_tweets.id',
                'exclude': []
            }
            
            if exclude_replies:
                params['exclude'].append('replies')
            if exclude_retweets:
                params['exclude'].append('retweets')
            
            if params['exclude']:
                params['exclude'] = ','.join(params['exclude'])
            else:
                del params['exclude']
            
            if pagination_token:
                params['pagination_token'] = pagination_token
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/users/{user_id}/tweets",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "tweets", True)
            
            result = response.json()
            tweets = [TwitterTweet.from_api_response(tweet_data) for tweet_data in result.get('data', [])]
            
            return {
                'tweets': tweets,
                'next_token': result.get('meta', {}).get('next_token'),
                'result_count': result.get('meta', {}).get('result_count', 0)
            }
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "tweets", False)
            logger.error(f"Failed to get user tweets", user_id=user_id, error=str(e))
            raise
    
    async def get_home_timeline(
        self,
        max_results: int = 10,
        pagination_token: Optional[str] = None,
        exclude_replies: bool = True
    ) -> Dict[str, Any]:
        """Get home timeline for authenticated user"""
        status = await self.rate_limiter.is_allowed("twitter", "timeline", user_id=self._authenticated_user.id if self._authenticated_user else None)
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'max_results': min(max_results, 100),
                'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text',
                'expansions': 'author_id,attachments.media_keys,referenced_tweets.id',
                'user.fields': 'created_at,description,location,profile_image_url,protected,public_metrics,url,username,verified'
            }
            
            if exclude_replies:
                params['exclude'] = 'replies'
            
            if pagination_token:
                params['pagination_token'] = pagination_token
            
            response = await self._make_request(
                method="GET",
                endpoint="/users/me/timelines/reverse_chronological",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "timeline", True, user_id=self._authenticated_user.id if self._authenticated_user else None)
            
            result = response.json()
            tweets = [TwitterTweet.from_api_response(tweet_data) for tweet_data in result.get('data', [])]
            
            return {
                'tweets': tweets,
                'next_token': result.get('meta', {}).get('next_token'),
                'result_count': result.get('meta', {}).get('result_count', 0),
                'includes': result.get('includes', {})
            }
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "timeline", False, user_id=self._authenticated_user.id if self._authenticated_user else None)
            logger.error(f"Failed to get home timeline", error=str(e))
            raise
    
    async def search_tweets(
        self,
        query: str,
        max_results: int = 10,
        next_token: Optional[str] = None,
        sort_order: str = "relevancy"
    ) -> Dict[str, Any]:
        """Search for tweets"""
        status = await self.rate_limiter.is_allowed("twitter", "search")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'query': query,
                'max_results': min(max_results, 100),
                'sort_order': sort_order,
                'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text',
                'expansions': 'author_id,attachments.media_keys,referenced_tweets.id',
                'user.fields': 'created_at,description,location,profile_image_url,protected,public_metrics,url,username,verified'
            }
            
            if next_token:
                params['next_token'] = next_token
            
            response = await self._make_request(
                method="GET",
                endpoint="/tweets/search/recent",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "search", True)
            
            result = response.json()
            tweets = [TwitterTweet.from_api_response(tweet_data) for tweet_data in result.get('data', [])]
            
            return {
                'tweets': tweets,
                'next_token': result.get('meta', {}).get('next_token'),
                'result_count': result.get('meta', {}).get('result_count', 0),
                'includes': result.get('includes', {})
            }
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "search", False)
            logger.error(f"Failed to search tweets", query=query, error=str(e))
            raise
    
    async def get_bookmarks(
        self,
        max_results: int = 10,
        pagination_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get user's bookmarked tweets"""
        status = await self.rate_limiter.is_allowed("twitter", "bookmarks", user_id=self._authenticated_user.id if self._authenticated_user else None)
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            user_id = self._authenticated_user.id if self._authenticated_user else 'me'
            
            params = {
                'max_results': min(max_results, 100),
                'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text',
                'expansions': 'author_id,attachments.media_keys,referenced_tweets.id',
                'user.fields': 'created_at,description,location,profile_image_url,protected,public_metrics,url,username,verified'
            }
            
            if pagination_token:
                params['pagination_token'] = pagination_token
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/users/{user_id}/bookmarks",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "bookmarks", True, user_id=self._authenticated_user.id if self._authenticated_user else None)
            
            result = response.json()
            tweets = [TwitterTweet.from_api_response(tweet_data) for tweet_data in result.get('data', [])]
            
            return {
                'tweets': tweets,
                'next_token': result.get('meta', {}).get('next_token'),
                'result_count': result.get('meta', {}).get('result_count', 0),
                'includes': result.get('includes', {})
            }
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "bookmarks", False, user_id=self._authenticated_user.id if self._authenticated_user else None)
            logger.error(f"Failed to get bookmarks", error=str(e))
            raise
    
    async def get_liked_tweets(
        self,
        user_id: Optional[str] = None,
        max_results: int = 10,
        pagination_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get tweets liked by user"""
        if not user_id:
            user_id = self._authenticated_user.id if self._authenticated_user else 'me'
        
        status = await self.rate_limiter.is_allowed("twitter", "likes", user_id=user_id)
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'max_results': min(max_results, 100),
                'tweet.fields': 'attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text',
                'expansions': 'author_id,attachments.media_keys,referenced_tweets.id',
                'user.fields': 'created_at,description,location,profile_image_url,protected,public_metrics,url,username,verified'
            }
            
            if pagination_token:
                params['pagination_token'] = pagination_token
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/users/{user_id}/liked_tweets",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "likes", True, user_id=user_id)
            
            result = response.json()
            tweets = [TwitterTweet.from_api_response(tweet_data) for tweet_data in result.get('data', [])]
            
            return {
                'tweets': tweets,
                'next_token': result.get('meta', {}).get('next_token'),
                'result_count': result.get('meta', {}).get('result_count', 0),
                'includes': result.get('includes', {})
            }
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "likes", False, user_id=user_id)
            logger.error(f"Failed to get liked tweets", user_id=user_id, error=str(e))
            raise
    
    async def get_following(
        self,
        user_id: Optional[str] = None,
        max_results: int = 100,
        pagination_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get users that a user is following"""
        if not user_id:
            user_id = self._authenticated_user.id if self._authenticated_user else 'me'
        
        status = await self.rate_limiter.is_allowed("twitter", "following", user_id=user_id)
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'max_results': min(max_results, 1000),
                'user.fields': 'created_at,description,location,profile_image_url,protected,public_metrics,url,username,verified'
            }
            
            if pagination_token:
                params['pagination_token'] = pagination_token
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/users/{user_id}/following",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "following", True, user_id=user_id)
            
            result = response.json()
            users = [TwitterUser.from_api_response(user_data) for user_data in result.get('data', [])]
            
            return {
                'users': users,
                'next_token': result.get('meta', {}).get('next_token'),
                'result_count': result.get('meta', {}).get('result_count', 0)
            }
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "following", False, user_id=user_id)
            logger.error(f"Failed to get following", user_id=user_id, error=str(e))
            raise
    
    async def get_followers(
        self,
        user_id: Optional[str] = None,
        max_results: int = 100,
        pagination_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get followers of a user"""
        if not user_id:
            user_id = self._authenticated_user.id if self._authenticated_user else 'me'
        
        status = await self.rate_limiter.is_allowed("twitter", "followers", user_id=user_id)
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'max_results': min(max_results, 1000),
                'user.fields': 'created_at,description,location,profile_image_url,protected,public_metrics,url,username,verified'
            }
            
            if pagination_token:
                params['pagination_token'] = pagination_token
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/users/{user_id}/followers",
                params=params
            )
            
            await self.rate_limiter.record_response("twitter", "followers", True, user_id=user_id)
            
            result = response.json()
            users = [TwitterUser.from_api_response(user_data) for user_data in result.get('data', [])]
            
            return {
                'users': users,
                'next_token': result.get('meta', {}).get('next_token'),
                'result_count': result.get('meta', {}).get('result_count', 0)
            }
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "followers", False, user_id=user_id)
            logger.error(f"Failed to get followers", user_id=user_id, error=str(e))
            raise
    
    async def post_tweet(
        self,
        text: str,
        reply_to_tweet_id: Optional[str] = None,
        media_ids: Optional[List[str]] = None,
        poll_options: Optional[List[str]] = None,
        poll_duration_minutes: int = 1440  # 24 hours
    ) -> TwitterTweet:
        """Post a new tweet"""
        # Note: Requires write permissions
        status = await self.rate_limiter.is_allowed("twitter", "post_tweet", user_id=self._authenticated_user.id if self._authenticated_user else None)
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            tweet_data = {
                'text': text
            }
            
            if reply_to_tweet_id:
                tweet_data['reply'] = {'in_reply_to_tweet_id': reply_to_tweet_id}
            
            if media_ids:
                tweet_data['media'] = {'media_ids': media_ids}
            
            if poll_options:
                tweet_data['poll'] = {
                    'options': poll_options,
                    'duration_minutes': poll_duration_minutes
                }
            
            response = await self._make_request(
                method="POST",
                endpoint="/tweets",
                data=tweet_data
            )
            
            await self.rate_limiter.record_response("twitter", "post_tweet", True, user_id=self._authenticated_user.id if self._authenticated_user else None)
            
            return TwitterTweet.from_api_response(response.json()['data'])
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "post_tweet", False, user_id=self._authenticated_user.id if self._authenticated_user else None)
            logger.error(f"Failed to post tweet", error=str(e))
            raise
    
    async def delete_tweet(self, tweet_id: str) -> bool:
        """Delete a tweet"""
        # Note: Requires write permissions
        status = await self.rate_limiter.is_allowed("twitter", "delete_tweet", user_id=self._authenticated_user.id if self._authenticated_user else None)
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            response = await self._make_request(
                method="DELETE",
                endpoint=f"/tweets/{tweet_id}"
            )
            
            await self.rate_limiter.record_response("twitter", "delete_tweet", True, user_id=self._authenticated_user.id if self._authenticated_user else None)
            
            result = response.json()
            return result.get('data', {}).get('deleted', False)
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "delete_tweet", False, user_id=self._authenticated_user.id if self._authenticated_user else None)
            logger.error(f"Failed to delete tweet", tweet_id=tweet_id, error=str(e))
            return False
    
    async def upload_media(self, media_path: Path, alt_text: Optional[str] = None) -> str:
        """Upload media for tweets"""
        # Note: Uses Twitter API v1.1 for media upload
        status = await self.rate_limiter.is_allowed("twitter", "upload_media", user_id=self._authenticated_user.id if self._authenticated_user else None)
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            # Read media file
            with open(media_path, 'rb') as f:
                media_data = f.read()
            
            # Upload media using v1.1 endpoint
            files = {'media': (media_path.name, media_data)}
            
            response = await self._make_request(
                method="POST",
                endpoint="https://upload.twitter.com/1.1/media/upload.json",
                files=files,
                authenticated=True
            )
            
            result = response.json()
            media_id = result['media_id_string']
            
            # Add alt text if provided
            if alt_text:
                await self._add_media_metadata(media_id, alt_text)
            
            await self.rate_limiter.record_response("twitter", "upload_media", True, user_id=self._authenticated_user.id if self._authenticated_user else None)
            
            return media_id
            
        except Exception as e:
            await self.rate_limiter.record_response("twitter", "upload_media", False, user_id=self._authenticated_user.id if self._authenticated_user else None)
            logger.error(f"Failed to upload media", media_path=str(media_path), error=str(e))
            raise
    
    async def _add_media_metadata(self, media_id: str, alt_text: str):
        """Add metadata to uploaded media"""
        metadata = {
            'media_id': media_id,
            'alt_text': {'text': alt_text}
        }
        
        await self._make_request(
            method="POST",
            endpoint="https://upload.twitter.com/1.1/media/metadata/create.json",
            data=metadata
        )
    
    async def export_data(
        self,
        output_path: Path,
        data_types: List[str] = None,
        max_items: int = 1000
    ) -> int:
        """Export Twitter data"""
        if data_types is None:
            data_types = ['tweets', 'likes', 'bookmarks', 'following']
        
        logger.info(f"Starting Twitter export", output_path=str(output_path), data_types=data_types)
        
        output_path.mkdir(parents=True, exist_ok=True)
        exported_count = 0
        
        for data_type in data_types:
            try:
                output_file = output_path / f"{data_type}.json"
                
                if data_type == 'tweets':
                    if self._authenticated_user:
                        result = await self.get_user_tweets(
                            self._authenticated_user.id, max_results=max_items
                        )
                        data = [tweet.__dict__ for tweet in result['tweets']]
                    else:
                        continue
                elif data_type == 'likes':
                    result = await self.get_liked_tweets(max_results=max_items)
                    data = [tweet.__dict__ for tweet in result['tweets']]
                elif data_type == 'bookmarks':
                    result = await self.get_bookmarks(max_results=max_items)
                    data = [tweet.__dict__ for tweet in result['tweets']]
                elif data_type == 'following':
                    result = await self.get_following(max_results=max_items)
                    data = [user.__dict__ for user in result['users']]
                else:
                    logger.warning(f"Unknown data type: {data_type}")
                    continue
                
                # Convert datetime objects to ISO strings for JSON serialization
                def serialize_datetime(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return obj
                
                serialized_data = json.loads(json.dumps(data, default=serialize_datetime))
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(serialized_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Exported {data_type}", count=len(data), output_file=str(output_file))
                exported_count += len(data)
                
            except Exception as e:
                logger.error(f"Failed to export {data_type}", error=str(e))
                continue
        
        logger.info(f"Twitter export completed", exported_count=exported_count)
        return exported_count