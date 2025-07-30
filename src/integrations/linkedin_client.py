"""
LinkedIn API Client Implementation

This module provides comprehensive LinkedIn API integration with:
- OAuth2 authentication
- Profile data access and management
- Post creation and management
- Connection management
- Company page operations (if applicable)
- Analytics data retrieval
- Professional network insights
- Proper scope management
- Compliance with LinkedIn's API usage policies
"""

import asyncio
import json
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog

from .base_client import BaseAPIClient, RateLimitConfig, CircuitBreakerConfig
from ..config.auth import AuthProvider, TokenType
from ..utils.rate_limiter import get_rate_limit_manager

logger = structlog.get_logger(__name__)


@dataclass
class LinkedInProfile:
    """LinkedIn profile data structure"""
    id: str
    first_name: str
    last_name: str
    email_address: Optional[str] = None
    headline: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    industry: Optional[str] = None
    positions: List[Dict[str, Any]] = field(default_factory=list)
    educations: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    profile_picture: Optional[str] = None
    public_profile_url: Optional[str] = None
    num_connections: Optional[int] = None
    num_connections_capped: bool = False
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'LinkedInProfile':
        """Create LinkedInProfile from API response"""
        return cls(
            id=data['id'],
            first_name=data.get('firstName', {}).get('localized', {}).get('en_US', ''),
            last_name=data.get('lastName', {}).get('localized', {}).get('en_US', ''),
            email_address=data.get('emailAddress'),
            headline=data.get('headline', {}).get('localized', {}).get('en_US'),
            summary=data.get('summary', {}).get('localized', {}).get('en_US'),
            location=data.get('location'),
            industry=data.get('industry'),
            positions=data.get('positions', {}).get('values', []),
            educations=data.get('educations', {}).get('values', []),
            skills=data.get('skills', {}).get('values', []),
            profile_picture=data.get('profilePicture', {}).get('displayImage'),
            public_profile_url=data.get('publicProfileUrl'),
            num_connections=data.get('numConnections'),
            num_connections_capped=data.get('numConnectionsCapped', False)
        )


@dataclass
class LinkedInPost:
    """LinkedIn post data structure"""
    id: str
    author: str
    created_time: datetime
    last_modified_time: datetime
    text: Optional[str] = None
    visibility: str = "PUBLIC"
    lifecycle_state: str = "PUBLISHED"
    specific_content: Dict[str, Any] = field(default_factory=dict)
    activity: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'LinkedInPost':
        """Create LinkedInPost from API response"""
        created_time = datetime.fromtimestamp(
            data['createdAt'] / 1000, timezone.utc
        ) if 'createdAt' in data else datetime.now(timezone.utc)
        
        last_modified_time = datetime.fromtimestamp(
            data['lastModifiedAt'] / 1000, timezone.utc
        ) if 'lastModifiedAt' in data else created_time
        
        # Extract text from specific content
        text = None
        specific_content = data.get('specificContent', {})
        if 'com.linkedin.ugc.ShareContent' in specific_content:
            share_content = specific_content['com.linkedin.ugc.ShareContent']
            share_commentary = share_content.get('shareCommentary', {})
            text = share_commentary.get('text')
        
        return cls(
            id=data['id'],
            author=data.get('author', ''),
            created_time=created_time,
            last_modified_time=last_modified_time,
            text=text,
            visibility=data.get('visibility', {}).get('com.linkedin.ugc.MemberNetworkVisibility', 'PUBLIC'),
            lifecycle_state=data.get('lifecycleState', 'PUBLISHED'),
            specific_content=specific_content,
            activity=data.get('activity', {})
        )


@dataclass
class LinkedInConnection:
    """LinkedIn connection data structure"""
    id: str
    first_name: str
    last_name: str
    headline: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    profile_picture: Optional[str] = None
    public_profile_url: Optional[str] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'LinkedInConnection':
        """Create LinkedInConnection from API response"""
        return cls(
            id=data['id'],
            first_name=data.get('firstName', {}).get('localized', {}).get('en_US', ''),
            last_name=data.get('lastName', {}).get('localized', {}).get('en_US', ''),
            headline=data.get('headline', {}).get('localized', {}).get('en_US'),
            location=data.get('location', {}).get('name'),
            industry=data.get('industry'),
            profile_picture=data.get('profilePicture', {}).get('displayImage'),
            public_profile_url=data.get('publicProfileUrl')
        )


@dataclass
class LinkedInCompany:
    """LinkedIn company data structure"""
    id: str
    name: str
    universal_name: Optional[str] = None
    description: Optional[str] = None
    website_url: Optional[str] = None
    industry: Optional[str] = None
    company_type: Optional[str] = None
    headquarters: Optional[Dict[str, Any]] = None
    logo: Optional[str] = None
    employee_count_range: Optional[Dict[str, Any]] = None
    founded: Optional[int] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'LinkedInCompany':
        """Create LinkedInCompany from API response"""
        return cls(
            id=str(data['id']),
            name=data.get('name', {}).get('localized', {}).get('en_US', ''),
            universal_name=data.get('universalName'),
            description=data.get('description', {}).get('localized', {}).get('en_US'),
            website_url=data.get('websiteUrl'),
            industry=data.get('industry'),
            company_type=data.get('companyType'),
            headquarters=data.get('headquarters'),
            logo=data.get('logo', {}).get('original'),
            employee_count_range=data.get('employeeCountRange'),
            founded=data.get('foundedOn', {}).get('year')
        )


class LinkedInClient(BaseAPIClient):
    """LinkedIn API client with comprehensive functionality"""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        scopes: Optional[List[str]] = None,
        **kwargs
    ):
        # LinkedIn-specific rate limiting (more restrictive)
        rate_limit_config = RateLimitConfig(
            requests_per_minute=100,  # Conservative limit
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_size=3
        )
        
        super().__init__(
            provider=AuthProvider.LINKEDIN,
            base_url="https://api.linkedin.com/v2",
            rate_limit_config=rate_limit_config,
            **kwargs
        )
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes or [
            'r_liteprofile',
            'r_emailaddress',
            'w_member_social'  # Only for reading saved posts
        ]
        
        self.rate_limiter = get_rate_limit_manager()
        
        # User context (filled after authentication)
        self._authenticated_user: Optional[LinkedInProfile] = None
    
    def _format_auth_header(self, token: str) -> Dict[str, str]:
        """Format authentication header for LinkedIn API"""
        return {"Authorization": f"Bearer {token}"}
    
    async def authenticate(self, redirect_uri: str = "http://localhost:8080/oauth/callback") -> bool:
        """Authenticate with LinkedIn using OAuth2"""
        try:
            # Build authorization URL
            auth_params = {
                'response_type': 'code',
                'client_id': self.client_id,
                'redirect_uri': redirect_uri,
                'scope': ' '.join(self.scopes),
                'state': self.auth_manager.create_session_state(
                    provider=self.provider,
                    redirect_uri=redirect_uri,
                    scopes=self.scopes
                )
            }
            
            auth_url = "https://www.linkedin.com/oauth/v2/authorization?" + urllib.parse.urlencode(auth_params)
            
            logger.info(
                "LinkedIn OAuth flow initiated",
                auth_url=auth_url,
                state=auth_params['state'][:8] + "...",
                scopes=self.scopes
            )
            
            raise Exception(f"Please visit this URL to authorize: {auth_url}")
            
        except Exception as e:
            logger.error(f"LinkedIn authentication failed", error=str(e))
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
                'code': authorization_code,
                'redirect_uri': session_data["redirect_uri"],
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = await self._make_request(
                method="POST",
                endpoint="https://www.linkedin.com/oauth/v2/accessToken",
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
                expires_in=token_response.get('expires_in', 5184000),  # Default ~60 days
                scopes=self.scopes,
                subject=None,  # Will be populated after getting user info
                client_id=self.client_id
            )
            
            # LinkedIn typically doesn't provide refresh tokens
            # in the standard OAuth flow
            
            # Get authenticated user info
            await self._load_authenticated_user()
            
            logger.info(
                "LinkedIn authentication successful",
                access_token_id=access_token_id,
                user_id=self._authenticated_user.id if self._authenticated_user else None,
                scopes=self.scopes
            )
            
            return True
            
        except Exception as e:
            logger.error(f"LinkedIn OAuth callback failed", error=str(e))
            return False
    
    async def refresh_token(self) -> bool:
        """Refresh LinkedIn access token"""
        # LinkedIn typically doesn't support token refresh in the standard flow
        # Users need to re-authenticate when tokens expire
        logger.warning("LinkedIn token refresh not supported - re-authentication required")
        return False
    
    async def _load_authenticated_user(self):
        """Load authenticated user information"""
        try:
            profile_data = await self.get_profile()
            self._authenticated_user = LinkedInProfile.from_api_response(profile_data)
        except Exception as e:
            logger.warning(f"Failed to load authenticated user", error=str(e))
    
    async def get_rate_limits(self) -> Dict[str, Any]:
        """Get current rate limit status for LinkedIn"""
        status = await self.rate_limiter.get_all_status()
        return status.get("linkedin", {})
    
    async def get_profile(self, person_id: str = "~") -> Dict[str, Any]:
        """Get LinkedIn profile information"""
        status = await self.rate_limiter.is_allowed("linkedin", "profile")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            # Build projection to specify fields
            projection = "(id,firstName,lastName,headline,summary,location,industry,positions,educations,skills,profilePicture(displayImage~:playableStreams),publicProfileUrl,numConnections,numConnectionsCapped)"
            
            response = await self._make_request(
                method="GET",
                endpoint=f"/people/{person_id}",
                params={'projection': projection}
            )
            
            await self.rate_limiter.record_response("linkedin", "profile", True)
            
            return response.json()
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "profile", False)
            logger.error(f"Failed to get LinkedIn profile", person_id=person_id, error=str(e))
            raise
    
    async def get_email_address(self) -> Dict[str, Any]:
        """Get authenticated user's email address"""
        status = await self.rate_limiter.is_allowed("linkedin", "email")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            projection = "(elements*(handle~))"
            
            response = await self._make_request(
                method="GET",
                endpoint="/emailAddress",
                params={'q': 'members', 'projection': projection}
            )
            
            await self.rate_limiter.record_response("linkedin", "email", True)
            
            result = response.json()
            elements = result.get('elements', [])
            
            if elements:
                email_info = elements[0].get('handle~', {})
                return {
                    'email_address': email_info.get('emailAddress'),
                    'primary': True
                }
            
            return {}
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "email", False)
            logger.error(f"Failed to get LinkedIn email", error=str(e))
            raise
    
    async def get_posts(
        self,
        author: str = "~",
        max_results: int = 20,
        start: int = 0
    ) -> List[LinkedInPost]:
        """Get posts by author"""
        status = await self.rate_limiter.is_allowed("linkedin", "posts")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'q': 'authors',
                'authors': f'urn:li:person:{author}' if author != "~" else 'urn:li:person:~',
                'count': min(max_results, 50),  # LinkedIn limit
                'start': start,
                'sortBy': 'LAST_MODIFIED'
            }
            
            response = await self._make_request(
                method="GET",
                endpoint="/ugcPosts",
                params=params
            )
            
            await self.rate_limiter.record_response("linkedin", "posts", True)
            
            result = response.json()
            posts = [LinkedInPost.from_api_response(post_data) for post_data in result.get('elements', [])]
            
            return posts
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "posts", False)
            logger.error(f"Failed to get LinkedIn posts", author=author, error=str(e))
            raise
    
    async def get_post(self, post_id: str) -> LinkedInPost:
        """Get a specific post"""
        status = await self.rate_limiter.is_allowed("linkedin", "posts")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/ugcPosts/{post_id}"
            )
            
            await self.rate_limiter.record_response("linkedin", "posts", True)
            
            return LinkedInPost.from_api_response(response.json())
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "posts", False)
            logger.error(f"Failed to get LinkedIn post", post_id=post_id, error=str(e))
            raise
    
    async def create_post(
        self,
        text: str,
        visibility: str = "PUBLIC",
        media_urls: Optional[List[str]] = None
    ) -> LinkedInPost:
        """Create a new LinkedIn post"""
        # Note: Requires write permissions
        status = await self.rate_limiter.is_allowed("linkedin", "create_post")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            # Build post data
            post_data = {
                'author': f'urn:li:person:{self._authenticated_user.id}' if self._authenticated_user else 'urn:li:person:~',
                'lifecycleState': 'PUBLISHED',
                'specificContent': {
                    'com.linkedin.ugc.ShareContent': {
                        'shareCommentary': {
                            'text': text
                        },
                        'shareMediaCategory': 'NONE'
                    }
                },
                'visibility': {
                    'com.linkedin.ugc.MemberNetworkVisibility': visibility
                }
            }
            
            # Add media if provided
            if media_urls:
                post_data['specificContent']['com.linkedin.ugc.ShareContent']['shareMediaCategory'] = 'IMAGE'
                post_data['specificContent']['com.linkedin.ugc.ShareContent']['media'] = [
                    {
                        'status': 'READY',
                        'description': {
                            'text': 'Shared image'
                        },
                        'media': url,
                        'title': {
                            'text': 'Image'
                        }
                    }
                    for url in media_urls
                ]
            
            response = await self._make_request(
                method="POST",
                endpoint="/ugcPosts",
                data=post_data
            )
            
            await self.rate_limiter.record_response("linkedin", "create_post", True)
            
            return LinkedInPost.from_api_response(response.json())
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "create_post", False)
            logger.error(f"Failed to create LinkedIn post", error=str(e))
            raise
    
    async def delete_post(self, post_id: str) -> bool:
        """Delete a LinkedIn post"""
        # Note: Requires write permissions
        status = await self.rate_limiter.is_allowed("linkedin", "delete_post")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            response = await self._make_request(
                method="DELETE",
                endpoint=f"/ugcPosts/{post_id}"
            )
            
            await self.rate_limiter.record_response("linkedin", "delete_post", True)
            
            return response.status_code == 204
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "delete_post", False)
            logger.error(f"Failed to delete LinkedIn post", post_id=post_id, error=str(e))
            return False
    
    async def get_connections(
        self,
        max_results: int = 100,
        start: int = 0
    ) -> List[LinkedInConnection]:
        """Get user's connections"""
        status = await self.rate_limiter.is_allowed("linkedin", "connections")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            projection = "(elements*(to~(id,firstName,lastName,headline,location,industry,profilePicture(displayImage~:playableStreams),publicProfileUrl)))"
            
            params = {
                'q': 'viewer',
                'start': start,
                'count': min(max_results, 500),  # LinkedIn limit
                'projection': projection
            }
            
            response = await self._make_request(
                method="GET",
                endpoint="/connections",
                params=params
            )
            
            await self.rate_limiter.record_response("linkedin", "connections", True)
            
            result = response.json()
            connections = []
            
            for element in result.get('elements', []):
                connection_data = element.get('to~', {})
                if connection_data:
                    connections.append(LinkedInConnection.from_api_response(connection_data))
            
            return connections
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "connections", False)
            logger.error(f"Failed to get LinkedIn connections", error=str(e))
            raise
    
    async def search_people(
        self,
        keywords: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        company: Optional[str] = None,
        title: Optional[str] = None,
        max_results: int = 25,
        start: int = 0
    ) -> List[Dict[str, Any]]:
        """Search for people on LinkedIn"""
        status = await self.rate_limiter.is_allowed("linkedin", "search")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'q': 'people',
                'start': start,
                'count': min(max_results, 50)  # LinkedIn limit
            }
            
            if keywords:
                params['keywords'] = keywords
            if first_name:
                params['firstName'] = first_name
            if last_name:
                params['lastName'] = last_name
            if company:
                params['company'] = company
            if title:
                params['title'] = title
            
            response = await self._make_request(
                method="GET",
                endpoint="/peopleSearch",
                params=params
            )
            
            await self.rate_limiter.record_response("linkedin", "search", True)
            
            result = response.json()
            return result.get('people', {}).get('values', [])
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "search", False)
            logger.error(f"Failed to search LinkedIn people", error=str(e))
            raise
    
    async def get_companies(
        self,
        company_ids: Optional[List[str]] = None,
        universal_names: Optional[List[str]] = None,
        max_results: int = 50
    ) -> List[LinkedInCompany]:
        """Get company information"""
        status = await self.rate_limiter.is_allowed("linkedin", "companies")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {}
            
            if company_ids:
                params['ids'] = ','.join(company_ids)
            elif universal_names:
                params['universalNames'] = ','.join(universal_names)
            else:
                # Get user's current company if no specific companies requested
                if self._authenticated_user and self._authenticated_user.positions:
                    current_position = self._authenticated_user.positions[0]  # Most recent
                    company_id = current_position.get('company', {}).get('id')
                    if company_id:
                        params['ids'] = str(company_id)
            
            if not params:
                return []
            
            response = await self._make_request(
                method="GET",
                endpoint="/companies",
                params=params
            )
            
            await self.rate_limiter.record_response("linkedin", "companies", True)
            
            result = response.json()
            companies = []
            
            for company_data in result.get('values', []):
                companies.append(LinkedInCompany.from_api_response(company_data))
            
            return companies[:max_results]
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "companies", False)
            logger.error(f"Failed to get LinkedIn companies", error=str(e))
            raise
    
    async def get_company_updates(
        self,
        company_id: str,
        max_results: int = 20,
        start: int = 0
    ) -> List[Dict[str, Any]]:
        """Get company updates/posts"""
        status = await self.rate_limiter.is_allowed("linkedin", "company_updates")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'company': company_id,
                'start': start,
                'count': min(max_results, 50)
            }
            
            response = await self._make_request(
                method="GET",
                endpoint="/companyUpdates",
                params=params
            )
            
            await self.rate_limiter.record_response("linkedin", "company_updates", True)
            
            result = response.json()
            return result.get('values', [])
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "company_updates", False)
            logger.error(f"Failed to get company updates", company_id=company_id, error=str(e))
            raise
    
    async def get_analytics(
        self,
        post_id: Optional[str] = None,
        time_range: str = "LAST_30_DAYS"
    ) -> Dict[str, Any]:
        """Get analytics data for posts or profile"""
        status = await self.rate_limiter.is_allowed("linkedin", "analytics")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if post_id:
                # Get post-specific analytics
                response = await self._make_request(
                    method="GET",
                    endpoint=f"/socialActions/{post_id}",
                    params={'projection': '(numComments,numLikes,numShares)'}
                )
            else:
                # Get profile analytics (if available)
                params = {
                    'q': 'company',
                    'company': f'urn:li:organization:{self._authenticated_user.id}' if self._authenticated_user else '',
                    'timeRange': time_range
                }
                
                response = await self._make_request(
                    method="GET",
                    endpoint="/networkSizes",
                    params=params
                )
            
            await self.rate_limiter.record_response("linkedin", "analytics", True)
            
            return response.json()
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "analytics", False)
            logger.error(f"Failed to get LinkedIn analytics", post_id=post_id, error=str(e))
            raise
    
    async def send_invitation(
        self,
        person_id: str,
        message: Optional[str] = None
    ) -> bool:
        """Send connection invitation"""
        # Note: Requires write permissions and may have restrictions
        status = await self.rate_limiter.is_allowed("linkedin", "invitations")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            invitation_data = {
                'recipients': [f'urn:li:person:{person_id}'],
                'subject': 'I would like to connect with you',
                'body': message or 'I would like to add you to my professional network on LinkedIn.'
            }
            
            response = await self._make_request(
                method="POST",
                endpoint="/invitations",
                data=invitation_data
            )
            
            await self.rate_limiter.record_response("linkedin", "invitations", True)
            
            return response.status_code == 201
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "invitations", False)
            logger.error(f"Failed to send LinkedIn invitation", person_id=person_id, error=str(e))
            return False
    
    async def export_data(
        self,
        output_path: Path,
        data_types: List[str] = None,
        max_items: int = 500
    ) -> int:
        """Export LinkedIn data"""
        if data_types is None:
            data_types = ['profile', 'posts', 'connections']
        
        logger.info(f"Starting LinkedIn export", output_path=str(output_path), data_types=data_types)
        
        output_path.mkdir(parents=True, exist_ok=True)
        exported_count = 0
        
        for data_type in data_types:
            try:
                output_file = output_path / f"{data_type}.json"
                
                if data_type == 'profile':
                    # Export profile information
                    profile_data = await self.get_profile()
                    email_data = await self.get_email_address()
                    
                    combined_data = {
                        'profile': profile_data,
                        'email': email_data
                    }
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(combined_data, f, indent=2, ensure_ascii=False)
                    
                    exported_count += 1
                    
                elif data_type == 'posts':
                    # Export user's posts
                    posts = await self.get_posts(max_results=max_items)
                    posts_data = [post.__dict__ for post in posts]
                    
                    # Convert datetime objects to ISO strings
                    def serialize_datetime(obj):
                        if isinstance(obj, datetime):
                            return obj.isoformat()
                        return obj
                    
                    serialized_data = json.loads(json.dumps(posts_data, default=serialize_datetime))
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(serialized_data, f, indent=2, ensure_ascii=False)
                    
                    exported_count += len(posts)
                    
                elif data_type == 'connections':
                    # Export connections
                    connections = await self.get_connections(max_results=max_items)
                    connections_data = [conn.__dict__ for conn in connections]
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(connections_data, f, indent=2, ensure_ascii=False)
                    
                    exported_count += len(connections)
                    
                else:
                    logger.warning(f"Unknown data type: {data_type}")
                    continue
                
                logger.info(f"Exported {data_type}", output_file=str(output_file))
                
            except Exception as e:
                logger.error(f"Failed to export {data_type}", error=str(e))
                continue
        
        logger.info(f"LinkedIn export completed", exported_count=exported_count)
        return exported_count
    
    async def get_saved_posts(
        self,
        max_results: int = 50,
        start: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user's saved posts/articles"""
        # Note: This functionality may be limited or require special permissions
        status = await self.rate_limiter.is_allowed("linkedin", "saved_posts")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            params = {
                'q': 'saved',
                'start': start,
                'count': min(max_results, 50)
            }
            
            response = await self._make_request(
                method="GET",
                endpoint="/posts",
                params=params
            )
            
            await self.rate_limiter.record_response("linkedin", "saved_posts", True)
            
            result = response.json()
            return result.get('elements', [])
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "saved_posts", False)
            logger.error(f"Failed to get saved posts", error=str(e))
            # Return empty list if feature not available
            return []
    
    async def get_network_insights(self) -> Dict[str, Any]:
        """Get professional network insights"""
        status = await self.rate_limiter.is_allowed("linkedin", "insights")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            # Get basic network statistics
            insights = {}
            
            # Profile data
            if self._authenticated_user:
                insights['profile'] = {
                    'num_connections': self._authenticated_user.num_connections,
                    'num_connections_capped': self._authenticated_user.num_connections_capped,
                    'industry': self._authenticated_user.industry,
                    'location': self._authenticated_user.location
                }
            
            # Recent posts performance
            posts = await self.get_posts(max_results=10)
            if posts:
                insights['recent_posts'] = {
                    'count': len(posts),
                    'latest_post_date': posts[0].created_time.isoformat() if posts else None
                }
            
            # Connections data
            connections = await self.get_connections(max_results=100)
            if connections:
                industries = {}
                locations = {}
                
                for conn in connections:
                    if conn.industry:
                        industries[conn.industry] = industries.get(conn.industry, 0) + 1
                    if conn.location:
                        locations[conn.location] = locations.get(conn.location, 0) + 1
                
                insights['network_analysis'] = {
                    'top_industries': sorted(industries.items(), key=lambda x: x[1], reverse=True)[:5],
                    'top_locations': sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5],
                    'total_analyzed': len(connections)
                }
            
            await self.rate_limiter.record_response("linkedin", "insights", True)
            
            return insights
            
        except Exception as e:
            await self.rate_limiter.record_response("linkedin", "insights", False)
            logger.error(f"Failed to get network insights", error=str(e))
            raise