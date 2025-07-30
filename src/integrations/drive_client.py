"""
Google Drive API Client Implementation

This module provides comprehensive Google Drive API integration with:
- OAuth2 authentication (shared with Gmail)
- File search, read, write, delete operations
- Folder management and organization
- Sharing and permissions management
- Version control and revision history
- Metadata extraction and indexing
- Support for various file types and formats
- Resumable uploads for large files
- Integration with encryption utilities
"""

import asyncio
import io
import json
import mimetypes
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, AsyncIterator, BinaryIO
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload, MediaIoBaseUpload

from .base_client import BaseAPIClient, RateLimitConfig, CircuitBreakerConfig
from ..config.auth import AuthProvider, TokenType
from ..utils.rate_limiter import get_rate_limit_manager

logger = structlog.get_logger(__name__)


@dataclass
class DriveFile:
    """Google Drive file data structure"""
    id: str
    name: str
    mime_type: str
    parents: List[str] = field(default_factory=list)
    size: Optional[int] = None
    created_time: Optional[datetime] = None
    modified_time: Optional[datetime] = None
    version: Optional[str] = None
    trashed: bool = False
    starred: bool = False
    shared: bool = False
    owners: List[Dict[str, Any]] = field(default_factory=list)
    permissions: List[Dict[str, Any]] = field(default_factory=list)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    properties: Dict[str, str] = field(default_factory=dict)
    app_properties: Dict[str, str] = field(default_factory=dict)
    web_view_link: Optional[str] = None
    web_content_link: Optional[str] = None
    thumbnail_link: Optional[str] = None
    icon_link: Optional[str] = None
    description: Optional[str] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'DriveFile':
        """Create DriveFile from API response"""
        # Parse timestamps
        created_time = None
        if 'createdTime' in data:
            created_time = datetime.fromisoformat(
                data['createdTime'].replace('Z', '+00:00')
            )
        
        modified_time = None
        if 'modifiedTime' in data:
            modified_time = datetime.fromisoformat(
                data['modifiedTime'].replace('Z', '+00:00')
            )
        
        return cls(
            id=data['id'],
            name=data['name'],
            mime_type=data.get('mimeType', ''),
            parents=data.get('parents', []),
            size=int(data['size']) if data.get('size') else None,
            created_time=created_time,
            modified_time=modified_time,
            version=data.get('version'),
            trashed=data.get('trashed', False),
            starred=data.get('starred', False),
            shared=data.get('shared', False),
            owners=data.get('owners', []),
            permissions=data.get('permissions', []),
            capabilities=data.get('capabilities', {}),
            properties=data.get('properties', {}),
            app_properties=data.get('appProperties', {}),
            web_view_link=data.get('webViewLink'),
            web_content_link=data.get('webContentLink'),
            thumbnail_link=data.get('thumbnailLink'),
            icon_link=data.get('iconLink'),
            description=data.get('description')
        )


@dataclass
class DrivePermission:
    """Google Drive permission data structure"""
    id: str
    type: str  # user, group, domain, anyone
    role: str  # owner, organizer, fileOrganizer, writer, commenter, reader
    email_address: Optional[str] = None
    display_name: Optional[str] = None
    photo_link: Optional[str] = None
    domain: Optional[str] = None
    allow_file_discovery: bool = True
    expiration_time: Optional[datetime] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'DrivePermission':
        """Create DrivePermission from API response"""
        expiration_time = None
        if 'expirationTime' in data:
            expiration_time = datetime.fromisoformat(
                data['expirationTime'].replace('Z', '+00:00')
            )
        
        return cls(
            id=data['id'],
            type=data['type'],
            role=data['role'],
            email_address=data.get('emailAddress'),
            display_name=data.get('displayName'),
            photo_link=data.get('photoLink'),
            domain=data.get('domain'),
            allow_file_discovery=data.get('allowFileDiscovery', True),
            expiration_time=expiration_time
        )


@dataclass
class DriveRevision:
    """Google Drive file revision data structure"""
    id: str
    mime_type: str
    modified_time: datetime
    size: Optional[int] = None
    original_filename: Optional[str] = None
    md5_checksum: Optional[str] = None
    keep_forever: bool = False
    published: bool = False
    published_link: Optional[str] = None
    publish_auto: bool = False
    last_modifying_user: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'DriveRevision':
        """Create DriveRevision from API response"""
        modified_time = datetime.fromisoformat(
            data['modifiedTime'].replace('Z', '+00:00')
        )
        
        return cls(
            id=data['id'],
            mime_type=data['mimeType'],
            modified_time=modified_time,
            size=int(data['size']) if data.get('size') else None,
            original_filename=data.get('originalFilename'),
            md5_checksum=data.get('md5Checksum'),
            keep_forever=data.get('keepForever', False),
            published=data.get('published', False),
            published_link=data.get('publishedLink'),
            publish_auto=data.get('publishAuto', False),
            last_modifying_user=data.get('lastModifyingUser')
        )


class DriveClient(BaseAPIClient):
    """Google Drive API client with full functionality"""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        scopes: Optional[List[str]] = None,
        **kwargs
    ):
        # Drive-specific rate limiting
        rate_limit_config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=6000,
            requests_per_day=100000,
            burst_size=5
        )
        
        super().__init__(
            provider=AuthProvider.GOOGLE,
            base_url="https://www.googleapis.com/drive/v3",
            rate_limit_config=rate_limit_config,
            **kwargs
        )
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes or [
            'https://www.googleapis.com/auth/drive.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]
        
        self.rate_limiter = get_rate_limit_manager()
        
        # Google API client (will be initialized after authentication)
        self._drive_service = None
        
        # Supported export formats for Google Workspace files
        self.export_formats = {
            'application/vnd.google-apps.document': {
                'pdf': 'application/pdf',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'txt': 'text/plain',
                'html': 'text/html'
            },
            'application/vnd.google-apps.spreadsheet': {
                'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'csv': 'text/csv',
                'pdf': 'application/pdf'
            },
            'application/vnd.google-apps.presentation': {
                'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'pdf': 'application/pdf',
                'txt': 'text/plain'
            }
        }
    
    def _format_auth_header(self, token: str) -> Dict[str, str]:
        """Format authentication header for Google APIs"""
        return {"Authorization": f"Bearer {token}"}
    
    async def authenticate(self, redirect_uri: str = "http://localhost:8080/oauth/callback") -> bool:
        """Authenticate with Google Drive using OAuth2"""
        # Similar to Gmail authentication - would use shared Google OAuth flow
        try:
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [redirect_uri]
                    }
                },
                scopes=self.scopes
            )
            flow.redirect_uri = redirect_uri
            
            auth_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            session_data = self.auth_manager.create_session_state(
                provider=self.provider,
                redirect_uri=redirect_uri,
                scopes=self.scopes
            )
            
            logger.info(
                "Google Drive OAuth flow initiated",
                auth_url=auth_url,
                state=state[:8] + "...",
                scopes=self.scopes
            )
            
            raise Exception(f"Please visit this URL to authorize: {auth_url}")
            
        except Exception as e:
            logger.error(f"Drive authentication failed", error=str(e))
            return False
    
    async def handle_oauth_callback(self, authorization_code: str, state: str) -> bool:
        """Handle OAuth callback with authorization code"""
        # Similar implementation to Gmail - shared Google OAuth
        try:
            session_data = self.auth_manager.consume_session_state(state)
            if not session_data:
                raise ValueError("Invalid or expired OAuth state")
            
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [session_data["redirect_uri"]]
                    }
                },
                scopes=self.scopes
            )
            flow.redirect_uri = session_data["redirect_uri"]
            
            flow.fetch_token(code=authorization_code)
            credentials = flow.credentials
            
            access_token_id = await self.auth_manager.store_token(
                provider=self.provider,
                token_type=TokenType.ACCESS_TOKEN,
                token_value=credentials.token,
                expires_in=3600,
                scopes=self.scopes,
                subject=None,
                client_id=self.client_id
            )
            
            if credentials.refresh_token:
                await self.auth_manager.store_token(
                    provider=self.provider,
                    token_type=TokenType.REFRESH_TOKEN,
                    token_value=credentials.refresh_token,
                    expires_in=None,
                    scopes=self.scopes,
                    subject=None,
                    client_id=self.client_id
                )
            
            await self._initialize_service()
            
            logger.info("Drive authentication successful", access_token_id=access_token_id)
            return True
            
        except Exception as e:
            logger.error(f"Drive OAuth callback failed", error=str(e))
            return False
    
    async def refresh_token(self) -> bool:
        """Refresh Google Drive access token"""
        # Similar to Gmail token refresh
        try:
            tokens = await self.auth_manager.list_tokens(
                provider=self.provider,
                token_type=TokenType.REFRESH_TOKEN
            )
            
            if not tokens:
                logger.error("No refresh token found for Drive")
                return False
            
            refresh_token = await self.auth_manager.retrieve_token(tokens[0]["token_id"])
            
            credentials = Credentials(
                token=None,
                refresh_token=refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=self.scopes
            )
            
            credentials.refresh(Request())
            
            await self.auth_manager.store_token(
                provider=self.provider,
                token_type=TokenType.ACCESS_TOKEN,
                token_value=credentials.token,
                expires_in=3600,
                scopes=self.scopes,
                subject=tokens[0].get("subject"),
                client_id=self.client_id
            )
            
            logger.info("Drive token refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Drive token refresh failed", error=str(e))
            return False
    
    async def _initialize_service(self):
        """Initialize Drive service client"""
        try:
            tokens = await self.auth_manager.list_tokens(
                provider=self.provider,
                token_type=TokenType.ACCESS_TOKEN
            )
            
            if not tokens:
                raise Exception("No access token available")
            
            token_value = await self.auth_manager.retrieve_token(tokens[0]["token_id"])
            if not token_value:
                raise Exception("Failed to retrieve access token")
            
            credentials = Credentials(token=token_value)
            self._drive_service = build('drive', 'v3', credentials=credentials)
            
            logger.info("Drive service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Drive service", error=str(e))
            raise
    
    async def get_rate_limits(self) -> Dict[str, Any]:
        """Get current rate limit status for Drive"""
        status = await self.rate_limiter.get_all_status()
        return status.get("drive", {})
    
    async def get_about(self) -> Dict[str, Any]:
        """Get information about Drive and user"""
        status = await self.rate_limiter.is_allowed("drive", "about")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.about().get(
                    fields='user,storageQuota,importFormats,exportFormats,maxImportSizes,maxUploadSize'
                ).execute()
            )
            
            await self.rate_limiter.record_response("drive", "about", True)
            
            return {
                "user": response.get("user", {}),
                "storage_quota": response.get("storageQuota", {}),
                "import_formats": response.get("importFormats", {}),
                "export_formats": response.get("exportFormats", {}),
                "max_import_sizes": response.get("maxImportSizes", {}),
                "max_upload_size": response.get("maxUploadSize")
            }
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "about", False)
            logger.error(f"Failed to get Drive about info", error=str(e))
            raise
    
    async def list_files(
        self,
        query: Optional[str] = None,
        page_size: int = 100,
        page_token: Optional[str] = None,
        order_by: Optional[str] = None,
        include_items_from_all_drives: bool = False,
        spaces: str = "drive"
    ) -> Dict[str, Any]:
        """List files in Drive"""
        status = await self.rate_limiter.is_allowed("drive", "list_files")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            params = {
                'pageSize': min(page_size, 1000),  # Drive limit
                'spaces': spaces,
                'includeItemsFromAllDrives': include_items_from_all_drives,
                'fields': 'nextPageToken,files(id,name,mimeType,parents,size,createdTime,modifiedTime,version,trashed,starred,shared,owners,capabilities,properties,webViewLink,webContentLink,thumbnailLink,iconLink,description)'
            }
            
            if query:
                params['q'] = query
            if page_token:
                params['pageToken'] = page_token
            if order_by:
                params['orderBy'] = order_by
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.files().list(**params).execute()
            )
            
            await self.rate_limiter.record_response("drive", "list_files", True)
            
            files = [DriveFile.from_api_response(file_data) for file_data in response.get('files', [])]
            
            return {
                "files": files,
                "next_page_token": response.get("nextPageToken")
            }
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "list_files", False)
            logger.error(f"Failed to list Drive files", error=str(e))
            raise
    
    async def get_file(self, file_id: str, include_permissions: bool = False) -> DriveFile:
        """Get a specific file"""
        status = await self.rate_limiter.is_allowed("drive", "get_file")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            fields = 'id,name,mimeType,parents,size,createdTime,modifiedTime,version,trashed,starred,shared,owners,capabilities,properties,appProperties,webViewLink,webContentLink,thumbnailLink,iconLink,description'
            
            if include_permissions:
                fields += ',permissions'
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.files().get(
                    fileId=file_id, fields=fields
                ).execute()
            )
            
            await self.rate_limiter.record_response("drive", "get_file", True)
            
            return DriveFile.from_api_response(response)
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "get_file", False)
            logger.error(f"Failed to get Drive file", file_id=file_id, error=str(e))
            raise
    
    async def download_file(self, file_id: str, output_path: Optional[Path] = None) -> bytes:
        """Download file content"""
        status = await self.rate_limiter.is_allowed("drive", "download")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            # Get file metadata first
            file_info = await self.get_file(file_id)
            
            # Handle Google Workspace files (need to export)
            if file_info.mime_type.startswith('application/vnd.google-apps'):
                return await self._export_file(file_id, file_info.mime_type, output_path)
            
            # Download regular files
            request = self._drive_service.files().get_media(fileId=file_id)
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while not done:
                        status_obj, done = await asyncio.get_event_loop().run_in_executor(
                            None, downloader.next_chunk
                        )
                
                with open(output_path, 'rb') as f:
                    content = f.read()
            else:
                file_io = io.BytesIO()
                downloader = MediaIoBaseDownload(file_io, request)
                done = False
                while not done:
                    status_obj, done = await asyncio.get_event_loop().run_in_executor(
                        None, downloader.next_chunk
                    )
                content = file_io.getvalue()
            
            await self.rate_limiter.record_response("drive", "download", True)
            
            return content
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "download", False)
            logger.error(f"Failed to download Drive file", file_id=file_id, error=str(e))
            raise
    
    async def _export_file(
        self,
        file_id: str,
        mime_type: str,
        output_path: Optional[Path] = None,
        export_format: str = 'pdf'
    ) -> bytes:
        """Export Google Workspace file to another format"""
        
        # Get export MIME type
        export_formats = self.export_formats.get(mime_type, {})
        if export_format not in export_formats:
            available_formats = ', '.join(export_formats.keys())
            raise ValueError(f"Format '{export_format}' not supported for {mime_type}. Available: {available_formats}")
        
        export_mime_type = export_formats[export_format]
        
        request = self._drive_service.files().export_media(
            fileId=file_id,
            mimeType=export_mime_type
        )
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status_obj, done = await asyncio.get_event_loop().run_in_executor(
                        None, downloader.next_chunk
                    )
            
            with open(output_path, 'rb') as f:
                content = f.read()
        else:
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            done = False
            while not done:
                status_obj, done = await asyncio.get_event_loop().run_in_executor(
                    None, downloader.next_chunk
                )
            content = file_io.getvalue()
        
        return content
    
    async def upload_file(
        self,
        file_path: Path,
        parent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        resumable: bool = True
    ) -> DriveFile:
        """Upload file to Drive"""
        status = await self.rate_limiter.is_allowed("drive", "upload")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            # Prepare file metadata
            file_metadata = {
                'name': name or file_path.name,
            }
            
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            if description:
                file_metadata['description'] = description
            
            # Guess MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            # Create media upload
            if resumable and file_path.stat().st_size > 5 * 1024 * 1024:  # 5MB
                media = MediaFileUpload(
                    str(file_path),
                    mimetype=mime_type,
                    resumable=True
                )
            else:
                media = MediaFileUpload(str(file_path), mimetype=mime_type)
            
            # Upload file
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id,name,mimeType,parents,size,createdTime,modifiedTime'
                ).execute()
            )
            
            await self.rate_limiter.record_response("drive", "upload", True)
            
            return DriveFile.from_api_response(response)
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "upload", False)
            logger.error(f"Failed to upload file to Drive", file_path=str(file_path), error=str(e))
            raise
    
    async def create_folder(
        self,
        name: str,
        parent_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> DriveFile:
        """Create a new folder"""
        status = await self.rate_limiter.is_allowed("drive", "create_folder")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            file_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            if description:
                file_metadata['description'] = description
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.files().create(
                    body=file_metadata,
                    fields='id,name,mimeType,parents,createdTime,modifiedTime'
                ).execute()
            )
            
            await self.rate_limiter.record_response("drive", "create_folder", True)
            
            return DriveFile.from_api_response(response)
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "create_folder", False)
            logger.error(f"Failed to create Drive folder", name=name, error=str(e))
            raise
    
    async def delete_file(self, file_id: str, permanent: bool = False) -> bool:
        """Delete or trash a file"""
        endpoint = "delete" if permanent else "trash"
        status = await self.rate_limiter.is_allowed("drive", endpoint)
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            if permanent:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._drive_service.files().delete(fileId=file_id).execute()
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._drive_service.files().update(
                        fileId=file_id,
                        body={'trashed': True}
                    ).execute()
                )
            
            await self.rate_limiter.record_response("drive", endpoint, True)
            
            return True
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", endpoint, False)
            logger.error(f"Failed to delete Drive file", file_id=file_id, permanent=permanent, error=str(e))
            return False
    
    async def copy_file(
        self,
        file_id: str,
        name: Optional[str] = None,
        parent_id: Optional[str] = None
    ) -> DriveFile:
        """Copy a file"""
        status = await self.rate_limiter.is_allowed("drive", "copy")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            body = {}
            if name:
                body['name'] = name
            if parent_id:
                body['parents'] = [parent_id]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.files().copy(
                    fileId=file_id,
                    body=body,
                    fields='id,name,mimeType,parents,size,createdTime,modifiedTime'
                ).execute()
            )
            
            await self.rate_limiter.record_response("drive", "copy", True)
            
            return DriveFile.from_api_response(response)
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "copy", False)
            logger.error(f"Failed to copy Drive file", file_id=file_id, error=str(e))
            raise
    
    async def move_file(self, file_id: str, new_parent_id: str, old_parent_id: Optional[str] = None) -> DriveFile:
        """Move a file to a different parent"""
        status = await self.rate_limiter.is_allowed("drive", "move")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            # Get current parents if not provided
            if not old_parent_id:
                file_info = await self.get_file(file_id)
                old_parents = ','.join(file_info.parents)
            else:
                old_parents = old_parent_id
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.files().update(
                    fileId=file_id,
                    addParents=new_parent_id,
                    removeParents=old_parents,
                    fields='id,name,mimeType,parents,size,createdTime,modifiedTime'
                ).execute()
            )
            
            await self.rate_limiter.record_response("drive", "move", True)
            
            return DriveFile.from_api_response(response)
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "move", False)
            logger.error(f"Failed to move Drive file", file_id=file_id, error=str(e))
            raise
    
    async def search_files(
        self,
        query: str,
        max_results: int = 100,
        include_folders: bool = True,
        include_trashed: bool = False
    ) -> List[DriveFile]:
        """Search files in Drive with enhanced query building"""
        
        # Build search query
        search_parts = [f"fullText contains '{query}'"]
        
        if not include_folders:
            search_parts.append("mimeType != 'application/vnd.google-apps.folder'")
        
        if not include_trashed:
            search_parts.append("trashed = false")
        
        full_query = " and ".join(search_parts)
        
        # Search files
        all_files = []
        page_token = None
        
        while len(all_files) < max_results:
            batch_size = min(100, max_results - len(all_files))
            result = await self.list_files(
                query=full_query,
                page_size=batch_size,
                page_token=page_token,
                order_by="relevance desc"
            )
            
            all_files.extend(result["files"])
            page_token = result.get("next_page_token")
            
            if not page_token:
                break
        
        return all_files[:max_results]
    
    async def get_file_permissions(self, file_id: str) -> List[DrivePermission]:
        """Get file permissions"""
        status = await self.rate_limiter.is_allowed("drive", "permissions")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.permissions().list(
                    fileId=file_id,
                    fields='permissions(id,type,role,emailAddress,displayName,photoLink,domain,allowFileDiscovery,expirationTime)'
                ).execute()
            )
            
            await self.rate_limiter.record_response("drive", "permissions", True)
            
            return [
                DrivePermission.from_api_response(perm_data)
                for perm_data in response.get('permissions', [])
            ]
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "permissions", False)
            logger.error(f"Failed to get Drive file permissions", file_id=file_id, error=str(e))
            raise
    
    async def share_file(
        self,
        file_id: str,
        email: str,
        role: str = "reader",
        send_notification: bool = True,
        email_message: Optional[str] = None
    ) -> DrivePermission:
        """Share file with user"""
        status = await self.rate_limiter.is_allowed("drive", "share")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            permission = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.permissions().create(
                    fileId=file_id,
                    body=permission,
                    sendNotificationEmail=send_notification,
                    emailMessage=email_message,
                    fields='id,type,role,emailAddress,displayName'
                ).execute()
            )
            
            await self.rate_limiter.record_response("drive", "share", True)
            
            return DrivePermission.from_api_response(response)
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "share", False)
            logger.error(f"Failed to share Drive file", file_id=file_id, email=email, error=str(e))
            raise
    
    async def get_file_revisions(self, file_id: str) -> List[DriveRevision]:
        """Get file revision history"""
        status = await self.rate_limiter.is_allowed("drive", "revisions")
        if not status.allowed:
            raise Exception(f"Rate limited. Retry after {status.retry_after}s")
        
        try:
            if not self._drive_service:
                await self._initialize_service()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._drive_service.revisions().list(
                    fileId=file_id,
                    fields='revisions(id,mimeType,modifiedTime,size,originalFilename,md5Checksum,keepForever,published,publishedLink,publishAuto,lastModifyingUser)'
                ).execute()
            )
            
            await self.rate_limiter.record_response("drive", "revisions", True)
            
            return [
                DriveRevision.from_api_response(rev_data)
                for rev_data in response.get('revisions', [])
            ]
            
        except Exception as e:
            await self.rate_limiter.record_response("drive", "revisions", False)
            logger.error(f"Failed to get Drive file revisions", file_id=file_id, error=str(e))
            raise
    
    async def export_files(
        self,
        output_directory: Path,
        query: Optional[str] = None,
        max_files: int = 100,
        export_format: str = 'original'
    ) -> int:
        """Export multiple files from Drive"""
        logger.info(f"Starting Drive export", output_dir=str(output_directory), query=query, max_files=max_files)
        
        # List files to export
        files_result = await self.list_files(query=query, page_size=max_files)
        files_to_export = files_result["files"]
        
        if not files_to_export:
            logger.info("No files found to export")
            return 0
        
        # Create output directory
        output_directory.mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        
        for file in files_to_export:
            try:
                # Skip folders unless specifically requested
                if file.mime_type == 'application/vnd.google-apps.folder':
                    continue
                
                # Determine output filename
                filename = file.name
                if export_format != 'original' and file.mime_type.startswith('application/vnd.google-apps'):
                    filename += f'.{export_format}'
                
                output_path = output_directory / filename
                
                # Download file
                content = await self.download_file(file.id, output_path)
                
                logger.info(f"Exported file", filename=filename, size=len(content))
                exported_count += 1
                
            except Exception as e:
                logger.error(f"Failed to export file", file_id=file.id, filename=file.name, error=str(e))
                continue
        
        logger.info(f"Drive export completed", exported_count=exported_count, total_files=len(files_to_export))
        return exported_count