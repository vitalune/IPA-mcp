# API Credentials Setup Guide

Complete guide to setting up API credentials for all supported services.

## 🔧 Overview

The Personal Knowledge Assistant integrates with multiple services to provide comprehensive personal data management. This guide walks you through setting up API credentials for each supported service.

## 📋 Supported Services

| Service | Features | Required Scopes |
|---------|----------|-----------------|
| **Gmail** | Email search, send, analysis | `gmail.readonly`, `gmail.send` |
| **Google Drive** | File search, content analysis | `drive.readonly` |
| **Twitter** | Tweet search, post, analytics | `tweet.read`, `tweet.write` |
| **LinkedIn** | Post creation, profile access | `r_liteprofile`, `w_member_social` |

## 📧 Gmail & Google Drive Setup

### Step 1: Create Google Cloud Project

1. **Go to Google Cloud Console**
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Sign in with your Google account

2. **Create a New Project**
   - Click "Select a project" at the top
   - Click "NEW PROJECT"
   - Enter project name: `Personal Knowledge Assistant`
   - Click "CREATE"

### Step 2: Enable APIs

1. **Navigate to APIs & Services**
   - In the left sidebar, click "APIs & Services" > "Library"

2. **Enable Gmail API**
   - Search for "Gmail API"
   - Click on "Gmail API"
   - Click "ENABLE"

3. **Enable Google Drive API**
   - Search for "Google Drive API"
   - Click on "Google Drive API"
   - Click "ENABLE"

### Step 3: Configure OAuth Consent Screen

1. **Go to OAuth Consent Screen**
   - Navigate to "APIs & Services" > "OAuth consent screen"

2. **Choose User Type**
   - Select "External" (unless you have a Google Workspace account)
   - Click "CREATE"

3. **Fill App Information**
   ```
   App name: Personal Knowledge Assistant
   User support email: your-email@gmail.com
   Application home page: http://localhost:8080
   Application privacy policy: http://localhost:8080/privacy
   Application terms of service: http://localhost:8080/terms
   Authorized domains: localhost
   Developer contact: your-email@gmail.com
   ```
   - Click "SAVE AND CONTINUE"

4. **Add Scopes**
   - Click "ADD OR REMOVE SCOPES"
   - Add these scopes:
     - `https://www.googleapis.com/auth/gmail.readonly`
     - `https://www.googleapis.com/auth/gmail.send`
     - `https://www.googleapis.com/auth/drive.readonly`
   - Click "UPDATE" then "SAVE AND CONTINUE"

5. **Add Test Users** (for development)
   - Click "ADD USERS"
   - Add your email address
   - Click "SAVE AND CONTINUE"

### Step 4: Create OAuth 2.0 Credentials

1. **Go to Credentials**
   - Navigate to "APIs & Services" > "Credentials"

2. **Create OAuth Client ID**
   - Click "CREATE CREDENTIALS" > "OAuth client ID"
   - Application type: "Desktop application"
   - Name: "Personal Knowledge Assistant Desktop"
   - Click "CREATE"

3. **Download Credentials**
   - Click "DOWNLOAD JSON"
   - Save as `config/google_credentials.json`

### Step 5: Configure in IPA-MCP

Add to your `config/config.yaml`:

```yaml
integrations:
  gmail:
    enabled: true
    credentials_file: "config/google_credentials.json"
    scopes:
      - "https://www.googleapis.com/auth/gmail.readonly"
      - "https://www.googleapis.com/auth/gmail.send"
    
  drive:
    enabled: true
    credentials_file: "config/google_credentials.json"
    scopes:
      - "https://www.googleapis.com/auth/drive.readonly"
```

### Step 6: Test Gmail Integration

```bash
# Run authentication test
python -c "
from src.integrations.gmail_client import GmailClient
import asyncio

async def test():
    client = GmailClient()
    await client.authenticate('http://localhost:8080/oauth/callback')

asyncio.run(test())
"
```

## 🐦 Twitter API Setup

### Step 1: Create Twitter Developer Account

1. **Apply for Developer Access**
   - Visit [Twitter Developer Portal](https://developer.twitter.com/)
   - Click "Sign up"
   - Complete the application form
   - Wait for approval (usually 1-2 days)

### Step 2: Create Twitter App

1. **Create New Project**
   - Go to [Developer Portal](https://developer.twitter.com/en/portal/dashboard)
   - Click "Create Project"
   - Fill in project details:
     ```
     Project name: Personal Knowledge Assistant
     Use case: Making a bot
     Description: Personal productivity and analytics tool
     ```

2. **Create App**
   - App name: `Personal-Knowledge-Assistant`
   - Click "Complete"

### Step 3: Configure App Settings

1. **App Settings**
   - Go to your app settings
   - Click "Set up" under "User authentication settings"

2. **Configure OAuth 2.0**
   ```
   App permissions: Read and write
   Type of app: Web App, Automated App or Bot
   App info:
     Callback URI: http://localhost:8080/oauth/callback/twitter
     Website URL: http://localhost:8080
   ```
   - Click "Save"

### Step 4: Get API Keys

1. **API Keys and Tokens**
   - Go to "Keys and tokens" tab
   - Copy the following:
     - API Key (Client ID)
     - API Key Secret (Client Secret)
     - Bearer Token

### Step 5: Configure in IPA-MCP

Add to your `config/config.yaml`:

```yaml
integrations:
  twitter:
    enabled: true
    client_id: "your-twitter-api-key"
    client_secret: "your-twitter-api-secret"
    bearer_token: "your-bearer-token"
    scopes:
      - "tweet.read"
      - "tweet.write"
      - "users.read"
```

### Step 6: Test Twitter Integration

```bash
# Test Twitter connection
python -c "
from src.integrations.twitter_client import TwitterClient
import asyncio

async def test():
    client = TwitterClient()
    await client.authenticate('http://localhost:8080/oauth/callback/twitter')
    tweets = await client.search_tweets('test', max_results=5)
    print(f'Found {len(tweets)} tweets')

asyncio.run(test())
"
```

## 💼 LinkedIn API Setup

### Step 1: Create LinkedIn App

1. **LinkedIn Developer Portal**
   - Visit [LinkedIn Developers](https://www.linkedin.com/developers/)
   - Sign in with your LinkedIn account
   - Click "Create app"

2. **App Details**
   ```
   App name: Personal Knowledge Assistant
   LinkedIn Page: Create a company page or use personal
   Privacy policy URL: http://localhost:8080/privacy
   App logo: Upload a logo (optional)
   ```
   - Check "I agree to LinkedIn API Terms of Use"
   - Click "Create app"

### Step 2: Configure App Products

1. **Request Products**
   - Go to "Products" tab
   - Request access to:
     - **Sign In with LinkedIn using OpenID Connect**
     - **Share on LinkedIn** (for posting capabilities)
     - **Marketing Developer Platform** (for analytics)

2. **Review Process**
   - LinkedIn will review your request
   - This can take several days to weeks
   - You'll receive email notifications about status

### Step 3: Configure OAuth Settings

1. **Auth Settings**
   - Go to "Auth" tab
   - Add redirect URL: `http://localhost:8080/oauth/callback/linkedin`

2. **Copy Credentials**
   - Client ID
   - Client Secret

### Step 4: Configure in IPA-MCP

Add to your `config/config.yaml`:

```yaml
integrations:
  linkedin:
    enabled: true
    client_id: "your-linkedin-client-id"
    client_secret: "your-linkedin-client-secret"
    scopes:
      - "r_liteprofile"
      - "r_emailaddress"
      - "w_member_social"
```

### Step 5: Test LinkedIn Integration

```bash
# Test LinkedIn connection
python -c "
from src.integrations.linkedin_client import LinkedInClient
import asyncio

async def test():
    client = LinkedInClient()
    await client.authenticate('http://localhost:8080/oauth/callback/linkedin')
    profile = await client.get_profile()
    print(f'Connected to profile: {profile.get(\"firstName\", \"Unknown\")}')

asyncio.run(test())
"
```

## 🌐 Complete Configuration Example

Here's a complete `config/config.yaml` with all integrations:

```yaml
# Personal Knowledge Assistant Configuration
app:
  name: "Personal Knowledge Assistant"
  version: "1.0.0"
  environment: "development"

# Server settings
server:
  host: "127.0.0.1"  
  port: 8080

# Security settings
security:
  encryption_enabled: true
  session_timeout_minutes: 60
  require_mfa: false

# Privacy settings  
privacy:
  anonymize_logs: true
  data_retention_days: 90
  enable_analytics: false

# API Integrations
integrations:
  # Gmail Integration
  gmail:
    enabled: true
    credentials_file: "config/google_credentials.json"
    scopes:
      - "https://www.googleapis.com/auth/gmail.readonly"
      - "https://www.googleapis.com/auth/gmail.send"
    rate_limit:
      requests_per_second: 5
      daily_quota: 1000000000  # 1 billion quota units per day

  # Google Drive Integration  
  drive:
    enabled: true
    credentials_file: "config/google_credentials.json"
    scopes:
      - "https://www.googleapis.com/auth/drive.readonly"
    rate_limit:
      requests_per_second: 10
      daily_quota: 20000

  # Twitter Integration
  twitter:
    enabled: true
    client_id: "${TWITTER_CLIENT_ID}"  # Use environment variable
    client_secret: "${TWITTER_CLIENT_SECRET}"
    bearer_token: "${TWITTER_BEARER_TOKEN}"
    scopes:
      - "tweet.read"
      - "tweet.write"
      - "users.read"
    rate_limit:
      requests_per_window: 300
      window_minutes: 15

  # LinkedIn Integration
  linkedin:
    enabled: true
    client_id: "${LINKEDIN_CLIENT_ID}"
    client_secret: "${LINKEDIN_CLIENT_SECRET}"
    scopes:
      - "r_liteprofile"
      - "r_emailaddress"
      - "w_member_social"
    rate_limit:
      requests_per_second: 2
      daily_quota: 500

# OAuth Callback Settings
oauth:
  redirect_uri: "http://localhost:8080/oauth/callback"
  state_expiry_minutes: 10
  
# Database settings
database:
  data_directory: "~/.ipa_mcp/data"
  cache_directory: "~/.ipa_mcp/cache"
  logs_directory: "~/.ipa_mcp/logs"
  encrypt_database: true

# Audit settings
audit:
  audit_enabled: true
  track_api_calls: true
  track_data_access: true
```

## 🔒 Security Best Practices

### Environment Variables

Store sensitive credentials as environment variables:

```bash
# Create .env file
cat > .env << EOF
TWITTER_CLIENT_ID=your-twitter-api-key
TWITTER_CLIENT_SECRET=your-twitter-api-secret
TWITTER_BEARER_TOKEN=your-bearer-token
LINKEDIN_CLIENT_ID=your-linkedin-client-id
LINKEDIN_CLIENT_SECRET=your-linkedin-client-secret
EOF

# Load environment variables
source .env
```

### Secure File Permissions

```bash
# Restrict access to credential files
chmod 600 config/google_credentials.json
chmod 600 config/config.yaml
chmod 600 .env

# Secure data directory
chmod 700 ~/.ipa_mcp/data
```

### Credential Rotation

Set up regular credential rotation:

```yaml
# config/config.yaml
security:
  token_rotation_enabled: true
  token_expiry_hours: 24
  refresh_token_expiry_days: 30
```

## 🧪 Testing API Connections

### Comprehensive Test Script

Create `test_apis.py`:

```python
#!/usr/bin/env python3
"""Test all API connections"""

import asyncio
import sys
from src.integrations.client_manager import APIClientManager

async def test_all_apis():
    """Test all configured API integrations"""
    manager = APIClientManager()
    await manager.initialize()
    
    # Configure services (load from config)
    services = ['gmail', 'drive', 'twitter', 'linkedin']
    
    results = {}
    
    for service in services:
        try:
            print(f"Testing {service} connection...")
            client = await manager.get_client(service)
            
            if client:
                health = await client.get_health_status()
                results[service] = health['status'] == 'healthy'
                print(f"✅ {service}: Connected successfully")
            else:
                results[service] = False
                print(f"❌ {service}: Client not available")
                
        except Exception as e:
            results[service] = False
            print(f"❌ {service}: {str(e)}")
    
    await manager.shutdown()
    
    # Summary
    print("\n📊 Connection Summary:")
    successful = sum(results.values())
    total = len(results)
    
    for service, success in results.items():
        status = "✅ Connected" if success else "❌ Failed"
        print(f"  {service}: {status}")
    
    print(f"\nTotal: {successful}/{total} services connected")
    
    if successful == total:
        print("🎉 All API connections successful!")
        return True
    else:
        print("⚠️ Some API connections failed. Check configuration.")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_all_apis())
    sys.exit(0 if success else 1)
```

Run the test:

```bash
python test_apis.py
```

### Individual Service Tests

Test each service individually:

```bash
# Test Gmail
python -c "
from src.integrations.gmail_client import GmailClient
import asyncio
async def test():
    client = GmailClient()
    messages = await client.search_messages('test', max_results=1)
    print(f'Gmail test: {\"SUCCESS\" if len(messages) >= 0 else \"FAILED\"}')
asyncio.run(test())
"

# Test Twitter  
python -c "
from src.integrations.twitter_client import TwitterClient
import asyncio
async def test():
    client = TwitterClient()
    tweets = await client.search_tweets('hello', max_results=1)
    print(f'Twitter test: {\"SUCCESS\" if len(tweets) >= 0 else \"FAILED\"}')
asyncio.run(test())
"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. OAuth Redirect URI Mismatch

**Error:** `redirect_uri_mismatch`

**Solution:**
- Ensure redirect URIs match exactly in app configuration
- Use `http://localhost:8080/oauth/callback` (not `https`)
- Check for trailing slashes

#### 2. Invalid Scopes

**Error:** `invalid_scope` or `insufficient_permissions`

**Solution:**
- Verify requested scopes are approved for your app
- Check scope spelling and format
- Some scopes require app review/approval

#### 3. Rate Limiting Issues

**Error:** `rate_limit_exceeded` or `429 Too Many Requests`

**Solution:**
```yaml
# Adjust rate limits in config
integrations:
  twitter:
    rate_limit:
      requests_per_window: 100  # Reduce from 300
      window_minutes: 15
```

#### 4. Authentication Token Expired

**Error:** `invalid_token` or `token_expired`

**Solution:**
```bash
# Clear stored tokens and re-authenticate
rm -rf ~/.ipa_mcp/data/tokens/
python -c "
from src.config.auth import get_auth_manager
import asyncio
async def refresh():
    auth = get_auth_manager()
    await auth.cleanup_expired_tokens()
asyncio.run(refresh())
"
```

### Debug Mode

Enable debug logging:

```yaml
# config/config.yaml
app:
  debug: true
  log_level: "DEBUG"

audit:
  track_api_calls: true
```

Check logs:
```bash
tail -f ~/.ipa_mcp/logs/debug.log
```

## 📞 Getting Help

### API-Specific Support

- **Google APIs**: [Google Cloud Support](https://cloud.google.com/support)
- **Twitter API**: [Twitter Developer Community](https://twittercommunity.com/)
- **LinkedIn API**: [LinkedIn Developer Support](https://www.linkedin.com/help/linkedin/ask)

### IPA-MCP Support

- **Documentation**: Check our [comprehensive docs](../README.md)
- **Issues**: [GitHub Issues](https://github.com/vitalune/IPA-mcp/issues)
- **Community**: [GitHub Discussions](https://github.com/vitalune/IPA-mcp/discussions)

---

**API Setup Complete! 🎉**

Your Personal Knowledge Assistant now has access to all configured services and can provide comprehensive insights across your digital life.