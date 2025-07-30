# Configuration Reference

Complete reference for configuring the Personal Knowledge Assistant MCP Server.

## 📋 Overview

The Personal Knowledge Assistant uses a YAML-based configuration system with support for environment variables, multiple environments, and secure credential management.

## 📁 Configuration Files

### Primary Configuration File
- **Location**: `config/config.yaml`
- **Purpose**: Main application configuration
- **Format**: YAML with environment variable substitution

### Environment File
- **Location**: `.env` (optional)
- **Purpose**: Environment-specific variables and secrets
- **Format**: Key-value pairs

### Credentials Files
- **Location**: `config/`
- **Files**: `google_credentials.json`, `twitter_credentials.json`, etc.
- **Purpose**: Service-specific API credentials

## ⚙️ Configuration Structure

### Application Settings

```yaml
app:
  name: "Personal Knowledge Assistant"
  version: "1.0.0"
  environment: "development"  # development, staging, production
  debug: true                 # Enable debug logging
  log_level: "INFO"          # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Parameters**:
- `name`: Application display name
- `version`: Application version for logging and reporting
- `environment`: Deployment environment (affects security settings)
- `debug`: Enable/disable debug mode
- `log_level`: Minimum logging level

### Server Configuration

```yaml
server:
  host: "127.0.0.1"          # Server bind address
  port: 8080                  # Server port
  max_connections: 100        # Maximum concurrent connections
  request_timeout: 30         # Request timeout in seconds
  enable_cors: true           # Enable CORS headers
  cors_origins:               # Allowed CORS origins
    - "http://localhost:3000"
    - "https://claude.ai"
```

### Security Settings

```yaml
security:
  # Authentication
  require_mfa: false                    # Require multi-factor authentication
  session_timeout_minutes: 60          # Session timeout
  max_login_attempts: 5                # Maximum failed login attempts
  lockout_duration_minutes: 15         # Account lockout duration
  
  # Rate Limiting
  api_rate_limit_per_minute: 60        # API requests per minute
  cache_rate_limit_per_hour: 500       # Cache operations per hour
  
  # Network Security
  allowed_origins:                      # Allowed CORS origins
    - "http://localhost:3000"
  trusted_proxies: []                   # Trusted proxy IP addresses
  
  # File Security
  max_file_size_mb: 10                 # Maximum file upload size
  allowed_file_types:                   # Allowed file extensions
    - ".txt"
    - ".pdf"
    - ".doc"
    - ".docx"
    - ".json"
    - ".csv"
    - ".xlsx"

  # Content Security
  content_security_policy:              # CSP header configuration
    default_src: "'self'"
    script_src: "'self' 'unsafe-inline'"
    style_src: "'self' 'unsafe-inline'"
    img_src: "'self' data: https:"
```

### Encryption Configuration

```yaml
encryption:
  # Master Key (base64 encoded, 32 bytes)
  master_key: "${ENCRYPTION_MASTER_KEY}"  # Use environment variable
  
  # Key Derivation
  pbkdf2_iterations: 100000               # PBKDF2 iteration count (min: 50000)
  key_salt: "${ENCRYPTION_KEY_SALT}"      # Key derivation salt
  
  # Password Hashing (Argon2)
  argon2_time_cost: 2                     # Time cost parameter
  argon2_memory_cost: 65536               # Memory cost in KB
  argon2_parallelism: 1                   # Parallelism parameter
  
  # Token Settings
  token_expiry_hours: 24                  # OAuth token expiry
  refresh_token_expiry_days: 30           # Refresh token expiry
```

### Privacy Settings

```yaml
privacy:
  # Data Collection
  collect_analytics: false               # Enable analytics collection
  anonymize_logs: true                   # Anonymize IP addresses in logs
  
  # Data Retention
  cache_retention_days: 7                # Cache data retention
  log_retention_days: 90                 # Log file retention
  audit_retention_years: 2               # Audit log retention
  
  # User Rights
  require_explicit_consent: true         # Require explicit user consent
  enable_data_export: true               # Enable data export feature
  enable_data_deletion: true             # Enable data deletion feature
  
  # Geographic Restrictions
  allowed_countries: []                  # Allowed country codes (empty = all)
  blocked_countries: []                  # Blocked country codes
  
  # Processing Purposes
  processing_purposes:                   # Legal purposes for data processing
    - "personal_knowledge_management"
    - "data_analysis"
    - "search_optimization"
```

### Database Configuration

```yaml
database:
  # Storage Paths
  data_directory: "~/.ipa_mcp/data"      # Main data storage
  cache_directory: "~/.ipa_mcp/cache"   # Cache storage
  logs_directory: "~/.ipa_mcp/logs"     # Log storage
  
  # Database Settings
  encrypt_database: true                 # Encrypt database files
  database_backup_enabled: true         # Enable automatic backups
  backup_retention_count: 5              # Number of backups to keep
  
  # Cache Configuration
  max_cache_size_mb: 100                # Maximum cache size
  cache_compression_enabled: true        # Enable cache compression
  cache_ttl_hours: 24                   # Cache time-to-live
```

### API Integration Settings

```yaml
integrations:
  # Gmail Configuration
  gmail:
    enabled: true
    credentials_file: "config/google_credentials.json"
    scopes:
      - "https://www.googleapis.com/auth/gmail.readonly"
      - "https://www.googleapis.com/auth/gmail.send"
    rate_limit:
      requests_per_second: 5
      daily_quota: 1000000000
    batch_size: 100                      # Batch size for bulk operations
    
  # Google Drive Configuration
  drive:
    enabled: true
    credentials_file: "config/google_credentials.json"
    scopes:
      - "https://www.googleapis.com/auth/drive.readonly"
    rate_limit:
      requests_per_second: 10
      daily_quota: 20000
    max_file_size_mb: 100               # Maximum file size to process
    
  # Twitter Configuration  
  twitter:
    enabled: true
    client_id: "${TWITTER_CLIENT_ID}"
    client_secret: "${TWITTER_CLIENT_SECRET}"
    bearer_token: "${TWITTER_BEARER_TOKEN}"
    scopes:
      - "tweet.read"
      - "tweet.write"
      - "users.read"
    rate_limit:
      requests_per_window: 300
      window_minutes: 15
    max_tweet_length: 280
    
  # LinkedIn Configuration
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
    max_post_length: 3000
```

### OAuth Configuration

```yaml
oauth:
  redirect_uri: "http://localhost:8080/oauth/callback"
  state_expiry_minutes: 10               # OAuth state parameter expiry
  
  # Service-specific redirect URIs
  service_redirects:
    gmail: "http://localhost:8080/oauth/callback/gmail"
    drive: "http://localhost:8080/oauth/callback/drive"
    twitter: "http://localhost:8080/oauth/callback/twitter"  
    linkedin: "http://localhost:8080/oauth/callback/linkedin"
```

### Audit Configuration

```yaml
audit:
  # Basic Audit Settings
  audit_enabled: true                    # Enable audit logging
  audit_file_rotation: true             # Enable log file rotation
  audit_max_file_size_mb: 50            # Maximum audit file size
  
  # Event Tracking
  track_api_calls: true                 # Track API calls
  track_data_access: true               # Track data access
  track_configuration_changes: true     # Track config changes
  
  # Alerting
  security_alerts_enabled: true         # Enable security alerts
  failed_login_alert_threshold: 3       # Failed login alert threshold
  
  # Log Integrity
  audit_log_signing: true               # Enable audit log signing
  log_integrity_check_interval_hours: 24 # Integrity check interval
```

### Analytics Configuration

```yaml
analytics:
  # NLP Settings
  nlp_model: "en_core_web_sm"           # SpaCy model for NLP
  sentiment_analysis_enabled: true      # Enable sentiment analysis
  entity_extraction_enabled: true       # Enable entity extraction
  keyword_extraction_enabled: true      # Enable keyword extraction
  
  # Analytics Engine
  correlation_threshold: 0.7            # Minimum correlation threshold
  trend_detection_window_days: 30       # Trend detection window
  anomaly_detection_enabled: true       # Enable anomaly detection
  
  # Caching
  analysis_cache_ttl_hours: 6          # Analysis result cache TTL
  enable_predictive_caching: true       # Precompute likely queries
```

## 🌍 Environment-Specific Configuration

### Development Environment

```yaml
app:
  environment: "development"
  debug: true
  log_level: "DEBUG"

security:
  require_mfa: false
  session_timeout_minutes: 120

privacy:
  anonymize_logs: false                 # Allow full logging for debugging
  collect_analytics: true

database:
  encrypt_database: false               # Optional for development
```

### Production Environment

```yaml
app:
  environment: "production"
  debug: false
  log_level: "WARNING"

security:
  require_mfa: true
  session_timeout_minutes: 30
  max_login_attempts: 3

privacy:
  anonymize_logs: true
  collect_analytics: false

database:
  encrypt_database: true                # Required for production
  database_backup_enabled: true
```

## 🔐 Environment Variables

### Required Variables

```bash
# Encryption
ENCRYPTION_MASTER_KEY=your-32-byte-base64-encoded-key
ENCRYPTION_KEY_SALT=your-16-byte-base64-encoded-salt

# API Credentials
TWITTER_CLIENT_ID=your-twitter-client-id
TWITTER_CLIENT_SECRET=your-twitter-client-secret
TWITTER_BEARER_TOKEN=your-twitter-bearer-token
LINKEDIN_CLIENT_ID=your-linkedin-client-id
LINKEDIN_CLIENT_SECRET=your-linkedin-client-secret

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/ipa_mcp
REDIS_URL=redis://localhost:6379/0
```

### Optional Variables

```bash
# Application
IPA_MCP_ENVIRONMENT=production
IPA_MCP_DEBUG=false
IPA_MCP_LOG_LEVEL=INFO

# Security
IPA_MCP_REQUIRE_MFA=true
IPA_MCP_SESSION_TIMEOUT=30

# Paths
IPA_MCP_DATA_DIRECTORY=/secure/data/path
IPA_MCP_CACHE_DIRECTORY=/tmp/ipa_mcp_cache
IPA_MCP_LOGS_DIRECTORY=/var/log/ipa_mcp
```

## 📝 Configuration Examples

### Minimal Configuration

```yaml
# config/config.yaml - Minimal setup
app:
  name: "Personal Knowledge Assistant"
  environment: "development"

integrations:
  gmail:
    enabled: true
    credentials_file: "config/google_credentials.json"
```

### High-Security Configuration

```yaml
# config/config.yaml - High security setup
app:
  environment: "production"
  debug: false

security:
  require_mfa: true
  session_timeout_minutes: 15
  max_login_attempts: 3
  lockout_duration_minutes: 30

encryption:
  master_key: "${ENCRYPTION_MASTER_KEY}"
  pbkdf2_iterations: 200000              # Higher iterations

privacy:
  anonymize_logs: true
  data_retention_days: 30                # Shorter retention
  require_explicit_consent: true

audit:
  audit_enabled: true
  security_alerts_enabled: true
  audit_log_signing: true
```

### Multi-Service Configuration

```yaml
# config/config.yaml - All services enabled
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
  
  twitter:
    enabled: true
    client_id: "${TWITTER_CLIENT_ID}"
    client_secret: "${TWITTER_CLIENT_SECRET}"
    bearer_token: "${TWITTER_BEARER_TOKEN}"
  
  linkedin:
    enabled: true
    client_id: "${LINKEDIN_CLIENT_ID}"
    client_secret: "${LINKEDIN_CLIENT_SECRET}"
```

## 🔧 Configuration Validation

### Validation Script

Create `validate_config.py`:

```python
#!/usr/bin/env python3
"""Validate configuration file"""

import yaml
import sys
from pathlib import Path
from src.config.settings import Settings

def validate_config():
    """Validate configuration file"""
    try:
        # Load configuration
        settings = Settings()
        
        print("✅ Configuration loaded successfully")
        print(f"   Environment: {settings.environment}")
        print(f"   Debug mode: {settings.debug}")
        print(f"   Data directory: {settings.database.data_directory}")
        
        # Check required directories
        required_dirs = [
            settings.database.data_directory,
            settings.database.cache_directory,
            settings.database.logs_directory
        ]
        
        for directory in required_dirs:
            if directory.exists():
                print(f"✅ Directory exists: {directory}")
            else:
                print(f"⚠️  Directory missing (will be created): {directory}")
        
        # Check integrations
        enabled_services = []
        if settings.integrations.gmail.enabled:
            enabled_services.append("Gmail")
        if settings.integrations.drive.enabled:
            enabled_services.append("Google Drive")
        if settings.integrations.twitter.enabled:
            enabled_services.append("Twitter")
        if settings.integrations.linkedin.enabled:
            enabled_services.append("LinkedIn")
        
        print(f"✅ Enabled services: {', '.join(enabled_services)}")
        
        # Security checks
        if settings.environment == "production":
            security_checks = [
                (settings.security.require_mfa, "MFA required"),
                (settings.database.encrypt_database, "Database encryption"),
                (settings.audit.audit_enabled, "Audit logging"),
                (settings.privacy.anonymize_logs, "Log anonymization")
            ]
            
            for check, description in security_checks:
                if check:
                    print(f"✅ Security: {description}")
                else:
                    print(f"⚠️  Security: {description} disabled")
        
        print("\n🎉 Configuration validation complete!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_config()
    sys.exit(0 if success else 1)
```

Run validation:
```bash
python validate_config.py
```

### Environment Variable Validation

```bash
# Check required environment variables
python -c "
import os
required_vars = [
    'ENCRYPTION_MASTER_KEY',
    'TWITTER_CLIENT_ID',
    'LINKEDIN_CLIENT_ID'
]

missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f'❌ Missing environment variables: {missing}')
    exit(1)
else:
    print('✅ All required environment variables are set')
"
```

## 🔄 Configuration Management

### Multiple Environments

```bash
# Development
cp config/config.yaml config/config.dev.yaml

# Production  
cp config/config.yaml config/config.prod.yaml

# Use environment-specific config
export IPA_MCP_CONFIG_FILE=config/config.prod.yaml
```

### Configuration Hot Reload

The application supports configuration hot reload for non-security settings:

```python
# Reload configuration
from src.config.settings import reload_settings
settings = reload_settings()
```

### Backup and Versioning

```bash
# Backup current configuration  
cp config/config.yaml config/config.yaml.backup.$(date +%Y%m%d)

# Version control configuration (without secrets)
git add config/config.example.yaml
git commit -m "Update configuration template"
```

## 🛡️ Security Best Practices

### Secure Configuration

1. **Use Environment Variables**: Store secrets in environment variables, not config files
2. **File Permissions**: Restrict access to configuration files
   ```bash
   chmod 600 config/config.yaml
   chmod 600 .env
   ```
3. **Separate Secrets**: Keep API credentials in separate files
4. **Regular Rotation**: Rotate encryption keys and API credentials regularly

### Production Checklist

- [ ] `environment` set to "production"
- [ ] `debug` set to false
- [ ] `require_mfa` enabled
- [ ] `encrypt_database` enabled
- [ ] `audit_enabled` set to true
- [ ] `anonymize_logs` enabled
- [ ] Strong encryption keys configured
- [ ] File permissions properly set
- [ ] Secrets stored in environment variables

---

**Configuration Complete! 🎉**

Your Personal Knowledge Assistant is now properly configured for your environment and security requirements.