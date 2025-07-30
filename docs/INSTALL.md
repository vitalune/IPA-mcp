# Installation Guide

Complete step-by-step installation guide for the Personal Knowledge Assistant MCP Server.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.9 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free disk space
- **Network**: Internet connection for API integrations

### Required Software

1. **Python 3.9+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify installation: `python --version`

2. **Git** (for cloning the repository)
   - Download from [git-scm.com](https://git-scm.com/)
   - Verify installation: `git --version`

3. **Claude Desktop** or compatible MCP client
   - Download from [Claude Desktop](https://claude.ai/desktop)

### Optional Dependencies

- **Docker** (for containerized deployment)
- **PostgreSQL** (for advanced data storage)
- **Redis** (for enhanced caching)

## 🚀 Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/vitalune/IPA-mcp.git
cd IPA-mcp

# Install using the setup script
./scripts/install.sh
```

### Method 2: Manual Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/vitalune/IPA-mcp.git
cd IPA-mcp
```

#### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r tests/requirements.txt
```

#### Step 4: Install the Package

```bash
pip install -e .
```

### Method 3: Docker Installation

```bash
# Clone the repository
git clone https://github.com/vitalune/IPA-mcp.git
cd IPA-mcp

# Build the Docker image
docker build -t ipa-mcp .

# Run the container
docker run -p 8080:8080 -v ./config:/app/config ipa-mcp
```

## ⚙️ Configuration

### Step 1: Create Configuration File

```bash
# Copy the example configuration
cp config/config.example.yaml config/config.yaml

# Edit the configuration file
nano config/config.yaml  # or use your preferred editor
```

### Step 2: Basic Configuration

Edit `config/config.yaml`:

```yaml
# Basic application settings
app:
  name: "Personal Knowledge Assistant"
  version: "1.0.0"
  environment: "development"  # Change to "production" when ready
  debug: true  # Set to false in production

# Server settings
server:
  host: "127.0.0.1"
  port: 8080

# Security settings
security:
  encryption_enabled: true
  session_timeout_minutes: 60
  require_mfa: false  # Enable for production
  max_login_attempts: 5

# Privacy settings
privacy:
  anonymize_logs: true
  data_retention_days: 90
  enable_analytics: false

# Database settings
database:
  data_directory: "~/.ipa_mcp/data"
  cache_directory: "~/.ipa_mcp/cache"
  logs_directory: "~/.ipa_mcp/logs"
  encrypt_database: true
```

### Step 3: Set Up API Credentials

You'll need to configure API credentials for the services you want to use. See our [API Setup Guide](API_SETUP.md) for detailed instructions.

**Quick Setup:**
```yaml
integrations:
  gmail:
    enabled: true
    client_id: "your-gmail-client-id"
    client_secret: "your-gmail-client-secret"
    scopes: ["https://www.googleapis.com/auth/gmail.readonly"]
  
  twitter:
    enabled: true
    client_id: "your-twitter-client-id"
    client_secret: "your-twitter-client-secret"
    scopes: ["tweet.read", "tweet.write"]
```

### Step 4: Initialize the Database

```bash
# Create data directories and initialize database
python -c "from src.config.settings import get_settings; settings = get_settings()"
```

## 🔗 Connect to Claude Desktop

### Step 1: Locate Claude Desktop Configuration

**On macOS:**
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**On Windows:**
```cmd
%APPDATA%\Claude\claude_desktop_config.json
```

**On Linux:**
```bash
~/.config/Claude/claude_desktop_config.json
```

### Step 2: Add MCP Server Configuration

Edit the Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "personal-knowledge-assistant": {
      "command": "python",
      "args": ["-m", "src.main"],
      "cwd": "/absolute/path/to/IPA-mcp",
      "env": {
        "PYTHONPATH": "/absolute/path/to/IPA-mcp"
      }
    }
  }
}
```

**Replace `/absolute/path/to/IPA-mcp` with your actual installation path.**

### Step 3: Restart Claude Desktop

Close and restart Claude Desktop to load the new MCP server.

## ✅ Verification

### Step 1: Test the Server

```bash
# Activate your virtual environment (if using)
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Run the server in test mode
python -m src.main --test
```

You should see output similar to:
```
INFO - Starting Personal Knowledge Assistant MCP Server
INFO - Server initialized successfully
INFO - 7 tools registered: send_email, analyze_email_patterns, post_social_media, analyze_social_engagement, manage_project_context, track_personal_metrics, generate_insights_report
INFO - MCP server ready for connections
```

### Step 2: Test in Claude Desktop

1. Open Claude Desktop
2. Start a new conversation
3. Try asking: "What MCP tools are available?"
4. You should see the Personal Knowledge Assistant tools listed

### Step 3: Test Basic Functionality

Try these commands in Claude Desktop:

```
"List all available MCP tools"
"Send a test email to myself"
"Analyze my email patterns for the last week"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Python Version Issues

**Error:** `ModuleNotFoundError` or version compatibility issues

**Solution:**
```bash
# Check Python version
python --version

# If using Python 3.9+, try:
python3.9 -m pip install -r requirements.txt
python3.9 -m src.main
```

#### 2. Virtual Environment Issues

**Error:** Commands not found or import errors

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Verify activation
which python  # Should point to venv/bin/python
```

#### 3. Permission Issues

**Error:** Permission denied when creating directories

**Solution:**
```bash
# On macOS/Linux
chmod +x scripts/install.sh
sudo chown -R $USER:$USER ~/.ipa_mcp

# On Windows, run as Administrator
```

#### 4. Claude Desktop Connection Issues

**Error:** MCP server not appearing in Claude Desktop

**Solutions:**
- Verify the absolute path in `claude_desktop_config.json`
- Check that Python is in your system PATH
- Restart Claude Desktop after configuration changes
- Check Claude Desktop logs for error messages

**View Claude Desktop logs:**
- **macOS**: `~/Library/Logs/Claude/`
- **Windows**: `%LOCALAPPDATA%\Claude\logs\`
- **Linux**: `~/.local/share/Claude/logs/`

#### 5. Network/Firewall Issues

**Error:** Connection timeouts or API failures

**Solutions:**
- Check firewall settings (allow port 8080)
- Verify internet connection
- Test API credentials separately
- Check corporate proxy settings

### Environment-Specific Issues

#### Windows

```cmd
# If you get SSL errors
pip install --upgrade certifi

# If pip fails
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### macOS

```bash
# If you get certificate errors
/Applications/Python\ 3.x/Install\ Certificates.command

# If using Homebrew Python
brew install python@3.9
pip3.9 install -r requirements.txt
```

#### Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv

# For CentOS/RHEL
sudo yum install python3-devel python3-pip
```

## 🔧 Advanced Configuration

### Production Deployment

For production deployment, update your configuration:

```yaml
# config/config.yaml
app:
  environment: "production"
  debug: false

security:
  require_mfa: true
  session_timeout_minutes: 30
  encryption_enabled: true

audit:
  audit_enabled: true
  audit_file_rotation: true
  security_alerts_enabled: true

database:
  encrypt_database: true
  database_backup_enabled: true
```

### Environment Variables

You can override configuration with environment variables:

```bash
export IPA_MCP_ENVIRONMENT=production
export IPA_MCP_DEBUG=false
export IPA_MCP_ENCRYPTION_ENABLED=true
export IPA_MCP_DATA_DIRECTORY=/secure/data/path
```

### Custom Installation Paths

```bash
# Install to custom location
pip install -e . --prefix /custom/path

# Update Claude Desktop config
{
  "mcpServers": {
    "personal-knowledge-assistant": {
      "command": "/custom/path/bin/python",
      "args": ["-m", "src.main"],
      "cwd": "/custom/path/IPA-mcp"
    }
  }
}
```

## 📊 Performance Optimization

### System Optimization

```bash
# Increase file descriptor limits (Linux/macOS)
ulimit -n 8192

# Optimize Python garbage collection
export PYTHONOPTIMIZE=1
```

### Database Optimization

```yaml
# config/config.yaml
database:
  cache_compression_enabled: true
  max_cache_size_mb: 500
  backup_retention_count: 10
```

## 🔄 Updates

### Updating the Installation

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart the server
# (Claude Desktop will automatically restart the MCP server)
```

### Version Migration

```bash
# Check current version
python -c "from src import __version__; print(__version__)"

# Run migration scripts (if needed)
python scripts/migrate.py --from-version 1.0.0 --to-version 1.1.0
```

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Look in `~/.ipa_mcp/logs/` for error messages
2. **Review configuration**: Verify your `config/config.yaml` settings
3. **Test components**: Run individual tests to isolate issues
4. **Consult documentation**: Check our [Troubleshooting Guide](TROUBLESHOOTING.md)
5. **Community support**: Ask questions in [GitHub Discussions](https://github.com/vitalune/IPA-mcp/discussions)
6. **Report bugs**: Create an issue on [GitHub Issues](https://github.com/vitalune/IPA-mcp/issues)

## ✅ Next Steps

After successful installation:

1. **Configure API Credentials**: Follow the [API Setup Guide](API_SETUP.md)
2. **Learn the Features**: Read the [User Guide](USER_GUIDE.md)
3. **Customize Settings**: Review the [Configuration Reference](CONFIGURATION.md)
4. **Explore Examples**: Try the sample workflows in [Examples](EXAMPLES.md)

---

**Installation complete! 🎉**

Your Personal Knowledge Assistant MCP Server is now ready to transform how you manage and analyze your personal information.