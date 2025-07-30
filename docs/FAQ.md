# Frequently Asked Questions (FAQ)

Common questions and answers about the Personal Knowledge Assistant MCP Server.

## 🚀 Getting Started

### What is the Personal Knowledge Assistant?

The Personal Knowledge Assistant (IPA-MCP) is an AI-powered MCP (Model Context Protocol) server that helps you manage, analyze, and gain insights from your personal information across email, social media, documents, and productivity metrics. It integrates with Claude Desktop to provide natural language access to your data.

### How does it integrate with Claude Desktop?

The Personal Knowledge Assistant implements the MCP protocol, allowing it to register as a tool provider for Claude Desktop. Once configured, you can ask Claude natural language questions about your data, and Claude will use the MCP tools to fetch and analyze your information.

### Is my data safe and private?

Yes! The Personal Knowledge Assistant is designed with privacy as a core principle:
- **Local Processing**: Most analysis happens on your machine
- **End-to-End Encryption**: All stored data is encrypted
- **No Cloud Storage**: Your data stays on your devices
- **GDPR Compliant**: Full data export and deletion capabilities
- **Audit Logging**: Complete transparency of data access

### What services can I connect?

Currently supported services:
- **Gmail**: Email search, sending, and analysis
- **Google Drive**: Document search and content analysis  
- **Twitter**: Tweet posting, search, and engagement analytics
- **LinkedIn**: Professional posting and network analysis

Additional services are planned for future releases.

## 🔧 Installation & Setup

### What are the system requirements?

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.9 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free disk space
- **Network**: Internet connection for API integrations

### How do I install the Personal Knowledge Assistant?

1. Clone the repository: `git clone https://github.com/vitalune/IPA-mcp.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure API credentials (see [API Setup Guide](API_SETUP.md))
4. Add to Claude Desktop configuration
5. Restart Claude Desktop

For detailed instructions, see our [Installation Guide](INSTALL.md).

### Why can't Claude Desktop see my MCP server?

Common causes and solutions:

1. **Configuration Path**: Ensure the `cwd` path in Claude Desktop config is absolute and correct
2. **Python Path**: Verify Python is in your system PATH or use absolute path
3. **Restart Required**: Restart Claude Desktop after configuration changes
4. **Permissions**: Check file permissions on the installation directory
5. **Logs**: Check Claude Desktop logs for specific error messages

### How do I get API credentials for each service?

Each service has different requirements:

- **Gmail/Drive**: Google Cloud Console OAuth 2.0 credentials
- **Twitter**: Twitter Developer Portal API keys  
- **LinkedIn**: LinkedIn Developer Program credentials

See our comprehensive [API Setup Guide](API_SETUP.md) for step-by-step instructions.

## 💡 Usage & Features

### What can I ask the Personal Knowledge Assistant?

You can ask natural language questions like:
- "Analyze my email patterns for the last month"
- "Who are my most frequent contacts and how quickly do we respond to each other?"
- "Create a weekly productivity report with insights"
- "Post an update about our project launch to Twitter and LinkedIn"
- "Show me my social media engagement trends"

### How accurate is the data analysis?

The accuracy depends on several factors:
- **Data Quality**: Clean, complete data produces better insights
- **Time Range**: Longer time periods provide more reliable trends
- **Service Connectivity**: All connected services must be functioning
- **Configuration**: Properly configured scopes and permissions

The system uses established statistical methods and clearly indicates confidence levels in its analysis.

### Can I customize what data is analyzed?

Yes! You can:
- **Configure Service Scopes**: Control which data types are accessed
- **Set Time Ranges**: Focus analysis on specific time periods
- **Filter by Keywords**: Search for specific topics or projects
- **Privacy Controls**: Exclude certain types of content
- **Custom Metrics**: Define your own productivity and goal tracking

### How often is my data updated?

- **Real-time Queries**: Email and social media data is fetched on-demand
- **Cached Results**: Some analysis is cached for performance (default 6 hours)
- **Background Sync**: Optional background updates can be configured
- **Manual Refresh**: You can always request fresh data explicitly

## 🔒 Privacy & Security

### Where is my data stored?

Your data is stored locally on your machine:
- **Database Location**: `~/.ipa_mcp/data/` (configurable)
- **Encryption**: All data encrypted at rest using AES-256
- **No Cloud Storage**: Data never leaves your device unless you explicitly export it
- **Temporary Cache**: Some data may be temporarily cached for performance

### What permissions does the app need?

The app requests minimal permissions:
- **Gmail**: Read emails, send emails (optional)
- **Google Drive**: Read-only access to files
- **Twitter**: Read tweets, post tweets (optional)
- **LinkedIn**: Read profile, post updates (optional)

You can configure which permissions to grant for each service.

### Can I delete my data?

Yes! You have full control:
- **Individual Service**: Disconnect and delete data for specific services
- **Complete Deletion**: Remove all stored data with one command
- **Partial Deletion**: Delete data older than a specified time period
- **Export First**: Export your data before deletion if desired

### How is my data encrypted?

- **Algorithm**: AES-256 encryption for data at rest
- **Key Management**: Secure key derivation using PBKDF2
- **Transport**: TLS encryption for all API communications
- **Passwords**: Argon2 hashing for authentication credentials

## 🛠️ Troubleshooting

### The server won't start - what should I check?

1. **Python Version**: Ensure you're using Python 3.9+
2. **Dependencies**: Run `pip install -r requirements.txt`
3. **Configuration**: Validate your `config/config.yaml` file
4. **Permissions**: Check file and directory permissions
5. **Ports**: Ensure port 8080 is available
6. **Logs**: Check `~/.ipa_mcp/logs/` for error messages

### I'm getting "rate limit exceeded" errors

This is normal and expected:
- **Twitter**: 300 requests per 15-minute window
- **Gmail**: Very high limits, rarely reached
- **LinkedIn**: 500 requests per day
- **Solution**: Wait for the rate limit window to reset, or reduce query frequency

### My email analysis shows no data

Possible causes:
1. **Gmail Not Connected**: Check API credentials and authentication
2. **Insufficient Permissions**: Ensure Gmail scopes include `gmail.readonly`
3. **Time Range**: Try a broader time range (e.g., "last 3 months")
4. **Email Volume**: You may not have enough emails in the specified period
5. **Filters**: Check if you're filtering out too much data

### Social media posting isn't working

Check these items:
1. **API Credentials**: Verify Twitter/LinkedIn credentials are correct
2. **Permissions**: Ensure write permissions (`tweet.write`, `w_member_social`)
3. **Content Length**: Check character limits (280 for Twitter, 3000 for LinkedIn)
4. **Rate Limits**: You may have exceeded posting limits
5. **Service Status**: Check if Twitter/LinkedIn APIs are operational

### The analysis results seem incorrect

Verify these factors:
1. **Data Completeness**: Ensure all relevant services are connected
2. **Time Synchronization**: Check system time and time zone settings
3. **Data Quality**: Review source data for completeness and accuracy
4. **Configuration**: Verify analysis parameters and filters
5. **Cache**: Clear cache and retry with fresh data

## 📊 Data Analysis

### How does sentiment analysis work?

The system uses natural language processing to analyze the emotional tone of your communications:
- **Algorithm**: Pre-trained models with domain-specific tuning
- **Accuracy**: Generally 80-90% accurate for clear emotional expressions
- **Languages**: Primarily English, with basic support for other languages
- **Context**: Considers context and relationship dynamics

### What does "relationship strength" mean?

Relationship strength is calculated based on:
- **Communication Frequency**: How often you exchange messages
- **Response Patterns**: How quickly you respond to each other
- **Bidirectional Communication**: Balance of who initiates conversations
- **Recency**: How recently you've communicated
- **Content Engagement**: Length and depth of conversations

### How are productivity patterns identified?

The system analyzes:
- **Temporal Patterns**: When you're most active and responsive
- **Work Distribution**: Balance between different types of activities
- **Response Times**: How quickly you handle different types of communications
- **Goal Achievement**: Progress toward stated objectives
- **Habit Consistency**: Regularity of positive behaviors

## 🔄 Updates & Maintenance

### How do I update the Personal Knowledge Assistant?

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart Claude Desktop to reload the server
```

### Do I need to reconfigure after updates?

Usually not, but:
- **Major Updates**: May require configuration changes
- **API Changes**: Service integrations might need credential updates
- **Security Updates**: May require encryption key rotation
- **Migration Scripts**: Will be provided for breaking changes

### How often should I back up my data?

Recommendations:
- **Automatic Backups**: Enabled by default (5 most recent)
- **Manual Backups**: Weekly for active users
- **Export Data**: Monthly for important analysis periods
- **Configuration Backup**: Before any major changes

## 🤝 Support & Community

### Where can I get help?

1. **Documentation**: Check our comprehensive [docs](../README.md)
2. **GitHub Issues**: Report bugs and request features
3. **GitHub Discussions**: Ask questions and share experiences
4. **Community Forum**: Connect with other users
5. **Email Support**: For security issues or sensitive questions

### How can I contribute to the project?

We welcome contributions:
1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new capabilities
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Improve guides and examples
5. **Testing**: Help test new features and updates

### Is commercial use allowed?

Yes, the Personal Knowledge Assistant is released under the MIT License, which allows commercial use. However:
- **Service APIs**: You're responsible for complying with each service's terms of use
- **Data Privacy**: Ensure compliance with applicable privacy laws
- **Attribution**: Please maintain copyright notices
- **Support**: Commercial support is available upon request

## 🚀 Future Development

### What features are planned?

Upcoming features include:
- **Calendar Integration**: Schedule optimization and meeting analysis
- **Mobile App**: Companion mobile application
- **Advanced Analytics**: Predictive insights and recommendations
- **Workflow Automation**: Automated responses and actions
- **Team Features**: Multi-user collaboration capabilities

### How can I request a new feature?

1. **Check Existing Issues**: See if someone has already requested it
2. **Create Feature Request**: Use our GitHub issue template
3. **Provide Details**: Explain the use case and expected behavior
4. **Community Discussion**: Engage with other users to refine the idea
5. **Implementation**: Consider contributing code if you're able

### Will my data be compatible with future versions?

Yes, we're committed to backward compatibility:
- **Data Migration**: Automatic migration tools for database changes
- **Configuration Updates**: Guided configuration updates
- **Export/Import**: Standard formats for data portability
- **Version Support**: Support for multiple configuration formats

---

## 🆘 Still Need Help?

If you can't find the answer to your question:

1. **Search Documentation**: Use Ctrl+F to search all docs
2. **Check Logs**: Look in `~/.ipa_mcp/logs/` for detailed error information
3. **GitHub Issues**: Search existing issues or create a new one
4. **Community Discussion**: Ask the community on GitHub Discussions
5. **Contact Support**: Email us for sensitive or urgent issues

**Remember**: The more specific your question, the better we can help you!