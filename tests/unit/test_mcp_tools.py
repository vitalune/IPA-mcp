"""
Unit tests for MCP Tools

Tests all 7 MCP tools for proper functionality, error handling, and protocol compliance.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from mcp.types import Tool, TextContent

# Import the MCP server components
from src.main import server, handle_call_tool, handle_list_tools, TOOL_DEFINITIONS


class TestMCPToolDefinitions:
    """Test MCP tool definitions and schema validation"""
    
    def test_all_tools_are_defined(self):
        """Test that all expected tools are defined"""
        expected_tools = [
            'send_email',
            'analyze_email_patterns', 
            'post_social_media',
            'analyze_social_engagement',
            'manage_project_context',
            'track_personal_metrics',
            'generate_insights_report'
        ]
        
        tool_names = [tool.name for tool in TOOL_DEFINITIONS]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
    
    def test_tool_schema_validity(self):
        """Test that all tool schemas are valid JSON Schema"""
        for tool in TOOL_DEFINITIONS:
            schema = tool.inputSchema
            
            # Basic schema structure checks
            assert isinstance(schema, dict), f"Tool {tool.name} schema must be a dict"
            assert schema.get('type') == 'object', f"Tool {tool.name} schema must be object type"
            assert 'properties' in schema, f"Tool {tool.name} schema must have properties"
            
            # Check required fields are valid
            if 'required' in schema:
                required_fields = schema['required']
                assert isinstance(required_fields, list), f"Tool {tool.name} required must be a list"
                
                for field in required_fields:
                    assert field in schema['properties'], f"Required field {field} not in properties for {tool.name}"
    
    def test_tool_descriptions_present(self):
        """Test that all tools have meaningful descriptions"""
        for tool in TOOL_DEFINITIONS:
            assert tool.description, f"Tool {tool.name} must have a description"
            assert len(tool.description) > 20, f"Tool {tool.name} description too short"


class TestMCPToolExecution:
    """Test MCP tool execution and responses"""
    
    @pytest.mark.asyncio
    async def test_send_email_tool(self):
        """Test send_email tool execution"""
        arguments = {
            "to": ["test@example.com"],
            "subject": "Test Email",
            "body": "This is a test email body."
        }
        
        result = await handle_call_tool("send_email", arguments)
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Result should not be empty"
        assert isinstance(result[0], TextContent), "Result items should be TextContent"
        assert "test@example.com" in result[0].text, "Result should mention recipient"
    
    @pytest.mark.asyncio
    async def test_send_email_with_optional_fields(self):
        """Test send_email tool with optional fields"""
        arguments = {
            "to": ["test@example.com"],
            "subject": "Test Email",
            "body": "This is a test email body.",
            "cc": ["cc@example.com"],
            "bcc": ["bcc@example.com"],
            "attachments": ["/path/to/file.pdf"]
        }
        
        result = await handle_call_tool("send_email", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_analyze_email_patterns_tool(self):
        """Test analyze_email_patterns tool execution"""
        arguments = {
            "timeframe": "month",
            "analysis_type": "patterns"
        }
        
        result = await handle_call_tool("analyze_email_patterns", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        assert "patterns" in result[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_analyze_email_patterns_with_contacts(self):
        """Test analyze_email_patterns with specific contacts"""
        arguments = {
            "timeframe": "week",
            "contacts": ["john@example.com", "jane@example.com"],
            "analysis_type": "frequency"
        }
        
        result = await handle_call_tool("analyze_email_patterns", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_post_social_media_tool(self):
        """Test post_social_media tool execution"""
        arguments = {
            "platforms": ["twitter", "linkedin"],
            "content": "Test post content #testing"
        }
        
        result = await handle_call_tool("post_social_media", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        assert "twitter" in result[0].text.lower() or "linkedin" in result[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_post_social_media_with_media(self):
        """Test post_social_media with media attachments"""
        arguments = {
            "platforms": ["twitter"],
            "content": "Test post with media",
            "media_urls": ["https://example.com/image.jpg"],
            "hashtags": ["test", "automation"]
        }
        
        result = await handle_call_tool("post_social_media", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_analyze_social_engagement_tool(self):
        """Test analyze_social_engagement tool execution"""
        arguments = {
            "platforms": ["twitter", "linkedin"],
            "timeframe": "week",
            "metrics": ["engagement", "reach"]
        }
        
        result = await handle_call_tool("analyze_social_engagement", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        assert "engagement" in result[0].text.lower() or "analysis" in result[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_manage_project_context_create(self):
        """Test manage_project_context tool with create action"""
        arguments = {
            "action": "create",
            "project_data": {
                "name": "Test Project",
                "description": "A test project for unit testing",
                "priority": "medium",
                "status": "planning"
            }
        }
        
        result = await handle_call_tool("manage_project_context", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        assert "create" in result[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_manage_project_context_list(self):
        """Test manage_project_context tool with list action"""
        arguments = {
            "action": "list"
        }
        
        result = await handle_call_tool("manage_project_context", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        assert "list" in result[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_track_personal_metrics_tool(self):
        """Test track_personal_metrics tool execution"""
        arguments = {
            "metric_type": "productivity",
            "action": "log",
            "data": {
                "hours_worked": 8,
                "tasks_completed": 5,
                "focus_score": 7
            }
        }
        
        result = await handle_call_tool("track_personal_metrics", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        assert "productivity" in result[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_track_personal_metrics_analyze(self):
        """Test track_personal_metrics with analyze action"""
        arguments = {
            "metric_type": "habits",
            "action": "analyze",
            "timeframe": "month"
        }
        
        result = await handle_call_tool("track_personal_metrics", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_generate_insights_report_tool(self):
        """Test generate_insights_report tool execution"""
        arguments = {
            "report_type": "weekly_digest",
            "data_sources": ["email", "social_media"],
            "format": "markdown",
            "include_recommendations": True
        }
        
        result = await handle_call_tool("generate_insights_report", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        assert "weekly_digest" in result[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_generate_insights_report_custom(self):
        """Test generate_insights_report with custom filters"""
        arguments = {
            "report_type": "custom",
            "data_sources": ["projects", "personal_metrics"],
            "format": "json",
            "custom_filters": {
                "priority": "high",
                "status": "active"
            }
        }
        
        result = await handle_call_tool("generate_insights_report", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)


class TestMCPToolErrorHandling:
    """Test error handling and edge cases for MCP tools"""
    
    @pytest.mark.asyncio
    async def test_unknown_tool_name(self):
        """Test handling of unknown tool names"""
        result = await handle_call_tool("unknown_tool", {})
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        assert "unknown tool" in result[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_missing_required_arguments(self):
        """Test handling of missing required arguments"""
        # send_email requires 'to', 'subject', and 'body'
        arguments = {
            "subject": "Test",
            # Missing 'to' and 'body'
        }
        
        # The tool should handle this gracefully
        result = await handle_call_tool("send_email", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_invalid_argument_types(self):
        """Test handling of invalid argument types"""
        arguments = {
            "to": "not_a_list",  # Should be a list
            "subject": 123,  # Should be a string
            "body": None  # Should be a string
        }
        
        result = await handle_call_tool("send_email", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_empty_arguments(self):
        """Test handling of empty arguments"""
        result = await handle_call_tool("send_email", {})
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_tool_execution_exception(self):
        """Test handling of exceptions during tool execution"""
        # Use a tool that might raise an exception
        arguments = {
            "action": "invalid_action"  # Not a valid action
        }
        
        result = await handle_call_tool("manage_project_context", arguments)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance"""
    
    @pytest.mark.asyncio
    async def test_list_tools_response_format(self):
        """Test that list_tools returns proper format"""
        tools = await handle_list_tools()
        
        assert isinstance(tools, list), "list_tools should return a list"
        
        for tool in tools:
            assert isinstance(tool, Tool), "Each tool should be a Tool instance"
            assert hasattr(tool, 'name'), "Tool should have a name"
            assert hasattr(tool, 'description'), "Tool should have a description"
            assert hasattr(tool, 'inputSchema'), "Tool should have an inputSchema"
    
    @pytest.mark.asyncio
    async def test_call_tool_response_format(self):
        """Test that call_tool returns proper format"""
        result = await handle_call_tool("send_email", {
            "to": ["test@example.com"],
            "subject": "Test",
            "body": "Test body"
        })
        
        assert isinstance(result, list), "call_tool should return a list"
        
        for item in result:
            assert isinstance(item, TextContent), "Each result item should be TextContent"
            assert hasattr(item, 'type'), "TextContent should have type"
            assert hasattr(item, 'text'), "TextContent should have text"
            assert item.type == 'text', "Type should be 'text'"
    
    def test_tool_schema_compliance(self):
        """Test that tool schemas comply with JSON Schema Draft 7"""
        for tool in TOOL_DEFINITIONS:
            schema = tool.inputSchema
            
            # Required fields for JSON Schema object
            assert schema.get('type') == 'object'
            assert 'properties' in schema
            
            # Properties should be a dict
            assert isinstance(schema['properties'], dict)
            
            # If required exists, it should be a list
            if 'required' in schema:
                assert isinstance(schema['required'], list)
            
            # Check each property definition
            for prop_name, prop_def in schema['properties'].items():
                assert isinstance(prop_def, dict), f"Property {prop_name} definition should be dict"
                assert 'type' in prop_def or 'enum' in prop_def, f"Property {prop_name} should have type or enum"


class TestMCPToolValidation:
    """Test input validation for MCP tools"""
    
    @pytest.mark.asyncio
    async def test_email_address_validation(self):
        """Test email address validation in send_email tool"""
        # Valid email addresses
        valid_emails = [
            "user@example.com",
            "test.user+tag@domain.co.uk",
            "123@test.org"
        ]
        
        for email in valid_emails:
            arguments = {
                "to": [email],
                "subject": "Test",
                "body": "Test body"
            }
            
            result = await handle_call_tool("send_email", arguments)
            assert isinstance(result, list)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_timeframe_validation(self):
        """Test timeframe validation in analysis tools"""
        valid_timeframes = ["week", "month", "quarter", "year"]
        
        for timeframe in valid_timeframes:
            arguments = {
                "timeframe": timeframe,
                "analysis_type": "patterns"
            }
            
            result = await handle_call_tool("analyze_email_patterns", arguments)
            assert isinstance(result, list)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_platform_validation(self):
        """Test platform validation in social media tools"""
        valid_platforms = ["twitter", "linkedin", "facebook", "instagram"]
        
        for platform in valid_platforms:
            arguments = {
                "platforms": [platform],
                "content": "Test post content"
            }
            
            result = await handle_call_tool("post_social_media", arguments)
            assert isinstance(result, list)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_priority_validation(self):
        """Test priority validation in project management"""
        valid_priorities = ["low", "medium", "high", "urgent"]
        
        for priority in valid_priorities:
            arguments = {
                "action": "create",
                "project_data": {
                    "name": "Test Project",
                    "priority": priority
                }
            }
            
            result = await handle_call_tool("manage_project_context", arguments)
            assert isinstance(result, list)
            assert len(result) > 0


class TestMCPToolPerformance:
    """Test performance characteristics of MCP tools"""
    
    @pytest.mark.asyncio
    async def test_tool_execution_timeout(self):
        """Test that tools complete within reasonable time"""
        import time
        
        start_time = time.time()
        
        arguments = {
            "to": ["test@example.com"],
            "subject": "Performance Test",
            "body": "Testing tool execution performance"
        }
        
        result = await handle_call_tool("send_email", arguments)
        
        execution_time = time.time() - start_time
        
        # Tool should complete within 5 seconds (generous timeout for placeholder implementations)
        assert execution_time < 5.0, f"Tool execution took {execution_time:.2f}s, expected < 5.0s"
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test that multiple tools can be executed concurrently"""
        import time
        
        start_time = time.time()
        
        # Execute multiple tools concurrently
        tasks = [
            handle_call_tool("send_email", {
                "to": ["test1@example.com"],
                "subject": "Concurrent Test 1",
                "body": "Test body 1"
            }),
            handle_call_tool("analyze_email_patterns", {
                "timeframe": "week"
            }),
            handle_call_tool("post_social_media", {
                "platforms": ["twitter"],
                "content": "Concurrent test post"
            })
        ]
        
        results = await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        
        # All tools should complete and return valid results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], TextContent)
        
        # Should complete faster than sequential execution
        assert execution_time < 10.0, f"Concurrent execution took {execution_time:.2f}s"


class TestMCPToolLogging:
    """Test logging behavior of MCP tools"""
    
    @pytest.mark.asyncio
    async def test_tool_execution_logging(self, caplog):
        """Test that tool execution is properly logged"""
        import logging
        
        with caplog.at_level(logging.INFO):
            arguments = {
                "to": ["test@example.com"],
                "subject": "Logging Test",
                "body": "Testing logging functionality"
            }
            
            result = await handle_call_tool("send_email", arguments)
            
            assert isinstance(result, list)
            
            # Check that tool execution was logged
            log_messages = [record.message for record in caplog.records]
            tool_logged = any("Tool called" in msg for msg in log_messages)
            assert tool_logged, "Tool execution should be logged"
    
    @pytest.mark.asyncio
    async def test_error_logging(self, caplog):
        """Test that errors are properly logged"""
        import logging
        
        with caplog.at_level(logging.ERROR):
            # This should trigger error logging
            result = await handle_call_tool("unknown_tool", {})
            
            assert isinstance(result, list)
            
            # Check for error log
            error_logged = any(record.levelname == "ERROR" for record in caplog.records)
            assert error_logged, "Errors should be logged at ERROR level"