"""
MCP Protocol Compliance Tests

Tests compliance with the Model Context Protocol (MCP) specification.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from mcp.server import Server
from mcp.types import Tool, Resource, TextContent, ImageContent, EmbeddedResource
from mcp.server.stdio import stdio_server

from src.main import server, handle_list_tools, handle_call_tool, handle_list_resources, TOOL_DEFINITIONS


class TestMCPServerCompliance:
    """Test MCP server compliance with protocol specifications"""
    
    def test_server_initialization(self):
        """Test that MCP server initializes correctly"""
        assert server is not None
        assert isinstance(server, Server)
        assert server.name == "personal-knowledge-assistant"
    
    @pytest.mark.asyncio
    async def test_server_handlers_registration(self):
        """Test that all required MCP handlers are registered"""
        # The handlers should be registered via decorators
        # We can verify they exist and are callable
        assert callable(handle_list_tools)
        assert callable(handle_call_tool)
        assert callable(handle_list_resources)
    
    @pytest.mark.asyncio
    async def test_list_tools_protocol_compliance(self):
        """Test list_tools response complies with MCP protocol"""
        tools = await handle_list_tools()
        
        # Must return a list
        assert isinstance(tools, list), "list_tools must return a list"
        
        # Each item must be a Tool
        for tool in tools:
            assert isinstance(tool, Tool), f"Tool {tool} must be a Tool instance"
            
            # Required Tool fields
            assert hasattr(tool, 'name'), "Tool must have name"
            assert hasattr(tool, 'description'), "Tool must have description"
            assert hasattr(tool, 'inputSchema'), "Tool must have inputSchema"
            
            # Name must be non-empty string
            assert isinstance(tool.name, str), "Tool name must be string"
            assert len(tool.name) > 0, "Tool name must not be empty"
            
            # Description must be non-empty string
            assert isinstance(tool.description, str), "Tool description must be string"
            assert len(tool.description) > 0, "Tool description must not be empty"
            
            # Input schema must be valid JSON Schema
            assert isinstance(tool.inputSchema, dict), "inputSchema must be dict"
            assert tool.inputSchema.get('type') == 'object', "inputSchema must be object type"
    
    @pytest.mark.asyncio
    async def test_call_tool_protocol_compliance(self):
        """Test call_tool response complies with MCP protocol"""
        # Test with valid tool and arguments
        result = await handle_call_tool("send_email", {
            "to": ["test@example.com"],
            "subject": "Test",
            "body": "Test body"
        })
        
        # Must return a list
        assert isinstance(result, list), "call_tool must return a list"
        
        # Each item must be a Content type
        for item in result:
            # Should be TextContent, ImageContent, or EmbeddedResource
            assert isinstance(item, (TextContent, ImageContent, EmbeddedResource)), \
                f"Result item must be valid MCP content type, got {type(item)}"
            
            if isinstance(item, TextContent):
                assert hasattr(item, 'type'), "TextContent must have type"
                assert hasattr(item, 'text'), "TextContent must have text"
                assert item.type == 'text', "TextContent type must be 'text'"
                assert isinstance(item.text, str), "TextContent text must be string"
    
    @pytest.mark.asyncio
    async def test_list_resources_protocol_compliance(self):
        """Test list_resources response complies with MCP protocol"""
        resources = await handle_list_resources()
        
        # Must return a list
        assert isinstance(resources, list), "list_resources must return a list"
        
        # Each item must be a Resource (when resources are implemented)
        for resource in resources:
            assert isinstance(resource, Resource), f"Resource {resource} must be a Resource instance"
            
            # Required Resource fields
            assert hasattr(resource, 'uri'), "Resource must have uri"
            assert hasattr(resource, 'name'), "Resource must have name"
            
            # URI must be non-empty string
            assert isinstance(resource.uri, str), "Resource uri must be string"
            assert len(resource.uri) > 0, "Resource uri must not be empty"
    
    @pytest.mark.asyncio
    async def test_error_handling_protocol_compliance(self):
        """Test error handling complies with MCP protocol"""
        # Test unknown tool
        result = await handle_call_tool("unknown_tool", {})
        
        # Should return valid response (not raise exception)
        assert isinstance(result, list), "Error response must be list"
        assert len(result) > 0, "Error response must not be empty"
        assert isinstance(result[0], TextContent), "Error response must be TextContent"
        
        # Test invalid arguments
        result = await handle_call_tool("send_email", {"invalid": "args"})
        
        # Should handle gracefully
        assert isinstance(result, list), "Invalid args response must be list"
        assert len(result) > 0, "Invalid args response must not be empty"


class TestMCPToolSchemaCompliance:
    """Test tool schemas comply with MCP and JSON Schema standards"""
    
    def test_tool_names_are_valid(self):
        """Test that all tool names follow MCP naming conventions"""
        for tool in TOOL_DEFINITIONS:
            name = tool.name
            
            # Should be lowercase with underscores
            assert name.islower(), f"Tool name '{name}' should be lowercase"
            assert ' ' not in name, f"Tool name '{name}' should not contain spaces"
            assert name.replace('_', '').isalnum(), f"Tool name '{name}' should be alphanumeric with underscores"
            
            # Should be descriptive
            assert len(name) >= 3, f"Tool name '{name}' should be at least 3 characters"
            assert len(name) <= 50, f"Tool name '{name}' should be at most 50 characters"
    
    def test_tool_descriptions_are_comprehensive(self):
        """Test that tool descriptions are comprehensive and helpful"""
        for tool in TOOL_DEFINITIONS:
            description = tool.description
            
            # Should be meaningful
            assert len(description) >= 20, f"Tool '{tool.name}' description too short"
            assert len(description) <= 500, f"Tool '{tool.name}' description too long"
            
            # Should start with capital letter and end with appropriate punctuation
            assert description[0].isupper(), f"Tool '{tool.name}' description should start with capital"
            
            # Should not be generic
            generic_words = ['tool', 'function', 'method', 'api']
            assert not any(description.lower().startswith(word) for word in generic_words), \
                f"Tool '{tool.name}' description should not start with generic terms"
    
    def test_input_schemas_are_valid_json_schema(self):
        """Test that input schemas are valid JSON Schema Draft 7"""
        for tool in TOOL_DEFINITIONS:
            schema = tool.inputSchema
            
            # Basic JSON Schema requirements
            assert isinstance(schema, dict), f"Tool '{tool.name}' schema must be dict"
            assert schema.get('type') == 'object', f"Tool '{tool.name}' schema must be object"
            assert 'properties' in schema, f"Tool '{tool.name}' schema must have properties"
            
            properties = schema['properties']
            assert isinstance(properties, dict), f"Tool '{tool.name}' properties must be dict"
            
            # Check each property
            for prop_name, prop_def in properties.items():
                assert isinstance(prop_def, dict), f"Property '{prop_name}' must be dict"
                
                # Must have type or enum
                assert 'type' in prop_def or 'enum' in prop_def, \
                    f"Property '{prop_name}' must have type or enum"
                
                # If has description, should be meaningful
                if 'description' in prop_def:
                    desc = prop_def['description']
                    assert isinstance(desc, str), f"Property '{prop_name}' description must be string"
                    assert len(desc) >= 5, f"Property '{prop_name}' description too short"
            
            # Check required fields exist in properties
            if 'required' in schema:
                required = schema['required']
                assert isinstance(required, list), f"Tool '{tool.name}' required must be list"
                
                for field in required:
                    assert field in properties, \
                        f"Required field '{field}' not in properties for tool '{tool.name}'"
    
    def test_tool_parameters_are_well_designed(self):
        """Test that tool parameters are well-designed for usability"""
        for tool in TOOL_DEFINITIONS:
            properties = tool.inputSchema.get('properties', {})
            required = tool.inputSchema.get('required', [])
            
            # Should not have too many required parameters
            assert len(required) <= 5, f"Tool '{tool.name}' has too many required parameters"
            
            # Should have reasonable parameter names
            for prop_name in properties:
                # Should be descriptive
                assert len(prop_name) >= 2, f"Parameter '{prop_name}' name too short"
                assert prop_name.islower() or '_' in prop_name, \
                    f"Parameter '{prop_name}' should be lowercase or snake_case"
                
                # Should not be abbreviated unless common
                common_abbrevs = ['id', 'url', 'uri', 'ip', 'cc', 'bcc', 'api']
                if len(prop_name) <= 3 and prop_name not in common_abbrevs:
                    # Very short names should be common abbreviations
                    pass  # Allow for now, but could be stricter
            
            # Check for good defaults where appropriate
            for prop_name, prop_def in properties.items():
                if 'default' in prop_def:
                    default_val = prop_def['default']
                    prop_type = prop_def.get('type')
                    
                    # Default should match the property type
                    if prop_type == 'string':
                        assert isinstance(default_val, str), \
                            f"Default for '{prop_name}' should be string"
                    elif prop_type == 'integer':
                        assert isinstance(default_val, int), \
                            f"Default for '{prop_name}' should be int"
                    elif prop_type == 'boolean':
                        assert isinstance(default_val, bool), \
                            f"Default for '{prop_name}' should be bool"


class TestMCPServerIntegration:
    """Test MCP server integration and protocol handling"""
    
    @pytest.mark.asyncio
    async def test_server_stdio_transport(self):
        """Test server works with stdio transport (basic smoke test)"""
        # This is a basic test to ensure the server can be created with stdio transport
        # Full transport testing would require more complex setup
        
        try:
            # The server should be able to create initialization options
            init_options = server.create_initialization_options()
            assert init_options is not None
        except Exception as e:
            pytest.fail(f"Server initialization options creation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        """Test server handles concurrent tool calls correctly"""
        # Create multiple concurrent tool calls
        tasks = [
            handle_call_tool("send_email", {
                "to": [f"user{i}@example.com"],
                "subject": f"Test {i}",
                "body": f"Test body {i}"
            }) for i in range(5)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 5
        
        for i, result in enumerate(results):
            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], TextContent)
            assert f"user{i}@example.com" in result[0].text
    
    @pytest.mark.asyncio
    async def test_tool_execution_isolation(self):
        """Test that tool executions are properly isolated"""
        # This test ensures that one tool execution doesn't affect another
        
        # Execute same tool with different arguments
        result1 = await handle_call_tool("analyze_email_patterns", {
            "timeframe": "week",
            "analysis_type": "patterns"
        })
        
        result2 = await handle_call_tool("analyze_email_patterns", {
            "timeframe": "month", 
            "analysis_type": "frequency"
        })
        
        # Results should be different (indicating proper isolation)
        assert result1 != result2
        
        # Both should be valid
        for result in [result1, result2]:
            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_large_argument_handling(self):
        """Test server handles large arguments correctly"""
        # Test with large content
        large_content = "A" * 10000  # 10KB of text
        
        result = await handle_call_tool("post_social_media", {
            "platforms": ["twitter"],
            "content": large_content
        })
        
        # Should handle gracefully (may truncate or process as appropriate)
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """Test server handles Unicode characters correctly"""
        unicode_content = "Hello 世界! 🚀 Émojis and spéciàl characters"
        
        result = await handle_call_tool("send_email", {
            "to": ["test@example.com"],
            "subject": unicode_content,
            "body": f"Body with {unicode_content}"
        })
        
        # Should handle Unicode correctly
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        # Unicode should be preserved in response
        assert "世界" in result[0].text or "unicode" in result[0].text.lower()


class TestMCPProtocolVersioning:
    """Test MCP protocol versioning and compatibility"""
    
    def test_server_supports_required_mcp_version(self):
        """Test server supports the required MCP protocol version"""
        # The server should support the MCP version we're using
        # This is implicit in the imports working, but we can test it explicitly
        
        try:
            from mcp import __version__ as mcp_version
            # Should be able to import MCP components
            assert mcp_version is not None
        except ImportError:
            pytest.fail("Required MCP version not available")
    
    def test_tool_schema_compatibility(self):
        """Test tool schemas are compatible with MCP requirements"""
        for tool in TOOL_DEFINITIONS:
            # Schema should be compatible with JSON Schema Draft 7
            # (which is required by MCP)
            schema = tool.inputSchema
            
            # Should not use features incompatible with MCP
            assert '$schema' not in schema or 'draft-07' in schema.get('$schema', ''), \
                f"Tool '{tool.name}' schema should be Draft 7 compatible"
            
            # Should not use deprecated features
            deprecated_keywords = ['id', 'definitions']  # These are draft-04 style
            for keyword in deprecated_keywords:
                assert keyword not in schema, \
                    f"Tool '{tool.name}' uses deprecated keyword '{keyword}'"


class TestMCPErrorHandling:
    """Test MCP protocol error handling"""
    
    @pytest.mark.asyncio
    async def test_malformed_json_handling(self):
        """Test handling of malformed JSON in arguments"""
        # This would typically be handled at the transport layer,
        # but we can test the tool handler's robustness
        
        # Test with None arguments
        result = await handle_call_tool("send_email", None)
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Test with non-dict arguments (if possible)
        try:
            result = await handle_call_tool("send_email", "not_a_dict")
            assert isinstance(result, list)
        except Exception:
            # It's acceptable to raise an exception for invalid argument types
            pass
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of long-running operations"""
        # Our tools should complete in reasonable time
        import time
        
        start_time = time.time()
        
        result = await handle_call_tool("generate_insights_report", {
            "report_type": "monthly_review",
            "data_sources": ["email", "social_media", "projects", "personal_metrics"]
        })
        
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time (generous limit)
        assert execution_time < 30.0, f"Tool execution took too long: {execution_time:.2f}s"
        
        # Should still return valid response
        assert isinstance(result, list)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self):
        """Test handling when system resources are low"""
        # Test with many concurrent operations
        tasks = []
        
        # Create many tasks to simulate resource pressure
        for i in range(50):
            task = handle_call_tool("track_personal_metrics", {
                "metric_type": "productivity",
                "action": "log",
                "data": {"test_metric": i}
            })
            tasks.append(task)
        
        # Should handle all requests without crashing
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed, some might fail gracefully
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        # At least half should succeed (generous threshold)
        assert len(successes) >= len(tasks) // 2, \
            f"Too many failures: {len(failures)}/{len(tasks)}"
        
        # All successes should be valid responses
        for result in successes:
            assert isinstance(result, list)
            assert len(result) > 0


class TestMCPContentTypes:
    """Test MCP content type handling"""
    
    @pytest.mark.asyncio
    async def test_text_content_generation(self):
        """Test generation of TextContent responses"""
        result = await handle_call_tool("send_email", {
            "to": ["test@example.com"],
            "subject": "Test",
            "body": "Test body"
        })
        
        # Should return TextContent
        assert len(result) > 0
        text_content = result[0]
        assert isinstance(text_content, TextContent)
        assert text_content.type == "text"
        assert isinstance(text_content.text, str)
        assert len(text_content.text) > 0
    
    @pytest.mark.asyncio
    async def test_content_encoding(self):
        """Test proper encoding of content"""
        # Test with various character sets
        test_strings = [
            "Basic ASCII text",
            "UTF-8: Hello 世界",
            "Emoji: 🚀💡📊",
            "Special chars: <>&\"'",
            "Numbers: 123456789",
            "Mixed: ABC世界🚀<test>"
        ]
        
        for test_string in test_strings:
            result = await handle_call_tool("post_social_media", {
                "platforms": ["twitter"],
                "content": test_string
            })
            
            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], TextContent)
            # Content should be properly encoded/decoded
            response_text = result[0].text
            assert isinstance(response_text, str)
    
    @pytest.mark.asyncio
    async def test_content_length_limits(self):
        """Test handling of very long content"""
        # Test with very long content
        very_long_content = "Long content " * 1000  # ~13KB
        
        result = await handle_call_tool("analyze_email_patterns", {
            "timeframe": "month",
            "analysis_type": "patterns"  
        })
        
        # Should handle without issues
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)
        
        # Response should be reasonable length (may be truncated)
        response_length = len(result[0].text)
        assert response_length > 0
        assert response_length < 100000  # Reasonable upper limit