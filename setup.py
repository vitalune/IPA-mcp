#!/usr/bin/env python3
"""
Personal Knowledge Assistant MCP Server
A comprehensive MCP server for managing personal information and communications.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
        return requirements

# Read long description from README if it exists
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Personal Knowledge Assistant MCP Server"

setup(
    name="personal-knowledge-mcp",
    version="0.1.0",
    author="MCP Protocol Architect",
    author_email="noreply@example.com",
    description="MCP server for personal knowledge and communication management",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/anthropic/personal-knowledge-mcp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Communications",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "mypy>=1.6.0",
            "flake8>=6.1.0",
        ],
        "websocket": ["websockets>=12.0"],
        "sse": ["sse-starlette>=1.6.5"],
    },
    entry_points={
        "console_scripts": [
            "personal-knowledge-mcp=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)