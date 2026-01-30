"""
Public API for model generation.

This module provides both:
- High-level Python API (builders, converters)
- REST API for HTTP access

Python API Usage:
    from model_generation import ManualBuilder, ParametricBuilder, quick_build

REST API Usage:
    from model_generation.api import ModelGenerationAPI, FlaskAdapter

    # With Flask
    from flask import Flask
    app = Flask(__name__)
    api = ModelGenerationAPI()
    FlaskAdapter(api).register(app)

    # With FastAPI
    from fastapi import FastAPI
    app = FastAPI()
    api = ModelGenerationAPI()
    FastAPIAdapter(api).register(app)
"""

# Re-export main API components from package root
try:
    from model_generation import (
        BuildResult,
        ManualBuilder,
        ParametricBuilder,
        quick_build,
        quick_urdf,
    )
except ImportError:
    # If package root not available, define stubs
    BuildResult = None
    ManualBuilder = None
    ParametricBuilder = None
    quick_build = None
    quick_urdf = None

# REST API components
from model_generation.api.rest_api import (
    APIRequest,
    APIResponse,
    FastAPIAdapter,
    FlaskAdapter,
    HTTPMethod,
    ModelGenerationAPI,
    Route,
)

__all__ = [
    # Python API
    "ManualBuilder",
    "ParametricBuilder",
    "BuildResult",
    "quick_urdf",
    "quick_build",
    # REST API
    "ModelGenerationAPI",
    "APIRequest",
    "APIResponse",
    "HTTPMethod",
    "Route",
    "FlaskAdapter",
    "FastAPIAdapter",
]
