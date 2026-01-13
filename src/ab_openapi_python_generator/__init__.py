"""Alias package to preserve imports when distribution renamed.

This package re-exports the real package located at
`openapi_python_generator` so existing imports like
`import ab_openapi_python_generator` continue to work.
"""
from openapi_python_generator import *  # noqa: F401,F403

# Preserve __all__ if present on the real package
try:
    __all__ = getattr(__import__("openapi_python_generator"), "__all__")
except Exception:
    __all__ = []
