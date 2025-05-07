"""Services package for Rolling Insights.

This package contains services that implement business logic by coordinating
between llm, storage, and analytics.
"""

from .llm_service import LLMService

__all__ = ("LLMService",)
