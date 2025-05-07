"""Insight-related exceptions for Rolling Insights.

This module contains exceptions related to insight generation.
"""

from typing import Optional
from rolling_insights.exceptions.base import InsightError


class InsightGenerationError(InsightError):
    """Exception raised when insight generation fails."""

    def __init__(
        self,
        message: str,
        insight_type: Optional[str] = None,
        error: Optional[Exception] = None,
    ):
        self.insight_type = insight_type
        self.original_error = error
        super_message = f"{message}"
        if insight_type:
            super_message += f" (Type: {insight_type})"
        if error:
            super_message += f" - {str(error)}"
        super().__init__(super_message)
