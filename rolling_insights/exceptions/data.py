"""Data-related exceptions for Rolling Insights.

This module contains exceptions related to data loading and validation.
"""

from typing import Optional
from rolling_insights.exceptions.base import InsightError


class DataLoadError(InsightError):
    """Exception raised when there's an error loading data."""

    def __init__(self, message: str, filepath: Optional[str] = None):
        self.filepath = filepath
        super_message = (
            f"{message}" if filepath is None else f"{message} (File: {filepath})"
        )
        super().__init__(super_message)


class DataValidationError(InsightError):
    """Exception raised when data validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super_message = f"{message}" if field is None else f"{message} (Field: {field})"
        super().__init__(super_message)
