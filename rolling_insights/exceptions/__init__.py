"""Exception classes for Rolling Insights."""

from rolling_insights.exceptions.base import InsightError
from rolling_insights.exceptions.data import DataLoadError, DataValidationError
from rolling_insights.exceptions.insights import InsightGenerationError

__all__ = (
    "InsightError",
    "DataLoadError",
    "DataValidationError",
    "InsightGenerationError",
)
