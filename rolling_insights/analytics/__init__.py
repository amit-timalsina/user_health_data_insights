"""Analytics package for Rolling Insights.

This package contains pure calculation functions for statistical analysis with no
dependencies on external frameworks.
"""

from .statistics import (
    calculate_correlation,
    calculate_sleep_stats,
    calculate_phone_stats,
    calculate_health_stats,
)

__all__ = [
    "calculate_correlation",
    "calculate_sleep_stats",
    "calculate_phone_stats",
    "calculate_health_stats",
]
