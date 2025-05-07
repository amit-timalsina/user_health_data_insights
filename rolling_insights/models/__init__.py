"""Contains Pydantic models representing the core business entities
for the application, organized by functional groupings:

- raw_data: Models for raw input data structures
- metrics: Models for metrics (sleep, phone, health) gotten from processing the raw_data
- insights: Models for statistics and insight payloads
"""

from .raw_data import RawHealthData, RawDataPoint
from .metrics import SleepMetrics, PhoneMetrics, HealthMetrics
from .insights import (
    SleepStats,
    PhoneStats,
    HealthStats,
    InsightPeriod,
    BaseInsightPayload,
    SleepInsightPayload,
    PhoneInsightPayload,
    HealthInsightPayload,
    InsightPayload,
)

__all__ = [
    # Raw data models
    "RawHealthData",
    "RawDataPoint",
    # Domain metrics
    "SleepMetrics",
    "PhoneMetrics",
    "HealthMetrics",
    # Insight models
    "SleepStats",
    "PhoneStats",
    "HealthStats",
    "InsightPeriod",
    "BaseInsightPayload",
    "SleepInsightPayload",
    "PhoneInsightPayload",
    "HealthInsightPayload",
    "InsightPayload",
]
