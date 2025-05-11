"""
Represent statistical summaries and insight payloads that are the main outputs of the Rolling
Insights system.
"""

from datetime import date
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class SleepStats(BaseModel):
    """Statistical measures for sleep metrics."""

    avg_total_sleep_minutes: float
    avg_deep_sleep_minutes: float
    avg_rem_sleep_minutes: float
    avg_light_sleep_minutes: float
    avg_sleep_efficiency: float
    min_total_sleep_minutes: float
    max_total_sleep_minutes: float
    deep_sleep_percentage: float
    rem_sleep_percentage: float
    light_sleep_percentage: float
    # Additional meaningful metrics
    sleep_duration_variability: float = 0.0  # Standard deviation of sleep duration
    deep_sleep_percentile: float = 0.0  # Relative to healthy adult benchmarks
    rem_sleep_percentile: float = 0.0
    sleep_efficiency_percentile: float = 0.0
    sleep_quality_score: float = 0.0  # Composite score based on multiple metrics
    # New trend analysis fields
    total_sleep_trend: float = 0.0  # Slope of the trend line
    total_sleep_trend_significant: bool = (
        False  # Whether trend is statistically significant
    )
    deep_sleep_trend: float = 0.0
    deep_sleep_trend_significant: bool = False
    rem_sleep_trend: float = 0.0
    rem_sleep_trend_significant: bool = False
    # Outlier detection
    has_outlier_nights: bool = False  # Whether any outlier nights were detected
    outlier_night_indices: List[int] = Field(
        default_factory=list
    )  # Indices of outlier nights


class InsightItem(BaseModel):
    """Represents a statistically significant correlation with contextual information."""

    name: str  # Name/identifier of the correlation
    correlation: float  # Correlation coefficient value
    p_value: float  # Statistical significance (p-value)
    strength: str  # Qualitative description (weak, moderate, strong, very strong)
    direction: str  # positive or negative
    variance_explained: float  # Percentage of variance explained (R-squared * 100)
    importance_score: float  # Metric for ranking insights


class PhoneStats(BaseModel):
    """Statistical measures for phone usage metrics."""

    avg_screen_time_minutes: float
    avg_pickups: float
    avg_screen_before_bed_minutes: float
    total_screen_time_minutes: int
    total_pickups: int
    screen_before_bed_total_sleep_correlation: float
    screen_before_bed_deep_sleep_correlation: float
    # Additional meaningful metrics
    screen_before_bed_total_sleep_variance_explained: float = 0.0  # R-squared value
    screen_before_bed_deep_sleep_variance_explained: float = 0.0  # R-squared value
    screen_time_percentile: float = 0.0  # Relative to average adult usage
    screen_before_bed_rem_sleep_correlation: float = 0.0
    pickups_sleep_correlation: float = 0.0
    morning_pickup_sleep_quality_correlation: float = 0.0
    # New trend analysis fields
    screen_time_trend: float = 0.0  # Slope of screen time trend
    screen_time_trend_significant: bool = (
        False  # Whether trend is statistically significant
    )
    # Additional range statistics
    screen_time_min: float = 0.0
    screen_time_max: float = 0.0
    pickups_min: float = 0.0
    pickups_max: float = 0.0
    # Key insights
    key_phone_insights: List[InsightItem] = Field(default_factory=list)


class HealthStats(BaseModel):
    """Statistical measures for health metrics."""

    avg_steps: float
    avg_active_energy_burned: float
    total_steps: int
    total_active_energy_burned: float
    steps_total_sleep_correlation: float
    activity_deep_sleep_correlation: float
    # Additional meaningful metrics
    activity_deep_sleep_variance_explained: float = 0.0  # R-squared value
    steps_sleep_efficiency_correlation: float = 0.0
    activity_rem_sleep_correlation: float = 0.0
    day_to_day_activity_variability: float = 0.0  # Activity pattern consistency
    activity_intensity_score: float = (
        0.0  # Activity intensity relative to recommended values
    )
    # New trend analysis fields
    steps_trend: float = 0.0  # Slope of steps trend
    steps_trend_significant: bool = False  # Whether trend is statistically significant
    activity_trend: float = 0.0  # Slope of activity trend
    activity_trend_significant: bool = (
        False  # Whether trend is statistically significant
    )
    # Additional range statistics
    steps_min: float = 0.0
    steps_max: float = 0.0
    activity_min: float = 0.0
    activity_max: float = 0.0
    # Key insights
    key_health_insights: List[InsightItem] = Field(default_factory=list)


class InsightPeriod(BaseModel):
    """Time period for insights."""

    start_date: date
    end_date: date


class BaseInsightPayload(BaseModel):
    """Base class for all insight payloads."""

    start_date: date
    end_date: date
    narrative_insights: str


class SleepInsightPayload(BaseInsightPayload):
    """Insight payload for sleep metrics."""

    stats: SleepStats
    daily_data: List[Dict[str, Any]]
    cross_metric_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "period": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
            },
            "stats": self.stats.model_dump(),
            "daily_data": self.daily_data,
            "narrative_insights": self.narrative_insights,
        }


class PhoneInsightPayload(BaseInsightPayload):
    """Insight payload for phone usage metrics."""

    stats: PhoneStats
    daily_data: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "period": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
            },
            "stats": self.stats.model_dump(),
            "daily_data": self.daily_data,
            "narrative_insights": self.narrative_insights,
        }


class HealthInsightPayload(BaseInsightPayload):
    """Insight payload for health metrics."""

    stats: HealthStats
    daily_data: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "period": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
            },
            "stats": self.stats.model_dump(),
            "daily_data": self.daily_data,
            "narrative_insights": self.narrative_insights,
        }


# Backwards compatibility
InsightPayload = Union[SleepInsightPayload, PhoneInsightPayload, HealthInsightPayload]
