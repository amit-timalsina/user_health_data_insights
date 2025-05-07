"""Statistical analysis functions for Rolling Insights.

Pure functions for statistical calculations and analysis with no side effects
or dependencies on external frameworks.
"""

from typing import List, Optional, TypeVar, Dict
import statistics
import math
from rolling_insights.models import (
    SleepMetrics,
    PhoneMetrics,
    HealthMetrics,
    SleepStats,
    PhoneStats,
    HealthStats,
)


T = TypeVar("T")

# Type alias for benchmark data structure
BenchmarkType = Dict[str, float]

# Benchmark data (typical ranges for healthy adults)
SLEEP_BENCHMARKS: Dict[str, BenchmarkType] = {
    "deep_sleep_percentage": {"min": 13.0, "max": 23.0, "median": 18.0},
    "rem_sleep_percentage": {"min": 12.0, "max": 21.0, "median": 16.5},
    "sleep_efficiency": {"min": 0.85, "max": 0.95, "median": 0.90},
    "screen_time_minutes": {"min": 60.0, "median": 180.0, "max": 300.0},
}


def calculate_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """Calculate Pearson correlation coefficient between two series."""
    if len(x) != len(y) or len(x) < 2:
        return None

    # Pure Python implementation to avoid numpy dependency in domain layer
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sq = sum(i**2 for i in x)
    sum_y_sq = sum(i**2 for i in y)

    sum_xy = sum(x[i] * y[i] for i in range(n))

    # Calculate Pearson correlation
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))

    if denominator == 0:
        return None

    return numerator / denominator


def calculate_variance_explained(x: List[float], y: List[float]) -> float:
    """Calculate R-squared (coefficient of determination) between two variables.

    This represents the proportion of variance in y explained by x.
    """
    corr = calculate_correlation(x, y)
    if corr is None:
        return 0.0
    # R-squared is the square of the correlation coefficient
    return corr**2


def calculate_percentile(value: float, benchmark: BenchmarkType) -> float:
    """Calculate approximate percentile ranking based on benchmarks.

    This is a simplified approach using min, median, and max values from benchmarks
    to approximate where a value falls within a typical distribution.
    """
    min_val = benchmark["min"]
    max_val = benchmark["max"]
    median = benchmark["median"]

    # Handle values outside the range
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 100.0

    # For values below median
    if value < median:
        return 50.0 * (value - min_val) / (median - min_val)

    # For values above median
    return 50.0 + 50.0 * (value - median) / (max_val - median)


def calculate_sleep_quality_score(stats: Dict[str, float]) -> float:
    """Calculate a composite sleep quality score based on multiple metrics.

    The score is scaled from 0-100, with higher being better.
    """
    # Weights for different components
    weights = {
        "efficiency": 0.3,
        "deep_sleep": 0.3,
        "rem_sleep": 0.25,
        "consistency": 0.15,
    }

    # Calculate component scores (0-100 scale)
    efficiency_score = min(100, max(0, (stats["avg_sleep_efficiency"] - 0.7) * 333.3))
    deep_score = min(100, max(0, stats["deep_sleep_percentage"] * 5))
    rem_score = min(100, max(0, stats["rem_sleep_percentage"] * 5))

    # Lower variability is better (up to a point)
    variability_factor = min(60, stats["sleep_duration_variability"]) / 60
    consistency_score = 100 * (1 - variability_factor)

    # Weighted sum of components
    return (
        weights["efficiency"] * efficiency_score
        + weights["deep_sleep"] * deep_score
        + weights["rem_sleep"] * rem_score
        + weights["consistency"] * consistency_score
    )


def calculate_activity_intensity_score(avg_steps: float, avg_calories: float) -> float:
    """Calculate a score for physical activity intensity relative to recommendations.

    The score is scaled from 0-100, with 100 representing optimal activity.
    """
    # Benchmarks based on general health recommendations
    recommended_steps = 10000  # Daily recommended steps
    recommended_active_calories = 300  # Daily active calorie burn for general health

    # Calculate component scores
    steps_score = min(100, (avg_steps / recommended_steps) * 100)
    calorie_score = min(100, (avg_calories / recommended_active_calories) * 100)

    # Combined score with equal weights
    return (steps_score + calorie_score) / 2


def calculate_sleep_stats(metrics: List[SleepMetrics]) -> SleepStats:
    """Calculate statistical measures from sleep metrics."""
    tot = [m.total_sleep_minutes for m in metrics]
    deep = [m.deep_sleep_minutes for m in metrics]
    rem = [m.rem_sleep_minutes for m in metrics]
    light = [m.light_sleep_minutes for m in metrics]
    eff = [m.sleep_efficiency for m in metrics]

    total_minutes = sum(tot)
    deep_sleep_pct = sum(deep) / total_minutes * 100 if total_minutes else 0
    rem_sleep_pct = sum(rem) / total_minutes * 100 if total_minutes else 0
    light_sleep_pct = sum(light) / total_minutes * 100 if total_minutes else 0

    # Calculate sleep duration variability (standard deviation)
    sleep_duration_variability = statistics.stdev(tot) if len(tot) > 1 else 0.0

    # Calculate percentile rankings
    deep_sleep_percentile = calculate_percentile(
        deep_sleep_pct, SLEEP_BENCHMARKS["deep_sleep_percentage"]
    )
    rem_sleep_percentile = calculate_percentile(
        rem_sleep_pct, SLEEP_BENCHMARKS["rem_sleep_percentage"]
    )
    sleep_efficiency_percentile = calculate_percentile(
        statistics.mean(eff), SLEEP_BENCHMARKS["sleep_efficiency"]
    )

    stats_dict = {
        "avg_total_sleep_minutes": statistics.mean(tot),
        "avg_deep_sleep_minutes": statistics.mean(deep),
        "avg_rem_sleep_minutes": statistics.mean(rem),
        "avg_light_sleep_minutes": statistics.mean(light),
        "avg_sleep_efficiency": statistics.mean(eff),
        "min_total_sleep_minutes": min(tot),
        "max_total_sleep_minutes": max(tot),
        "deep_sleep_percentage": deep_sleep_pct,
        "rem_sleep_percentage": rem_sleep_pct,
        "light_sleep_percentage": light_sleep_pct,
        "sleep_duration_variability": sleep_duration_variability,
        "deep_sleep_percentile": deep_sleep_percentile,
        "rem_sleep_percentile": rem_sleep_percentile,
        "sleep_efficiency_percentile": sleep_efficiency_percentile,
    }

    # Calculate the composite sleep quality score
    sleep_quality_score = calculate_sleep_quality_score(stats_dict)
    stats_dict["sleep_quality_score"] = sleep_quality_score

    return SleepStats(**stats_dict)


def calculate_phone_stats(
    metrics: List[PhoneMetrics], sleep_metrics: List[SleepMetrics]
) -> PhoneStats:
    """Calculate statistical measures from phone usage metrics."""
    screen = [float(m.screen_time_minutes) for m in metrics]
    pickups = [float(m.pickups) for m in metrics]
    screen_bed = [float(m.screen_time_before_bed_minutes) for m in metrics]
    first_pickup = [float(m.first_pickup_after_wakeup_minutes) for m in metrics]

    # Calculate correlations with sleep
    total_sleep = [m.total_sleep_minutes for m in sleep_metrics]
    deep_sleep = [m.deep_sleep_minutes for m in sleep_metrics]
    rem_sleep = [m.rem_sleep_minutes for m in sleep_metrics]
    sleep_efficiency = [m.sleep_efficiency for m in sleep_metrics]

    # Screen time before bed correlations
    screen_bed_total_corr = calculate_correlation(screen_bed, total_sleep) or 0
    screen_bed_deep_corr = calculate_correlation(screen_bed, deep_sleep) or 0
    screen_bed_rem_corr = calculate_correlation(screen_bed, rem_sleep) or 0

    # Other correlations
    pickups_sleep_corr = calculate_correlation(pickups, total_sleep) or 0
    morning_pickup_corr = calculate_correlation(first_pickup, sleep_efficiency) or 0

    # Calculate variance explained (R-squared)
    screen_bed_total_variance = calculate_variance_explained(screen_bed, total_sleep)
    screen_bed_deep_variance = calculate_variance_explained(screen_bed, deep_sleep)

    # Calculate screen time percentile
    screen_time_percentile = calculate_percentile(
        statistics.mean(screen), SLEEP_BENCHMARKS["screen_time_minutes"]
    )

    return PhoneStats(
        avg_screen_time_minutes=statistics.mean(screen),
        avg_pickups=statistics.mean(pickups),
        avg_screen_before_bed_minutes=statistics.mean(screen_bed),
        total_screen_time_minutes=sum([m.screen_time_minutes for m in metrics]),
        total_pickups=sum([m.pickups for m in metrics]),
        screen_before_bed_total_sleep_correlation=screen_bed_total_corr,
        screen_before_bed_deep_sleep_correlation=screen_bed_deep_corr,
        # New metrics
        screen_before_bed_total_sleep_variance_explained=screen_bed_total_variance,
        screen_before_bed_deep_sleep_variance_explained=screen_bed_deep_variance,
        screen_time_percentile=screen_time_percentile,
        screen_before_bed_rem_sleep_correlation=screen_bed_rem_corr,
        pickups_sleep_correlation=pickups_sleep_corr,
        morning_pickup_sleep_quality_correlation=morning_pickup_corr,
    )


def calculate_health_stats(
    metrics: List[HealthMetrics], sleep_metrics: List[SleepMetrics]
) -> HealthStats:
    """Calculate statistical measures from health metrics."""
    steps = [float(m.total_steps) for m in metrics]
    active = [float(m.active_energy_burned) for m in metrics]

    # Calculate correlations with sleep
    total_sleep = [m.total_sleep_minutes for m in sleep_metrics]
    deep_sleep = [m.deep_sleep_minutes for m in sleep_metrics]
    rem_sleep = [m.rem_sleep_minutes for m in sleep_metrics]
    sleep_efficiency = [m.sleep_efficiency for m in sleep_metrics]

    # Basic correlations
    steps_sleep_corr = calculate_correlation(steps, total_sleep) or 0
    activity_deep_corr = calculate_correlation(active, deep_sleep) or 0

    # Additional correlations
    steps_efficiency_corr = calculate_correlation(steps, sleep_efficiency) or 0
    activity_rem_corr = calculate_correlation(active, rem_sleep) or 0

    # Calculate variance explained
    activity_deep_variance = calculate_variance_explained(active, deep_sleep)

    # Calculate activity variability (day-to-day consistency)
    day_to_day_variability = statistics.stdev(steps) if len(steps) > 1 else 0.0

    # Calculate activity intensity score
    avg_steps_value = statistics.mean(steps) if steps else 0
    avg_active_calories = statistics.mean(active) if active else 0
    activity_intensity = calculate_activity_intensity_score(
        avg_steps_value, avg_active_calories
    )

    return HealthStats(
        avg_steps=avg_steps_value,
        avg_active_energy_burned=avg_active_calories,
        total_steps=sum([m.total_steps for m in metrics]),
        total_active_energy_burned=sum([m.active_energy_burned for m in metrics]),
        steps_total_sleep_correlation=steps_sleep_corr,
        activity_deep_sleep_correlation=activity_deep_corr,
        # New metrics
        activity_deep_sleep_variance_explained=activity_deep_variance,
        steps_sleep_efficiency_correlation=steps_efficiency_corr,
        activity_rem_sleep_correlation=activity_rem_corr,
        day_to_day_activity_variability=day_to_day_variability,
        activity_intensity_score=activity_intensity,
    )
