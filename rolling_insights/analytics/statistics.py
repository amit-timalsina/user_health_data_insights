"""Statistical analysis functions for Rolling Insights.

Pure functions for statistical calculations and analysis with no side effects
or dependencies on external frameworks.
"""

from typing import List, Optional, TypeVar, Dict
import numpy as np
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

# ==================== CONSTANTS ====================

# Benchmark data (typical ranges for healthy adults)
SLEEP_BENCHMARKS: Dict[str, BenchmarkType] = {
    "deep_sleep_percentage": {"min": 13.0, "max": 23.0, "median": 18.0},
    "rem_sleep_percentage": {"min": 12.0, "max": 21.0, "median": 16.5},
    "sleep_efficiency": {"min": 0.85, "max": 0.95, "median": 0.90},
    "screen_time_minutes": {"min": 60.0, "median": 180.0, "max": 300.0},
}

# Sleep quality score configuration
SLEEP_QUALITY_WEIGHTS = {
    "efficiency": 0.30,
    "deep_sleep": 0.30,
    "rem_sleep": 0.25,
    "consistency": 0.15,
}

# Sleep quality thresholds
SLEEP_QUALITY_THRESHOLDS = {
    # Efficiency scoring thresholds (research shows 80% is minimum healthy)
    "min_efficiency": 0.80,  # Minimum acceptable efficiency (was 0.7)
    "optimal_efficiency": 0.95,  # Optimal efficiency target
    # Deep sleep scoring (percentage of total sleep)
    "min_deep_sleep_pct": 13.0,  # Min percentage for healthy deep sleep
    "optimal_deep_sleep_pct": 20.0,  # Optimal deep sleep percentage
    # REM sleep scoring
    "min_rem_sleep_pct": 12.0,  # Min percentage for healthy REM sleep
    "optimal_rem_sleep_pct": 20.0,  # Optimal REM sleep percentage
    # Consistency scoring
    "max_variability_minutes": 60.0,  # Maximum acceptable night-to-night variability
}

# Activity benchmarks
ACTIVITY_BENCHMARKS = {
    "daily_recommended_steps": 10000,  # WHO and CDC guideline
    "daily_recommended_calories": 300,  # Average active calorie burn recommendation
    "steps_weight": 0.6,  # Weight given to steps in activity score
    "calories_weight": 0.4,  # Weight given to calories in activity score
}


def calculate_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """Calculate Pearson correlation coefficient between two series."""
    if len(x) != len(y) or len(x) < 2:
        return None

    # Convert to numpy arrays for vectorized operations
    x_arr = np.array(x)
    y_arr = np.array(y)

    # Use numpy's corrcoef function which returns a correlation matrix
    # We want the off-diagonal element [0,1] which is the correlation between x and y
    corr_matrix = np.corrcoef(x_arr, y_arr)
    if np.isnan(corr_matrix[0, 1]):
        return None

    return corr_matrix[0, 1]


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
    # Get weights from constants
    weights = SLEEP_QUALITY_WEIGHTS
    thresholds = SLEEP_QUALITY_THRESHOLDS

    # Calculate component scores (0-100 scale)
    # Efficiency score: Linear scaling from min_efficiency to optimal_efficiency
    efficiency_range = thresholds["optimal_efficiency"] - thresholds["min_efficiency"]
    efficiency_score = min(
        100,
        max(
            0,
            (stats["avg_sleep_efficiency"] - thresholds["min_efficiency"])
            * (100 / efficiency_range),
        ),
    )

    # Deep sleep score: Linear scaling from min to optimal percentage
    deep_sleep_range = (
        thresholds["optimal_deep_sleep_pct"] - thresholds["min_deep_sleep_pct"]
    )
    deep_score = min(
        100,
        max(
            0,
            (
                (stats["deep_sleep_percentage"] - thresholds["min_deep_sleep_pct"])
                / deep_sleep_range
            )
            * 100,
        ),
    )

    # REM sleep score: Linear scaling from min to optimal percentage
    rem_sleep_range = (
        thresholds["optimal_rem_sleep_pct"] - thresholds["min_rem_sleep_pct"]
    )
    rem_score = min(
        100,
        max(
            0,
            (
                (stats["rem_sleep_percentage"] - thresholds["min_rem_sleep_pct"])
                / rem_sleep_range
            )
            * 100,
        ),
    )

    # Consistency score: Lower variability is better (up to the defined threshold)
    variability_ratio = (
        min(stats["sleep_duration_variability"], thresholds["max_variability_minutes"])
        / thresholds["max_variability_minutes"]
    )
    consistency_score = 100 * (1 - variability_ratio)

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
    Weights can be adjusted based on individual health goals.
    """
    # Get benchmarks from constants
    recommended_steps = ACTIVITY_BENCHMARKS["daily_recommended_steps"]
    recommended_active_calories = ACTIVITY_BENCHMARKS["daily_recommended_calories"]
    steps_weight = ACTIVITY_BENCHMARKS["steps_weight"]
    calories_weight = ACTIVITY_BENCHMARKS["calories_weight"]

    # Calculate component scores
    steps_score = min(100, (avg_steps / recommended_steps) * 100)
    calorie_score = min(100, (avg_calories / recommended_active_calories) * 100)

    # Combined score with weighted components
    return (steps_weight * steps_score) + (calories_weight * calorie_score)


def calculate_sleep_stats(metrics: List[SleepMetrics]) -> SleepStats:
    """Calculate statistical measures from sleep metrics."""
    tot = [m.total_sleep_minutes for m in metrics]
    deep = [m.deep_sleep_minutes for m in metrics]
    rem = [m.rem_sleep_minutes for m in metrics]
    light = [m.light_sleep_minutes for m in metrics]
    eff = [m.sleep_efficiency for m in metrics]

    # Convert to numpy arrays for vectorized operations
    tot_arr = np.array(tot)
    deep_arr = np.array(deep)
    rem_arr = np.array(rem)
    light_arr = np.array(light)
    eff_arr = np.array(eff)

    total_minutes = np.sum(tot_arr)
    deep_sleep_pct = np.sum(deep_arr) / total_minutes * 100 if total_minutes else 0
    rem_sleep_pct = np.sum(rem_arr) / total_minutes * 100 if total_minutes else 0
    light_sleep_pct = np.sum(light_arr) / total_minutes * 100 if total_minutes else 0

    # Calculate sleep duration variability (standard deviation)
    sleep_duration_variability = np.std(tot_arr, ddof=1) if len(tot) > 1 else 0.0

    # Calculate percentile rankings
    deep_sleep_percentile = calculate_percentile(
        deep_sleep_pct, SLEEP_BENCHMARKS["deep_sleep_percentage"]
    )
    rem_sleep_percentile = calculate_percentile(
        rem_sleep_pct, SLEEP_BENCHMARKS["rem_sleep_percentage"]
    )
    sleep_efficiency_percentile = calculate_percentile(
        np.mean(eff_arr), SLEEP_BENCHMARKS["sleep_efficiency"]
    )

    stats_dict = {
        "avg_total_sleep_minutes": np.mean(tot_arr),
        "avg_deep_sleep_minutes": np.mean(deep_arr),
        "avg_rem_sleep_minutes": np.mean(rem_arr),
        "avg_light_sleep_minutes": np.mean(light_arr),
        "avg_sleep_efficiency": np.mean(eff_arr),
        "min_total_sleep_minutes": np.min(tot_arr),
        "max_total_sleep_minutes": np.max(tot_arr),
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

    # Convert to numpy arrays
    screen_arr = np.array(screen)
    pickups_arr = np.array(pickups)
    screen_bed_arr = np.array(screen_bed)

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
        np.mean(screen_arr), SLEEP_BENCHMARKS["screen_time_minutes"]
    )

    return PhoneStats(
        avg_screen_time_minutes=np.mean(screen_arr),
        avg_pickups=np.mean(pickups_arr),
        avg_screen_before_bed_minutes=np.mean(screen_bed_arr),
        total_screen_time_minutes=np.sum(screen_arr),
        total_pickups=int(np.sum(pickups_arr)),
        screen_before_bed_total_sleep_correlation=screen_bed_total_corr,
        screen_before_bed_deep_sleep_correlation=screen_bed_deep_corr,
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

    # Convert to numpy arrays
    steps_arr = np.array(steps)
    active_arr = np.array(active)

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
    day_to_day_variability = np.std(steps_arr, ddof=1) if len(steps) > 1 else 0.0

    # Calculate activity intensity score
    avg_steps_value = np.mean(steps_arr) if steps else 0
    avg_active_calories = np.mean(active_arr) if active else 0
    activity_intensity = calculate_activity_intensity_score(
        avg_steps_value, avg_active_calories
    )

    return HealthStats(
        avg_steps=avg_steps_value,
        avg_active_energy_burned=avg_active_calories,
        total_steps=int(np.sum(steps_arr)),
        total_active_energy_burned=float(np.sum(active_arr)),
        steps_total_sleep_correlation=steps_sleep_corr,
        activity_deep_sleep_correlation=activity_deep_corr,
        activity_deep_sleep_variance_explained=activity_deep_variance,
        steps_sleep_efficiency_correlation=steps_efficiency_corr,
        activity_rem_sleep_correlation=activity_rem_corr,
        day_to_day_activity_variability=day_to_day_variability,
        activity_intensity_score=activity_intensity,
    )
