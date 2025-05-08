"""Statistical analysis functions for Rolling Insights.

Advanced statistical calculations for meaningful sleep, phone, and health insights.
"""

from typing import List, Optional, TypeVar, Dict, Tuple, Any, cast, TypedDict
import numpy as np
from scipy import stats  # type: ignore[import-untyped]
from rolling_insights.models import (
    SleepMetrics,
    PhoneMetrics,
    HealthMetrics,
    SleepStats,
    PhoneStats,
    HealthStats,
    InsightItem,
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
    "min_efficiency": 0.80,  # Minimum acceptable efficiency
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

# Correlation strength thresholds for insights
CORRELATION_STRENGTH = {
    "weak": 0.2,
    "moderate": 0.4,
    "strong": 0.6,
    "very_strong": 0.8,
}

# Statistical significance threshold (p-value)
SIGNIFICANCE_THRESHOLD = 0.1  # More permissive for small sample size (7 days)


def calculate_correlation(
    x: List[float], y: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate Pearson correlation coefficient between two series with p-value.

    Returns:
        Tuple containing (correlation coefficient, p-value)
    """
    if len(x) != len(y) or len(x) < 2:
        return None, None

    # Convert to numpy arrays for better statistical computation
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    # Calculate correlation and p-value
    corr, p_value = stats.pearsonr(x_arr, y_arr)

    return corr, p_value


def calculate_variance_explained(x: List[float], y: List[float]) -> float:
    """Calculate R-squared (coefficient of determination) between two variables.

    This represents the proportion of variance in y explained by x.
    """
    corr, _ = calculate_correlation(x, y)
    if corr is None:
        return 0.0
    # R-squared is the square of the correlation coefficient
    return corr**2


def detect_trend(data: List[float]) -> Tuple[float, float, bool]:
    """Detect if there's a significant trend in time series data.

    Returns:
        Tuple containing (slope, p-value, is_significant)
    """
    if len(data) < 3:
        return 0.0, 1.0, False

    # Use linear regression to detect trend
    x = np.arange(len(data))
    y = np.array(data)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    is_significant = p_value < SIGNIFICANCE_THRESHOLD

    return slope, p_value, is_significant


def detect_outliers(data: List[float], threshold: float = 1.5) -> List[int]:
    """Detect outliers in a data series using IQR method.

    Args:
        data: List of values to check for outliers
        threshold: IQR multiplier (default 1.5)

    Returns:
        List of indices where outliers are found
    """
    if len(data) < 4:  # Need reasonable sample size
        return []

    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outliers = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
    return outliers


def calculate_percentile(value: float, benchmark: BenchmarkType) -> float:
    """Calculate approximate percentile ranking based on benchmarks.

    Uses a more sophisticated interpolation based on population distribution.
    """
    min_val = benchmark["min"]
    max_val = benchmark["max"]
    median = benchmark["median"]

    # Handle values outside the range
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 100.0

    # For values below median - use a slightly curved distribution
    if value < median:
        # Non-linear scaling to better approximate actual population distribution
        # Uses a power function to create a more realistic curve
        normalized_value = (value - min_val) / (median - min_val)
        percentile = 50.0 * np.power(normalized_value, 0.9)
        return percentile

    # For values above median
    normalized_value = (value - median) / (max_val - median)
    percentile = 50.0 + 50.0 * np.power(normalized_value, 1.1)
    return percentile


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
    efficiency_score = np.clip(
        (
            (stats["avg_sleep_efficiency"] - thresholds["min_efficiency"])
            * (100 / efficiency_range)
        ),
        0,
        100,
    )

    # Deep sleep score: Linear scaling from min to optimal percentage
    deep_sleep_range = (
        thresholds["optimal_deep_sleep_pct"] - thresholds["min_deep_sleep_pct"]
    )
    deep_score = np.clip(
        (
            (stats["deep_sleep_percentage"] - thresholds["min_deep_sleep_pct"])
            / deep_sleep_range
        )
        * 100,
        0,
        100,
    )

    # REM sleep score: Linear scaling from min to optimal percentage
    rem_sleep_range = (
        thresholds["optimal_rem_sleep_pct"] - thresholds["min_rem_sleep_pct"]
    )
    rem_score = np.clip(
        (
            (stats["rem_sleep_percentage"] - thresholds["min_rem_sleep_pct"])
            / rem_sleep_range
        )
        * 100,
        0,
        100,
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

    # Calculate component scores using a more nuanced approach
    # Uses a modified sigmoid function to give diminishing returns for very high values
    # and steeper penalties for very low values

    # Steps score (ranges from 0-100)
    steps_ratio = avg_steps / recommended_steps
    steps_score = np.minimum(100, 100 * (1 / (1 + np.exp(-5 * (steps_ratio - 0.5)))))

    # Calorie score (ranges from 0-100)
    calorie_ratio = avg_calories / recommended_active_calories
    calorie_score = np.minimum(
        100, 100 * (1 / (1 + np.exp(-5 * (calorie_ratio - 0.5))))
    )

    # Combined score with weighted components
    return (steps_weight * steps_score) + (calories_weight * calorie_score)


class InsightDict(TypedDict):
    """Type definition for insight dictionary."""

    name: str
    correlation: float
    p_value: float
    strength: str
    direction: str
    variance_explained: float
    importance_score: float


def find_key_insights(
    corr_dict: Dict[str, float], p_values: Dict[str, float]
) -> List[InsightDict]:
    """Find statistically significant correlations from a dictionary.

    Args:
        corr_dict: Dictionary of correlation names and values
        p_values: Dictionary of corresponding p-values

    Returns:
        List of dictionaries with key insights, sorted by significance
    """
    insights: List[InsightDict] = []

    for name, corr in corr_dict.items():
        if name in p_values and abs(corr) > CORRELATION_STRENGTH["weak"]:
            p_val = p_values[name]
            is_significant = p_val < SIGNIFICANCE_THRESHOLD

            if is_significant:
                # Determine strength description
                if abs(corr) >= CORRELATION_STRENGTH["very_strong"]:
                    strength = "very strong"
                elif abs(corr) >= CORRELATION_STRENGTH["strong"]:
                    strength = "strong"
                elif abs(corr) >= CORRELATION_STRENGTH["moderate"]:
                    strength = "moderate"
                else:
                    strength = "weak"

                # Determine direction
                direction = "positive" if corr > 0 else "negative"

                # Variance explained
                variance_explained = corr**2 * 100  # percentage

                insights.append(
                    {
                        "name": name,
                        "correlation": corr,
                        "p_value": p_val,
                        "strength": strength,
                        "direction": direction,
                        "variance_explained": variance_explained,
                        "importance_score": abs(corr) * (1 - p_val),  # For sorting
                    }
                )

    # Sort by importance score (strong correlations with low p-values)
    insights.sort(key=lambda x: x["importance_score"], reverse=True)
    return insights


class SleepStatsDict(TypedDict):
    """Type definition for sleep stats dictionary to pass to SleepStats."""

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
    sleep_duration_variability: float
    deep_sleep_percentile: float
    rem_sleep_percentile: float
    sleep_efficiency_percentile: float
    total_sleep_trend: float
    total_sleep_trend_significant: bool
    deep_sleep_trend: float
    deep_sleep_trend_significant: bool
    rem_sleep_trend: float
    rem_sleep_trend_significant: bool
    has_outlier_nights: bool
    outlier_night_indices: List[int]
    sleep_quality_score: float


def calculate_sleep_stats(metrics: List[SleepMetrics]) -> SleepStats:
    """Calculate statistical measures from sleep metrics."""
    # Extract metrics
    tot = [m.total_sleep_minutes for m in metrics]
    deep = [m.deep_sleep_minutes for m in metrics]
    rem = [m.rem_sleep_minutes for m in metrics]
    light = [m.light_sleep_minutes for m in metrics]
    eff = [m.sleep_efficiency for m in metrics]

    # Convert to numpy arrays for better calculations
    tot_arr = np.array(tot)
    deep_arr = np.array(deep)
    rem_arr = np.array(rem)
    light_arr = np.array(light)
    eff_arr = np.array(eff)

    # Calculate percentages
    total_minutes = np.sum(tot_arr)
    deep_sleep_pct = (np.sum(deep_arr) / total_minutes * 100) if total_minutes else 0
    rem_sleep_pct = (np.sum(rem_arr) / total_minutes * 100) if total_minutes else 0
    light_sleep_pct = (np.sum(light_arr) / total_minutes * 100) if total_minutes else 0

    # Calculate sleep duration variability (standard deviation)
    sleep_duration_variability = np.std(tot_arr, ddof=1) if len(tot) > 1 else 0.0

    # Detect trends
    total_trend, total_p, total_significant = detect_trend(tot)
    deep_trend, deep_p, deep_significant = detect_trend(deep)
    rem_trend, rem_p, rem_significant = detect_trend(rem)

    # Find outliers
    tot_outliers = detect_outliers(tot)

    # Calculate percentile rankings
    deep_sleep_percentile = calculate_percentile(
        deep_sleep_pct, SLEEP_BENCHMARKS["deep_sleep_percentage"]
    )
    rem_sleep_percentile = calculate_percentile(
        rem_sleep_pct, SLEEP_BENCHMARKS["rem_sleep_percentage"]
    )
    sleep_efficiency_percentile = calculate_percentile(
        float(np.mean(eff_arr)), SLEEP_BENCHMARKS["sleep_efficiency"]
    )

    # Create a dictionary with the right types for the model
    stats_dict: Dict[str, Any] = {
        "avg_total_sleep_minutes": float(np.mean(tot_arr)),
        "avg_deep_sleep_minutes": float(np.mean(deep_arr)),
        "avg_rem_sleep_minutes": float(np.mean(rem_arr)),
        "avg_light_sleep_minutes": float(np.mean(light_arr)),
        "avg_sleep_efficiency": float(np.mean(eff_arr)),
        "min_total_sleep_minutes": float(np.min(tot_arr)),
        "max_total_sleep_minutes": float(np.max(tot_arr)),
        "deep_sleep_percentage": float(deep_sleep_pct),
        "rem_sleep_percentage": float(rem_sleep_pct),
        "light_sleep_percentage": float(light_sleep_pct),
        "sleep_duration_variability": float(sleep_duration_variability),
        "deep_sleep_percentile": float(deep_sleep_percentile),
        "rem_sleep_percentile": float(rem_sleep_percentile),
        "sleep_efficiency_percentile": float(sleep_efficiency_percentile),
        # Add trend data
        "total_sleep_trend": float(total_trend),
        "total_sleep_trend_significant": bool(total_significant),
        "deep_sleep_trend": float(deep_trend),
        "deep_sleep_trend_significant": bool(deep_significant),
        "rem_sleep_trend": float(rem_trend),
        "rem_sleep_trend_significant": bool(rem_significant),
        # Add outlier detection
        "has_outlier_nights": bool(len(tot_outliers) > 0),
        "outlier_night_indices": list(tot_outliers),
    }

    # Calculate the composite sleep quality score
    # Create a filtered dict with only float values for the sleep quality function
    sleep_quality_dict = {
        k: float(v)
        for k, v in stats_dict.items()
        if k
        in [
            "avg_sleep_efficiency",
            "deep_sleep_percentage",
            "rem_sleep_percentage",
            "sleep_duration_variability",
        ]
    }
    sleep_quality_score = calculate_sleep_quality_score(
        cast(Dict[str, float], sleep_quality_dict)
    )
    stats_dict["sleep_quality_score"] = sleep_quality_score

    # Cast the dict to the properly typed dict and pass to SleepStats
    typed_dict = cast(SleepStatsDict, stats_dict)
    return SleepStats(**typed_dict)


class PhoneStatsDict(TypedDict):
    """Type definition for phone stats dictionary to pass to PhoneStats."""

    avg_screen_time_minutes: float
    avg_pickups: float
    avg_screen_before_bed_minutes: float
    total_screen_time_minutes: int
    total_pickups: int
    screen_before_bed_total_sleep_correlation: float
    screen_before_bed_deep_sleep_correlation: float
    screen_before_bed_total_sleep_variance_explained: float
    screen_before_bed_deep_sleep_variance_explained: float
    screen_time_percentile: float
    screen_before_bed_rem_sleep_correlation: float
    pickups_sleep_correlation: float
    morning_pickup_sleep_quality_correlation: float
    screen_time_trend: float
    screen_time_trend_significant: bool
    screen_time_min: float
    screen_time_max: float
    pickups_min: float
    pickups_max: float
    key_phone_insights: List[InsightItem]


class HealthStatsDict(TypedDict):
    """Type definition for health stats dictionary to pass to HealthStats."""

    avg_steps: float
    avg_active_energy_burned: float
    total_steps: int
    total_active_energy_burned: float
    steps_total_sleep_correlation: float
    activity_deep_sleep_correlation: float
    activity_deep_sleep_variance_explained: float
    steps_sleep_efficiency_correlation: float
    activity_rem_sleep_correlation: float
    day_to_day_activity_variability: float
    activity_intensity_score: float
    steps_trend: float
    steps_trend_significant: bool
    activity_trend: float
    activity_trend_significant: bool
    steps_min: float
    steps_max: float
    activity_min: float
    activity_max: float
    key_health_insights: List[InsightItem]


def calculate_phone_stats(
    metrics: List[PhoneMetrics], sleep_metrics: List[SleepMetrics]
) -> PhoneStats:
    """Calculate statistical measures from phone usage metrics."""
    # Extract metrics as numpy arrays
    screen = np.array([float(m.screen_time_minutes) for m in metrics])
    pickups = np.array([float(m.pickups) for m in metrics])
    screen_bed = np.array([float(m.screen_time_before_bed_minutes) for m in metrics])
    first_pickup = np.array(
        [float(m.first_pickup_after_wakeup_minutes) for m in metrics]
    )

    # Extract sleep metrics
    total_sleep = np.array([m.total_sleep_minutes for m in sleep_metrics])
    deep_sleep = np.array([m.deep_sleep_minutes for m in sleep_metrics])
    rem_sleep = np.array([m.rem_sleep_minutes for m in sleep_metrics])
    sleep_efficiency = np.array([m.sleep_efficiency for m in sleep_metrics])

    # Calculate correlations with p-values
    correlations: Dict[str, float] = {}
    p_values: Dict[str, float] = {}

    # Screen time before bed correlations
    screen_bed_total_corr, screen_bed_total_p = calculate_correlation(
        screen_bed.tolist(), total_sleep.tolist()
    )
    screen_bed_deep_corr, screen_bed_deep_p = calculate_correlation(
        screen_bed.tolist(), deep_sleep.tolist()
    )
    screen_bed_rem_corr, screen_bed_rem_p = calculate_correlation(
        screen_bed.tolist(), rem_sleep.tolist()
    )

    correlations["screen_time_before_bed_total_sleep"] = screen_bed_total_corr or 0
    correlations["screen_time_before_bed_deep_sleep"] = screen_bed_deep_corr or 0
    correlations["screen_time_before_bed_rem_sleep"] = screen_bed_rem_corr or 0

    p_values["screen_time_before_bed_total_sleep"] = screen_bed_total_p or 1.0
    p_values["screen_time_before_bed_deep_sleep"] = screen_bed_deep_p or 1.0
    p_values["screen_time_before_bed_rem_sleep"] = screen_bed_rem_p or 1.0

    # Other correlations
    pickups_sleep_corr, pickups_sleep_p = calculate_correlation(
        pickups.tolist(), total_sleep.tolist()
    )
    morning_pickup_corr, morning_pickup_p = calculate_correlation(
        first_pickup.tolist(), sleep_efficiency.tolist()
    )

    correlations["pickups_total_sleep"] = pickups_sleep_corr or 0
    correlations["morning_pickup_sleep_efficiency"] = morning_pickup_corr or 0

    p_values["pickups_total_sleep"] = pickups_sleep_p or 1.0
    p_values["morning_pickup_sleep_efficiency"] = morning_pickup_p or 1.0

    # Identify key insights from correlations
    key_insights_dicts = find_key_insights(correlations, p_values)
    key_insights = [InsightItem(**insight) for insight in key_insights_dicts[:3]]

    # Calculate variance explained
    screen_bed_total_variance = calculate_variance_explained(
        screen_bed.tolist(), total_sleep.tolist()
    )
    screen_bed_deep_variance = calculate_variance_explained(
        screen_bed.tolist(), deep_sleep.tolist()
    )

    # Calculate trends
    screen_trend, screen_trend_p, screen_trend_significant = detect_trend(
        screen.tolist()
    )

    # Calculate screen time percentile
    screen_time_percentile = calculate_percentile(
        float(np.mean(screen)), SLEEP_BENCHMARKS["screen_time_minutes"]
    )

    # Build the stats dictionary with proper typing
    stats_dict: PhoneStatsDict = {
        "avg_screen_time_minutes": float(np.mean(screen)),
        "avg_pickups": float(np.mean(pickups)),
        "avg_screen_before_bed_minutes": float(np.mean(screen_bed)),
        "total_screen_time_minutes": int(np.sum(screen)),
        "total_pickups": int(np.sum(pickups)),
        "screen_before_bed_total_sleep_correlation": correlations[
            "screen_time_before_bed_total_sleep"
        ],
        "screen_before_bed_deep_sleep_correlation": correlations[
            "screen_time_before_bed_deep_sleep"
        ],
        "screen_before_bed_total_sleep_variance_explained": screen_bed_total_variance,
        "screen_before_bed_deep_sleep_variance_explained": screen_bed_deep_variance,
        "screen_time_percentile": screen_time_percentile,
        "screen_before_bed_rem_sleep_correlation": correlations[
            "screen_time_before_bed_rem_sleep"
        ],
        "pickups_sleep_correlation": correlations["pickups_total_sleep"],
        "morning_pickup_sleep_quality_correlation": correlations[
            "morning_pickup_sleep_efficiency"
        ],
        # Additional statistics
        "screen_time_trend": float(screen_trend),
        "screen_time_trend_significant": bool(screen_trend_significant),
        "screen_time_min": float(np.min(screen)),
        "screen_time_max": float(np.max(screen)),
        "pickups_min": float(np.min(pickups)),
        "pickups_max": float(np.max(pickups)),
        "key_phone_insights": key_insights,
    }

    return PhoneStats(**stats_dict)


def calculate_health_stats(
    metrics: List[HealthMetrics], sleep_metrics: List[SleepMetrics]
) -> HealthStats:
    """Calculate statistical measures from health metrics."""
    # Extract metrics as numpy arrays
    steps = np.array([float(m.total_steps) for m in metrics])
    active = np.array([float(m.active_energy_burned) for m in metrics])

    # Extract sleep metrics
    total_sleep = np.array([m.total_sleep_minutes for m in sleep_metrics])
    deep_sleep = np.array([m.deep_sleep_minutes for m in sleep_metrics])
    rem_sleep = np.array([m.rem_sleep_minutes for m in sleep_metrics])
    sleep_efficiency = np.array([m.sleep_efficiency for m in sleep_metrics])

    # Calculate correlations with p-values
    correlations: Dict[str, float] = {}
    p_values: Dict[str, float] = {}

    # Steps correlations
    steps_sleep_corr, steps_sleep_p = calculate_correlation(
        steps.tolist(), total_sleep.tolist()
    )
    steps_efficiency_corr, steps_efficiency_p = calculate_correlation(
        steps.tolist(), sleep_efficiency.tolist()
    )

    correlations["steps_total_sleep"] = steps_sleep_corr or 0
    correlations["steps_sleep_efficiency"] = steps_efficiency_corr or 0

    p_values["steps_total_sleep"] = steps_sleep_p or 1.0
    p_values["steps_sleep_efficiency"] = steps_efficiency_p or 1.0

    # Activity correlations
    activity_deep_corr, activity_deep_p = calculate_correlation(
        active.tolist(), deep_sleep.tolist()
    )
    activity_rem_corr, activity_rem_p = calculate_correlation(
        active.tolist(), rem_sleep.tolist()
    )

    correlations["activity_deep_sleep"] = activity_deep_corr or 0
    correlations["activity_rem_sleep"] = activity_rem_corr or 0

    p_values["activity_deep_sleep"] = activity_deep_p or 1.0
    p_values["activity_rem_sleep"] = activity_rem_p or 1.0

    # Identify key insights from correlations
    key_insights_dicts = find_key_insights(correlations, p_values)
    key_insights = [InsightItem(**insight) for insight in key_insights_dicts[:3]]

    # Calculate variance explained
    activity_deep_variance = calculate_variance_explained(
        active.tolist(), deep_sleep.tolist()
    )

    # Calculate activity variability (day-to-day consistency)
    day_to_day_variability = float(np.std(steps, ddof=1)) if len(steps) > 1 else 0.0

    # Detect activity trends
    steps_trend, steps_p, steps_significant = detect_trend(steps.tolist())
    active_trend, active_p, active_significant = detect_trend(active.tolist())

    # Calculate activity intensity score
    avg_steps_value = float(np.mean(steps)) if len(steps) > 0 else 0
    avg_active_calories = float(np.mean(active)) if len(active) > 0 else 0
    activity_intensity = calculate_activity_intensity_score(
        avg_steps_value, avg_active_calories
    )

    # Build the stats dictionary with proper typing
    stats_dict: HealthStatsDict = {
        "avg_steps": avg_steps_value,
        "avg_active_energy_burned": avg_active_calories,
        "total_steps": int(np.sum(steps)),
        "total_active_energy_burned": float(np.sum(active)),
        "steps_total_sleep_correlation": correlations["steps_total_sleep"],
        "activity_deep_sleep_correlation": correlations["activity_deep_sleep"],
        "activity_deep_sleep_variance_explained": activity_deep_variance,
        "steps_sleep_efficiency_correlation": correlations["steps_sleep_efficiency"],
        "activity_rem_sleep_correlation": correlations["activity_rem_sleep"],
        "day_to_day_activity_variability": day_to_day_variability,
        "activity_intensity_score": activity_intensity,
        # Additional statistics
        "steps_trend": float(steps_trend),
        "steps_trend_significant": bool(steps_significant),
        "activity_trend": float(active_trend),
        "activity_trend_significant": bool(active_significant),
        "steps_min": float(np.min(steps)),
        "steps_max": float(np.max(steps)),
        "activity_min": float(np.min(active)),
        "activity_max": float(np.max(active)),
        "key_health_insights": key_insights,
    }

    return HealthStats(**stats_dict)
