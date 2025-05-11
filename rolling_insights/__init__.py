"""
Rolling Insights – Generate data-driven health and sleep insights from wearable device data.

This package analyzes 7-day periods of health, sleep, and phone usage data to produce
meaningful insights for users. It identifies patterns, correlations, and anomalies across
three key domains:

1. Sleep Insights - Analyzes sleep quality, efficiency, and composition (deep, REM, light)
2. Phone Usage Insights - Examines screen time patterns and their impact on sleep quality
3. Health Insights - Evaluates physical activity and its relationship with sleep recovery

The package calculates statistical metrics, identifies correlations between different
domains (e.g., how phone usage affects deep sleep), and leverages AI to generate
narrative explanations of the most significant findings.

Key features:
- Cross-metric correlation analysis (e.g., screen time vs. sleep phases)
- Percentile rankings and variance explained calculations
- Natural language narrative insights powered by LLMs
- Comprehensive 7-day rolling analysis window

The codebase is organized into logical layers including models, analytics, and services to
maintain a clean and maintainable structure.
"""

from rolling_insights.services.insights.all_insights_service import AllInsightsService
from rolling_insights.services.llm_service import LLMService
from rolling_insights.services.json_file_service import JsonFileService

__all__ = (
    "AllInsightsService",
    "JsonFileService",
    "LLMService",
)
