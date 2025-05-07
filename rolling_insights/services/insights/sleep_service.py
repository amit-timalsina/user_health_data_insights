"""Sleep insight service for generating insights from sleep data."""

from datetime import date
from typing import List

from rolling_insights.services.insights.base import InsightService
from rolling_insights.models import SleepMetrics, SleepInsightPayload, RawDataPoint
from rolling_insights.analytics.statistics import calculate_sleep_stats
from rolling_insights.exceptions.data import DataValidationError


class SleepInsightService(InsightService):
    """Service for generating sleep insights."""

    @staticmethod
    def extract_metrics(raw_data: List[RawDataPoint]) -> List[SleepMetrics]:
        """Extract sleep metrics from raw data."""
        metrics = []

        for day_data in raw_data:
            try:
                health = day_data.health_data
                deep = sum(health.SLEEP_DEEP)
                rem = sum(health.SLEEP_REM)
                light = sum(health.SLEEP_LIGHT)
                awake = sum(health.SLEEP_AWAKE)
                total = deep + rem + light

                metrics.append(
                    SleepMetrics(
                        date=date.fromisoformat(
                            day_data.start_date.replace("Z", "+00:00").split("T")[0]
                        ),
                        total_sleep_minutes=total,
                        deep_sleep_minutes=deep,
                        rem_sleep_minutes=rem,
                        light_sleep_minutes=light,
                        sleep_efficiency=total / (total + awake)
                        if total + awake
                        else 0,
                        sleep_latency_minutes=0,  # Simplified: latency unavailable in sample
                    )
                )
            except Exception as e:
                raise DataValidationError(f"Invalid data format: {str(e)}")

        return metrics

    async def execute(self, raw_data: List[RawDataPoint]) -> SleepInsightPayload:
        """Execute the service to generate sleep insights."""
        # Extract metrics from raw data
        metrics = self.extract_metrics(raw_data)

        # Calculate statistics
        stats = calculate_sleep_stats(metrics)

        # Generate narrative insights
        narrative = await self.llm_service.generate_sleep_narrative(stats)

        # Create insight payload
        return SleepInsightPayload(
            start_date=metrics[0].date,
            end_date=metrics[-1].date,
            stats=stats,
            daily_data=[m.to_dict() for m in metrics],
            narrative_insights=narrative,
        )
