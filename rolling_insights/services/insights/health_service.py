"""Health insight service for generating insights from health data."""

from datetime import date
from typing import List, Optional

from rolling_insights.services.insights.base import InsightService
from rolling_insights.models import (
    HealthMetrics,
    SleepMetrics,
    HealthInsightPayload,
    RawDataPoint,
)
from rolling_insights.analytics.statistics import calculate_health_stats
from rolling_insights.exceptions.data import DataValidationError


class HealthInsightService(InsightService):
    """Service for generating health insights."""

    @staticmethod
    def extract_metrics(raw_data: List[RawDataPoint]) -> List[HealthMetrics]:
        """Extract health metrics from raw data."""
        metrics = []
        for day_data in raw_data:
            try:
                date_str = day_data.start_date.replace("Z", "+00:00").split("T")[0]

                metrics.append(
                    HealthMetrics(
                        date=date.fromisoformat(date_str),
                        avg_heart_rate=None,  # Simplified: not in sample
                        total_steps=sum(day_data.health_data.STEPS),
                        active_energy_burned=sum(
                            day_data.health_data.ACTIVE_ENERGY_BURNED
                        ),
                    )
                )
            except Exception as e:
                raise DataValidationError(f"Invalid data format: {str(e)}")

        return metrics

    async def execute(
        self,
        raw_data: List[RawDataPoint],
        sleep_metrics: Optional[List[SleepMetrics]] = None,
    ) -> HealthInsightPayload:
        """Execute the service to generate health insights."""
        from rolling_insights.services.insights.sleep_service import (
            SleepInsightService,
        )

        if sleep_metrics is None:
            sleep_metrics = SleepInsightService.extract_metrics(raw_data)

        # Extract metrics from raw data
        metrics = self.extract_metrics(raw_data)

        # Calculate statistics
        stats = calculate_health_stats(metrics, sleep_metrics)

        # Generate narrative insights
        narrative = await self.llm_service.generate_health_narrative(stats)

        # Create insight payload
        return HealthInsightPayload(
            start_date=metrics[0].date,
            end_date=metrics[-1].date,
            stats=stats,
            daily_data=[m.to_dict() for m in metrics],
            narrative_insights=narrative,
        )
