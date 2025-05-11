"""Phone insight service for generating insights from phone usage data."""

from datetime import date
from typing import List, Optional

from rolling_insights.services.insights.base import InsightService
from rolling_insights.models import (
    PhoneMetrics,
    SleepMetrics,
    PhoneInsightPayload,
    RawDataPoint,
)
from rolling_insights.analytics.statistics import calculate_phone_stats
from rolling_insights.exceptions.data import DataValidationError


class PhoneInsightService(InsightService):
    """Service for generating phone usage insights."""

    @staticmethod
    def extract_metrics(
        raw_data: List[RawDataPoint], sleep_metrics: List[SleepMetrics]
    ) -> List[PhoneMetrics]:
        """Extract phone metrics from raw data."""
        import random

        random.seed(42)  # For reproducibility in demo mode

        metrics = []
        for i, day_data in enumerate(raw_data):
            try:
                date_str = day_data.start_date.replace("Z", "+00:00").split("T")[0]

                # In a real implementation, extract actual phone data
                # For this demo, generate synthetic data
                metrics.append(
                    PhoneMetrics(
                        date=date.fromisoformat(date_str),
                        screen_time_minutes=random.randint(110, 180),
                        pickups=random.randint(40, 80),
                        screen_time_before_bed_minutes=random.randint(25, 55),
                        first_pickup_after_wakeup_minutes=random.randint(3, 15),
                    )
                )
            except Exception as e:
                raise DataValidationError(f"Invalid data format: {str(e)}")

        return metrics

    async def execute(
        self,
        raw_data: List[RawDataPoint],
        sleep_metrics: Optional[List[SleepMetrics]] = None,
    ) -> PhoneInsightPayload:
        """Execute the service to generate phone usage insights."""
        from rolling_insights.services.insights.sleep_service import (
            SleepInsightService,
        )

        if sleep_metrics is None:
            sleep_metrics = SleepInsightService.extract_metrics(raw_data)

        # Extract metrics from raw data
        metrics = self.extract_metrics(raw_data, sleep_metrics)

        # Calculate statistics
        stats = calculate_phone_stats(metrics, sleep_metrics)

        # Generate narrative insights
        narrative = await self.llm_service.generate_phone_narrative(stats)

        # Create insight payload
        return PhoneInsightPayload(
            start_date=metrics[0].date,
            end_date=metrics[-1].date,
            stats=stats,
            daily_data=[m.to_dict() for m in metrics],
            narrative_insights=narrative,
        )
