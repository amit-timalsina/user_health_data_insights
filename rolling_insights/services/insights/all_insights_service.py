"""Service for generating all types of insights."""

from typing import Dict, List

from rolling_insights.models.insights import (
    HealthInsightPayload,
    InsightPayload,
    PhoneInsightPayload,
    SleepInsightPayload,
)
from rolling_insights.models.metrics import HealthMetrics, PhoneMetrics, SleepMetrics
from rolling_insights.models.raw_data import RawDataPoint
from rolling_insights.services.llm_service import LLMService
from rolling_insights.services.json_file_service import JsonFileService
from rolling_insights.services.insights.sleep_service import SleepInsightService
from rolling_insights.services.insights.phone_service import PhoneInsightService
from rolling_insights.services.insights.health_service import HealthInsightService
from rolling_insights.analytics.statistics import (
    calculate_correlation,
    calculate_variance_explained,
)


class AllInsightsService:
    """Service to generate all three insight types."""

    def __init__(
        self, llm_service: LLMService, storage_service: JsonFileService, output_dir: str
    ):
        self.llm_service = llm_service
        self.storage_service = storage_service
        self.output_dir = output_dir

        # Initialize the three specific services
        self.sleep_service = SleepInsightService(llm_service, storage_service)
        self.phone_service = PhoneInsightService(llm_service, storage_service)
        self.health_service = HealthInsightService(llm_service, storage_service)

    async def execute(self, input_path: str) -> None:
        """Execute the service to generate all insight types."""
        # Ensure output directory exists
        output_dir = self.storage_service.ensure_directory(self.output_dir)

        # Load raw data
        raw_data = self.storage_service.load_data(input_path)
        raw_data_points = [RawDataPoint(**data) for data in raw_data]

        # Extract sleep metrics (needed for correlation analysis in other metrics)
        sleep_metrics = SleepInsightService.extract_metrics(raw_data_points)

        # Extract phone and health metrics for cross-correlations
        phone_metrics = PhoneInsightService.extract_metrics(
            raw_data_points, sleep_metrics
        )
        health_metrics = HealthInsightService.extract_metrics(raw_data_points)

        # Generate all three insight types
        sleep_insights = await self.sleep_service.execute(raw_data_points)
        phone_insights = await self.phone_service.execute(
            raw_data_points, sleep_metrics
        )
        health_insights = await self.health_service.execute(
            raw_data_points, sleep_metrics
        )

        # Enhance insights with cross-metric correlations
        enhanced_insights = self._add_cross_metric_correlations(
            sleep_insights,
            phone_insights,
            health_insights,
            sleep_metrics,
            phone_metrics,
            health_metrics,
        )

        # Save results
        self.storage_service.save_insights(
            enhanced_insights["sleep"], output_dir / "sleepInsights.json"
        )
        self.storage_service.save_insights(
            enhanced_insights["phone"], output_dir / "phoneUsage.json"
        )
        self.storage_service.save_insights(
            enhanced_insights["health"], output_dir / "healthInsights.json"
        )

    def _add_cross_metric_correlations(
        self,
        sleep_insights: SleepInsightPayload,
        phone_insights: PhoneInsightPayload,
        health_insights: HealthInsightPayload,
        sleep_metrics: List[SleepMetrics],
        phone_metrics: List[PhoneMetrics],
        health_metrics: List[HealthMetrics],
    ) -> Dict[str, InsightPayload]:
        """Add cross-metric correlations to enhance insights.

        This method enriches the insights by analyzing relationships between different
        metric types (sleep vs phone vs health) to provide a more comprehensive view.
        """
        # Extract key metrics for correlation analysis
        sleep_deep = [m.deep_sleep_minutes for m in sleep_metrics]
        sleep_rem = [m.rem_sleep_minutes for m in sleep_metrics]
        sleep_efficiency = [m.sleep_efficiency for m in sleep_metrics]

        phone_screen = [float(m.screen_time_minutes) for m in phone_metrics]
        phone_pickups = [float(m.pickups) for m in phone_metrics]
        phone_before_bed = [
            float(m.screen_time_before_bed_minutes) for m in phone_metrics
        ]

        steps = [float(m.total_steps) for m in health_metrics]
        active_energy = [float(m.active_energy_burned) for m in health_metrics]

        # Calculate cross-metric correlations
        cross_correlations = {
            # Phone impact on specific sleep phases
            "phone_screen_deep_sleep_corr": calculate_correlation(
                phone_screen, sleep_deep
            )
            or 0,
            "phone_screen_rem_sleep_corr": calculate_correlation(
                phone_screen, sleep_rem
            )
            or 0,
            "pickups_deep_sleep_corr": calculate_correlation(phone_pickups, sleep_deep)
            or 0,
            # Health impact on specific sleep phases
            "steps_deep_sleep_corr": calculate_correlation(steps, sleep_deep) or 0,
            "steps_rem_sleep_corr": calculate_correlation(steps, sleep_rem) or 0,
            "activity_sleep_efficiency_corr": calculate_correlation(
                active_energy, sleep_efficiency
            )
            or 0,
            # Phone-health correlations
            "phone_activity_corr": calculate_correlation(phone_screen, active_energy)
            or 0,
        }

        # Calculate variance explained
        variance_explained = {
            "phone_screen_deep_sleep_var": calculate_variance_explained(
                phone_screen, sleep_deep
            ),
            "screen_before_bed_deep_sleep_var": calculate_variance_explained(
                phone_before_bed, sleep_deep
            ),
            "activity_deep_sleep_var": calculate_variance_explained(
                active_energy, sleep_deep
            ),
        }

        # Add cross-correlations to the insights
        sleep_insights_enhanced = sleep_insights
        phone_insights_enhanced = phone_insights
        health_insights_enhanced = health_insights
        # Add cross-correlation data to the stats section
        phone_insights_enhanced.stats.screen_before_bed_deep_sleep_correlation = (
            cross_correlations["phone_screen_deep_sleep_corr"]
        )
        phone_insights_enhanced.stats.screen_before_bed_rem_sleep_correlation = (
            cross_correlations["phone_screen_rem_sleep_corr"]
        )
        phone_insights_enhanced.stats.screen_before_bed_total_sleep_variance_explained = variance_explained[
            "phone_screen_deep_sleep_var"
        ]

        health_insights_enhanced.stats.activity_deep_sleep_correlation = (
            cross_correlations["steps_deep_sleep_corr"]
        )
        health_insights_enhanced.stats.activity_rem_sleep_correlation = (
            cross_correlations["steps_rem_sleep_corr"]
        )
        health_insights_enhanced.stats.steps_sleep_efficiency_correlation = (
            cross_correlations["activity_sleep_efficiency_corr"]
        )

        # Add a summary of cross-metric findings
        cross_metric_summary = {
            "sleep_phone_health_correlations": cross_correlations,
            "variance_explained": variance_explained,
        }

        # We could add this to all three insight types, but for now just add to the sleep insights
        # as it's the primary focus of the analysis
        sleep_insights_enhanced.cross_metric_analysis = cross_metric_summary

        return {
            "sleep": sleep_insights_enhanced,
            "phone": phone_insights_enhanced,
            "health": health_insights_enhanced,
        }
