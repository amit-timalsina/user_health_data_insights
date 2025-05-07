"""Tests for the insights models."""

from datetime import date
from rolling_insights.models.insights import (
    SleepStats,
    PhoneStats,
    HealthStats,
    InsightPeriod,
    BaseInsightPayload,
    SleepInsightPayload,
    PhoneInsightPayload,
    HealthInsightPayload,
)


class TestSleepStats:
    def test_sleep_stats_init(self):
        """Test initialization of SleepStats with valid data."""
        stats = SleepStats(
            avg_total_sleep_minutes=420.0,
            avg_deep_sleep_minutes=120.0,
            avg_rem_sleep_minutes=90.0,
            avg_light_sleep_minutes=210.0,
            avg_sleep_efficiency=0.9,
            min_total_sleep_minutes=360.0,
            max_total_sleep_minutes=480.0,
            deep_sleep_percentage=0.28,
            rem_sleep_percentage=0.21,
            light_sleep_percentage=0.51,
            sleep_duration_variability=30.0,
            deep_sleep_percentile=0.75,
            rem_sleep_percentile=0.65,
            sleep_efficiency_percentile=0.8,
            sleep_quality_score=85.0,
        )

        assert stats.avg_total_sleep_minutes == 420.0
        assert stats.avg_deep_sleep_minutes == 120.0
        assert stats.sleep_quality_score == 85.0

    def test_sleep_stats_optional_fields(self):
        """Test initialization with only required fields."""
        stats = SleepStats(
            avg_total_sleep_minutes=420.0,
            avg_deep_sleep_minutes=120.0,
            avg_rem_sleep_minutes=90.0,
            avg_light_sleep_minutes=210.0,
            avg_sleep_efficiency=0.9,
            min_total_sleep_minutes=360.0,
            max_total_sleep_minutes=480.0,
            deep_sleep_percentage=0.28,
            rem_sleep_percentage=0.21,
            light_sleep_percentage=0.51,
        )

        assert stats.sleep_duration_variability == 0.0
        assert stats.deep_sleep_percentile == 0.0
        assert stats.sleep_quality_score == 0.0


class TestPhoneStats:
    def test_phone_stats_init(self):
        """Test initialization of PhoneStats with valid data."""
        stats = PhoneStats(
            avg_screen_time_minutes=180.0,
            avg_pickups=50.0,
            avg_screen_before_bed_minutes=30.0,
            total_screen_time_minutes=1260,
            total_pickups=350,
            screen_before_bed_total_sleep_correlation=-0.6,
            screen_before_bed_deep_sleep_correlation=-0.4,
            screen_before_bed_total_sleep_variance_explained=0.36,
            screen_time_percentile=0.7,
            screen_before_bed_rem_sleep_correlation=-0.3,
            pickups_sleep_correlation=-0.5,
            morning_pickup_sleep_quality_correlation=0.2,
        )

        assert stats.avg_screen_time_minutes == 180.0
        assert stats.total_pickups == 350
        assert stats.screen_before_bed_total_sleep_correlation == -0.6

    def test_phone_stats_optional_fields(self):
        """Test initialization with only required fields."""
        stats = PhoneStats(
            avg_screen_time_minutes=180.0,
            avg_pickups=50.0,
            avg_screen_before_bed_minutes=30.0,
            total_screen_time_minutes=1260,
            total_pickups=350,
            screen_before_bed_total_sleep_correlation=-0.6,
            screen_before_bed_deep_sleep_correlation=-0.4,
        )

        assert stats.screen_before_bed_total_sleep_variance_explained == 0.0
        assert stats.screen_time_percentile == 0.0
        assert stats.pickups_sleep_correlation == 0.0


class TestHealthStats:
    def test_health_stats_init(self):
        """Test initialization of HealthStats with valid data."""
        stats = HealthStats(
            avg_steps=8000.0,
            avg_active_energy_burned=300.0,
            total_steps=56000,
            total_active_energy_burned=2100.0,
            steps_total_sleep_correlation=0.4,
            activity_deep_sleep_correlation=0.5,
            activity_deep_sleep_variance_explained=0.25,
            steps_sleep_efficiency_correlation=0.3,
            activity_rem_sleep_correlation=0.2,
            day_to_day_activity_variability=0.15,
            activity_intensity_score=0.7,
        )

        assert stats.avg_steps == 8000.0
        assert stats.total_active_energy_burned == 2100.0
        assert stats.activity_deep_sleep_correlation == 0.5

    def test_health_stats_optional_fields(self):
        """Test initialization with only required fields."""
        stats = HealthStats(
            avg_steps=8000.0,
            avg_active_energy_burned=300.0,
            total_steps=56000,
            total_active_energy_burned=2100.0,
            steps_total_sleep_correlation=0.4,
            activity_deep_sleep_correlation=0.5,
        )

        assert stats.activity_deep_sleep_variance_explained == 0.0
        assert stats.steps_sleep_efficiency_correlation == 0.0
        assert stats.activity_intensity_score == 0.0


class TestInsightPeriod:
    def test_insight_period_init(self):
        """Test initialization of InsightPeriod."""
        period = InsightPeriod(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
        )

        assert period.start_date == date(2023, 1, 1)
        assert period.end_date == date(2023, 1, 7)


class TestBaseInsightPayload:
    def test_base_insight_payload_init(self):
        """Test initialization of BaseInsightPayload."""
        payload = BaseInsightPayload(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
            narrative_insights="Test narrative insights",
        )

        assert payload.start_date == date(2023, 1, 1)
        assert payload.end_date == date(2023, 1, 7)
        assert payload.narrative_insights == "Test narrative insights"


class TestSleepInsightPayload:
    def test_sleep_insight_payload_init(self):
        """Test initialization of SleepInsightPayload."""
        stats = SleepStats(
            avg_total_sleep_minutes=420.0,
            avg_deep_sleep_minutes=120.0,
            avg_rem_sleep_minutes=90.0,
            avg_light_sleep_minutes=210.0,
            avg_sleep_efficiency=0.9,
            min_total_sleep_minutes=360.0,
            max_total_sleep_minutes=480.0,
            deep_sleep_percentage=0.28,
            rem_sleep_percentage=0.21,
            light_sleep_percentage=0.51,
        )

        daily_data = [
            {"date": "2023-01-01", "total_sleep_minutes": 420},
            {"date": "2023-01-02", "total_sleep_minutes": 380},
        ]

        payload = SleepInsightPayload(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
            narrative_insights="Sleep insights narrative",
            stats=stats,
            daily_data=daily_data,
        )

        assert payload.start_date == date(2023, 1, 1)
        assert payload.end_date == date(2023, 1, 7)
        assert payload.narrative_insights == "Sleep insights narrative"
        assert payload.stats == stats
        assert payload.daily_data == daily_data

    def test_sleep_insight_payload_to_dict(self):
        """Test to_dict method of SleepInsightPayload."""
        stats = SleepStats(
            avg_total_sleep_minutes=420.0,
            avg_deep_sleep_minutes=120.0,
            avg_rem_sleep_minutes=90.0,
            avg_light_sleep_minutes=210.0,
            avg_sleep_efficiency=0.9,
            min_total_sleep_minutes=360.0,
            max_total_sleep_minutes=480.0,
            deep_sleep_percentage=0.28,
            rem_sleep_percentage=0.21,
            light_sleep_percentage=0.51,
        )

        daily_data = [
            {"date": "2023-01-01", "total_sleep_minutes": 420},
            {"date": "2023-01-02", "total_sleep_minutes": 380},
        ]

        payload = SleepInsightPayload(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
            narrative_insights="Sleep insights narrative",
            stats=stats,
            daily_data=daily_data,
        )

        dict_data = payload.to_dict()

        assert dict_data["period"]["start_date"] == "2023-01-01"
        assert dict_data["period"]["end_date"] == "2023-01-07"
        assert dict_data["narrative_insights"] == "Sleep insights narrative"
        assert dict_data["stats"] == stats.model_dump()
        assert dict_data["daily_data"] == daily_data


class TestPhoneInsightPayload:
    def test_phone_insight_payload_init(self):
        """Test initialization of PhoneInsightPayload."""
        stats = PhoneStats(
            avg_screen_time_minutes=180.0,
            avg_pickups=50.0,
            avg_screen_before_bed_minutes=30.0,
            total_screen_time_minutes=1260,
            total_pickups=350,
            screen_before_bed_total_sleep_correlation=-0.6,
            screen_before_bed_deep_sleep_correlation=-0.4,
        )

        daily_data = [
            {"date": "2023-01-01", "screen_time_minutes": 200, "pickups": 55},
            {"date": "2023-01-02", "screen_time_minutes": 160, "pickups": 45},
        ]

        payload = PhoneInsightPayload(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
            narrative_insights="Phone usage insights",
            stats=stats,
            daily_data=daily_data,
        )

        assert payload.start_date == date(2023, 1, 1)
        assert payload.end_date == date(2023, 1, 7)
        assert payload.narrative_insights == "Phone usage insights"
        assert payload.stats == stats
        assert payload.daily_data == daily_data

    def test_phone_insight_payload_to_dict(self):
        """Test to_dict method of PhoneInsightPayload."""
        stats = PhoneStats(
            avg_screen_time_minutes=180.0,
            avg_pickups=50.0,
            avg_screen_before_bed_minutes=30.0,
            total_screen_time_minutes=1260,
            total_pickups=350,
            screen_before_bed_total_sleep_correlation=-0.6,
            screen_before_bed_deep_sleep_correlation=-0.4,
        )

        daily_data = [
            {"date": "2023-01-01", "screen_time_minutes": 200, "pickups": 55},
            {"date": "2023-01-02", "screen_time_minutes": 160, "pickups": 45},
        ]

        payload = PhoneInsightPayload(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
            narrative_insights="Phone usage insights",
            stats=stats,
            daily_data=daily_data,
        )

        dict_data = payload.to_dict()

        assert dict_data["period"]["start_date"] == "2023-01-01"
        assert dict_data["period"]["end_date"] == "2023-01-07"
        assert dict_data["narrative_insights"] == "Phone usage insights"
        assert dict_data["stats"] == stats.model_dump()
        assert dict_data["daily_data"] == daily_data


class TestHealthInsightPayload:
    def test_health_insight_payload_init(self):
        """Test initialization of HealthInsightPayload."""
        stats = HealthStats(
            avg_steps=8000.0,
            avg_active_energy_burned=300.0,
            total_steps=56000,
            total_active_energy_burned=2100.0,
            steps_total_sleep_correlation=0.4,
            activity_deep_sleep_correlation=0.5,
        )

        daily_data = [
            {"date": "2023-01-01", "steps": 8500, "active_energy_burned": 320},
            {"date": "2023-01-02", "steps": 7500, "active_energy_burned": 280},
        ]

        payload = HealthInsightPayload(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
            narrative_insights="Health insights narrative",
            stats=stats,
            daily_data=daily_data,
        )

        assert payload.start_date == date(2023, 1, 1)
        assert payload.end_date == date(2023, 1, 7)
        assert payload.narrative_insights == "Health insights narrative"
        assert payload.stats == stats
        assert payload.daily_data == daily_data

    def test_health_insight_payload_to_dict(self):
        """Test to_dict method of HealthInsightPayload."""
        stats = HealthStats(
            avg_steps=8000.0,
            avg_active_energy_burned=300.0,
            total_steps=56000,
            total_active_energy_burned=2100.0,
            steps_total_sleep_correlation=0.4,
            activity_deep_sleep_correlation=0.5,
        )

        daily_data = [
            {"date": "2023-01-01", "steps": 8500, "active_energy_burned": 320},
            {"date": "2023-01-02", "steps": 7500, "active_energy_burned": 280},
        ]

        payload = HealthInsightPayload(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 7),
            narrative_insights="Health insights narrative",
            stats=stats,
            daily_data=daily_data,
        )

        dict_data = payload.to_dict()

        assert dict_data["period"]["start_date"] == "2023-01-01"
        assert dict_data["period"]["end_date"] == "2023-01-07"
        assert dict_data["narrative_insights"] == "Health insights narrative"
        assert dict_data["stats"] == stats.model_dump()
        assert dict_data["daily_data"] == daily_data
