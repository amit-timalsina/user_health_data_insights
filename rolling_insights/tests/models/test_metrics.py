"""Tests for the metrics models."""

from datetime import date

from rolling_insights.models import (
    SleepMetrics,
    PhoneMetrics,
    HealthMetrics,
)


class TestSleepMetrics:
    """Test SleepMetrics model."""

    def test_init(self):
        """Test initialization with valid data."""
        metrics = SleepMetrics(
            date=date(2023, 1, 1),
            total_sleep_minutes=450,
            deep_sleep_minutes=120,
            rem_sleep_minutes=90,
            light_sleep_minutes=240,
            sleep_efficiency=0.9,
            sleep_latency_minutes=15,
        )
        assert metrics.date == date(2023, 1, 1)
        assert metrics.total_sleep_minutes == 450
        assert metrics.deep_sleep_minutes == 120
        assert metrics.rem_sleep_minutes == 90
        assert metrics.light_sleep_minutes == 240
        assert metrics.sleep_efficiency == 0.9
        assert metrics.sleep_latency_minutes == 15

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = SleepMetrics(
            date=date(2023, 1, 1),
            total_sleep_minutes=450,
            deep_sleep_minutes=120,
            rem_sleep_minutes=90,
            light_sleep_minutes=240,
            sleep_efficiency=0.9,
            sleep_latency_minutes=15,
        )
        result = metrics.to_dict()
        assert result["date"] == "2023-01-01"
        assert result["total_sleep_minutes"] == 450
        assert result["deep_sleep_minutes"] == 120
        assert result["rem_sleep_minutes"] == 90
        assert result["light_sleep_minutes"] == 240
        assert result["sleep_efficiency"] == 0.9
        assert result["sleep_latency_minutes"] == 15


class TestPhoneMetrics:
    """Test PhoneMetrics model."""

    def test_init(self):
        """Test initialization with valid data."""
        metrics = PhoneMetrics(
            date=date(2023, 1, 1),
            screen_time_minutes=180,
            pickups=50,
            screen_time_before_bed_minutes=30,
            first_pickup_after_wakeup_minutes=5,
        )
        assert metrics.date == date(2023, 1, 1)
        assert metrics.screen_time_minutes == 180
        assert metrics.pickups == 50
        assert metrics.screen_time_before_bed_minutes == 30
        assert metrics.first_pickup_after_wakeup_minutes == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PhoneMetrics(
            date=date(2023, 1, 1),
            screen_time_minutes=180,
            pickups=50,
            screen_time_before_bed_minutes=30,
            first_pickup_after_wakeup_minutes=5,
        )
        result = metrics.to_dict()
        assert result["date"] == "2023-01-01"
        assert result["screen_time_minutes"] == 180
        assert result["pickups"] == 50
        assert result["screen_time_before_bed_minutes"] == 30
        assert result["first_pickup_after_wakeup_minutes"] == 5


class TestHealthMetrics:
    """Test HealthMetrics model."""

    def test_init(self):
        """Test initialization with valid data."""
        metrics = HealthMetrics(
            date=date(2023, 1, 1),
            total_steps=10000,
            active_energy_burned=400,
        )
        assert metrics.date == date(2023, 1, 1)
        assert metrics.avg_heart_rate is None
        assert metrics.total_steps == 10000
        assert metrics.active_energy_burned == 400

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = HealthMetrics(
            date=date(2023, 1, 1),
            total_steps=10000,
            active_energy_burned=400,
        )
        result = metrics.to_dict()
        assert result["date"] == "2023-01-01"
        assert result["avg_heart_rate"] is None
        assert result["total_steps"] == 10000
        assert result["active_energy_burned"] == 400
