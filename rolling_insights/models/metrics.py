"""
Represent the metrics gotten from processing the raw data.
"""

from datetime import date
from typing import Dict, Any, Optional
from pydantic import BaseModel


class SleepMetrics(BaseModel):
    """Core entity representing sleep metrics for a single night."""

    date: date
    total_sleep_minutes: float
    deep_sleep_minutes: float
    rem_sleep_minutes: float
    light_sleep_minutes: float
    sleep_efficiency: float
    sleep_latency_minutes: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "date": self.date.isoformat(),
            "total_sleep_minutes": self.total_sleep_minutes,
            "deep_sleep_minutes": self.deep_sleep_minutes,
            "rem_sleep_minutes": self.rem_sleep_minutes,
            "light_sleep_minutes": self.light_sleep_minutes,
            "sleep_efficiency": self.sleep_efficiency,
            "sleep_latency_minutes": self.sleep_latency_minutes,
        }


class PhoneMetrics(BaseModel):
    """Core entity representing phone usage metrics for a single day."""

    date: date
    screen_time_minutes: int
    pickups: int
    screen_time_before_bed_minutes: int
    first_pickup_after_wakeup_minutes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "date": self.date.isoformat(),
            "screen_time_minutes": self.screen_time_minutes,
            "pickups": self.pickups,
            "screen_time_before_bed_minutes": self.screen_time_before_bed_minutes,
            "first_pickup_after_wakeup_minutes": self.first_pickup_after_wakeup_minutes,
        }


class HealthMetrics(BaseModel):
    """Core entity representing health metrics for a single day."""

    date: date
    avg_heart_rate: Optional[float] = None
    total_steps: int
    active_energy_burned: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "date": self.date.isoformat(),
            "avg_heart_rate": self.avg_heart_rate,
            "total_steps": self.total_steps,
            "active_energy_burned": self.active_energy_burned,
        }
