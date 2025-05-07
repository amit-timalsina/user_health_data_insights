from rolling_insights.models.raw_data import RawDataPoint, RawHealthData


class TestRawDataPoint:
    """Test RawDataPoint model."""

    def test_init_with_standard_format(self):
        """Test initialization with standard format."""
        raw_data = RawDataPoint(
            id="123",
            user_id="user123",
            start_date="2023-01-01T00:00:00Z",
            end_date="2023-01-02T00:00:00Z",
            health_data=RawHealthData(
                SLEEP_DEEP=[30, 60, 90],
                SLEEP_REM=[20, 40, 60],
                SLEEP_LIGHT=[50, 100, 150],
                SLEEP_AWAKE=[5, 10, 15],
                STEPS=[1000, 2000, 3000],
                ACTIVE_ENERGY_BURNED=[100, 200, 300],
            ),
        )
        assert raw_data.id == "123"
        assert raw_data.user_id == "user123"
        assert raw_data.start_date == "2023-01-01T00:00:00Z"
        assert raw_data.end_date == "2023-01-02T00:00:00Z"
        assert sum(raw_data.health_data.SLEEP_DEEP) == 180
        assert sum(raw_data.health_data.SLEEP_REM) == 120
        assert sum(raw_data.health_data.SLEEP_LIGHT) == 300
        assert sum(raw_data.health_data.SLEEP_AWAKE) == 30
        assert sum(raw_data.health_data.STEPS) == 6000
        assert sum(raw_data.health_data.ACTIVE_ENERGY_BURNED) == 600

    def test_init_with_mongo_format(self):
        """Test initialization with MongoDB-style data format (_id field)."""
        raw_data = RawDataPoint(
            _id="user123_night_67ea53b99fecbc62a5adcea4f7c7db317a246",
            start_date="2023-01-01T00:00:00Z",
            end_date="2023-01-02T00:00:00Z",
            health_data=RawHealthData(
                SLEEP_DEEP=[30, 60, 90],
                SLEEP_REM=[20, 40, 60],
                SLEEP_LIGHT=[50, 100, 150],
                SLEEP_AWAKE=[5, 10, 15],
                STEPS=[1000, 2000, 3000],
                ACTIVE_ENERGY_BURNED=[100, 200, 300],
            ),
        )
        # Verify id and user_id were properly extracted
        assert raw_data.id == "user123_night_67ea53b99fecbc62a5adcea4f7c7db317a246"
        assert raw_data.user_id == "user123"
        assert raw_data.start_date == "2023-01-01T00:00:00Z"
        assert raw_data.end_date == "2023-01-02T00:00:00Z"
