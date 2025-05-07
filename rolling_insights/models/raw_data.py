"""
Represent the raw input data structures received from data sources.
They handle validation and normalization of incoming data.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class RawHealthData(BaseModel):
    """Raw health data from the input file."""

    SLEEP_DEEP: List[float] = Field(default_factory=list)
    SLEEP_REM: List[float] = Field(default_factory=list)
    SLEEP_LIGHT: List[float] = Field(default_factory=list)
    SLEEP_AWAKE: List[float] = Field(default_factory=list)
    STEPS: List[int] = Field(default_factory=list)
    ACTIVE_ENERGY_BURNED: List[float] = Field(default_factory=list)

    # Add more flexibility to accept any additional fields
    model_config = {
        "extra": "allow",
    }


class RawDataPoint(BaseModel):
    """Raw data point from the input file.

    The model is flexible to support various field names from the data source.
    """

    # Use aliases to support both id and _id formats
    id: Optional[str] = Field(None, alias="_id")
    user_id: Optional[str] = Field(None)  # Make user_id optional

    # Required fields that must be present
    start_date: str
    end_date: str
    health_data: RawHealthData

    # Allow extra fields that might be present in the data
    model_config = {
        "extra": "allow",
        "populate_by_name": True,  # Allow populating by attribute name or alias
    }

    @model_validator(mode="before")
    @classmethod
    def check_id_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle different ID field formats in the data."""
        if isinstance(data, dict):
            # If _id exists but id doesn't, copy it
            if "_id" in data and "id" not in data:
                data["id"] = data["_id"]

            # If we have a user_id field in a different format, extract it
            if "user_id" not in data:
                _id = data.get("_id")
                if _id is not None and isinstance(_id, str):
                    data["user_id"] = _id.split("_")[0] if "_" in _id else _id

        return data
