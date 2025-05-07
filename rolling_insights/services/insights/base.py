"""Abstract base classes for insight services."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from rolling_insights.models.raw_data import RawDataPoint
from rolling_insights.services.llm_service import LLMService
from rolling_insights.services.json_file_service import JsonFileService


class InsightService(ABC):
    """Abstract base class for insight generation services."""

    def __init__(self, llm_service: LLMService, storage_service: JsonFileService):
        self.llm_service = llm_service
        self.storage_service = storage_service

    @abstractmethod
    async def execute(self, raw_data: List[RawDataPoint]) -> Dict[str, Any]:
        """Execute the service to generate insights."""
