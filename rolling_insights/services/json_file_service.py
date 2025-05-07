"""Service for file operations.

This module handles interactions with the file system for loading
raw data and saving processed insights.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union


class JsonFileService:
    """Service for loading and saving data from/to files."""

    @staticmethod
    def load_data(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load JSON data from the specified file path."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r") as file:
            return json.load(file)

    @staticmethod
    def save_insights(insights: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save insight data to the specified JSON file path."""
        filepath = Path(filepath)

        # Ensure the directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as file:
            json.dump(insights, file, indent=2)

    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """Ensure the specified directory exists and return its Path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
