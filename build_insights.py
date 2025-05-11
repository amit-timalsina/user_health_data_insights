#!/usr/bin/env python3
"""
Rolling Insights - Main script for generating insights from health data.

This script reads raw health data from the samples directory and generates
three insight files:
- sleepInsights.json
- phoneUsage.json
- healthInsights.json

Usage:
    python build_insights.py [--input FILEPATH] [--output-dir DIR]
"""

import argparse
import asyncio
import logging
from pathlib import Path

from rolling_insights.services.llm_service import LLMService
from rolling_insights.services.json_file_service import JsonFileService
from rolling_insights.services.insights import AllInsightsService
from rolling_insights.exceptions.base import InsightError


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate 7-day rolling insights.")
    parser.add_argument(
        "--input",
        default="samples/user_0e7f18ae_2025-03-24_to_2025-03-30.json",
        help="Path to the raw JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="insights",
        help="Directory where the three insight JSON files will be written.",
    )
    return parser.parse_args()


async def main_async() -> None:
    """Asynchronous main function to generate insights."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir

    logger.info(f"Processing {input_path}")
    logger.info(f"Outputting results to {output_dir}")

    try:
        # Create dependencies
        llm_service = LLMService()
        storage_service = JsonFileService()

        # Create and execute the insight service
        service = AllInsightsService(
            llm_service=llm_service,
            storage_service=storage_service,
            output_dir=output_dir,
        )

        await service.execute(input_path)
        logger.info(
            f"✅ Generation complete. Files written to {Path(output_dir).resolve()}"
        )
    except InsightError as e:
        logger.error(f"❌ Error generating insights: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise


def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
