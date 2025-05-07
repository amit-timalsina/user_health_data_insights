"""LLM service for generating narrative insights.

This module handles interactions with external AI services like OpenAI
for generating natural language narratives from statistical data.
"""

import os
import json
import logging
from typing import Optional, cast, Any
from openai import AsyncOpenAI

from rolling_insights.models import SleepStats, PhoneStats, HealthStats

logger = logging.getLogger(__name__)


class LLMService:
    """Service to handle LLM interactions for narrative generation."""

    FALLBACK_NARRATIVE = "AI service unavailable – no narrative generated."
    MAX_RETRIES = 3
    DEFAULT_MODEL = "gpt-4o-mini"

    # Define client at class level with proper typing
    client: Any  # Using Any to avoid complex conditional typing

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the LLM client if possible."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)

        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None

    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        return self.client is not None

    async def generate_narrative(self, prompt: str) -> str:
        """Generate narrative insights from a prompt using the LLM."""
        if not self.is_available():
            logger.warning("LLM service unavailable, using fallback")
            return self.FALLBACK_NARRATIVE

        if self.client is None:
            return self.FALLBACK_NARRATIVE

        for attempt in range(self.MAX_RETRIES):
            try:
                model_name = cast(str, self.model)  # Cast to satisfy type checker
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7,
                )
                content = response.choices[0].message.content
                if content is None:
                    return self.FALLBACK_NARRATIVE
                return content.strip()
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "Error generating insights (%s/%s): %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    exc,
                )
                if attempt == self.MAX_RETRIES - 1:
                    return f"Error generating insights: {exc}"
        return self.FALLBACK_NARRATIVE

    # Domain-specific prompts for different insight types

    async def generate_sleep_narrative(self, stats: SleepStats) -> str:
        """Generate narrative insights for sleep data."""
        prompt = (
            "Generate 5 insightful observations about sleep quality based on the "
            "following 7-day statistics. Focus on patterns, anomalies, and meaningful "
            "comparisons to healthy benchmarks. Express observations in plain English with "
            "1-2 sentences each, highlighting the most salient findings.\n\n"
            "In your insights, include specific percentile comparisons where relevant "
            "(e.g., 'top 10% of all nights'). When discussing correlations, express their "
            "practical significance (e.g., 'X% of the variability in Y can be explained by Z').\n\n"
            "Include precise numbers and avoid vague statements. Each insight should focus on a "
            "specific aspect of sleep health and its implications.\n\n"
            "Statistics:\n" + json.dumps(stats.model_dump(), indent=2)
        )
        return await self.generate_narrative(prompt)

    async def generate_phone_narrative(self, stats: PhoneStats) -> str:
        """Generate narrative insights for phone usage data."""
        prompt = (
            "Generate 5 insightful observations about phone usage and its impact on "
            "sleep based on the following statistics. Focus specifically on how phone "
            "behavior affects different aspects of sleep quality and quantity.\n\n"
            "Emphasize cause-effect relationships where possible, and express correlations "
            "in terms of variance explained (e.g., 'X% of the variability in deep sleep "
            "can be explained by screen time before bed'). Compare usage patterns to typical "
            "benchmarks using percentile rankings where available.\n\n"
            "Each insight should be 1-2 sentences, clear, precise, and highlight a specific "
            "finding rather than general statements. Include actionable implications when possible.\n\n"
            "Statistics:\n" + json.dumps(stats.model_dump(), indent=2)
        )
        return await self.generate_narrative(prompt)

    async def generate_health_narrative(self, stats: HealthStats) -> str:
        """Generate narrative insights for health data."""
        prompt = (
            "Generate 5 insightful observations about health metrics and their "
            "impact on sleep based on the following statistics. Focus on how physical "
            "activity patterns specifically influence different aspects of sleep quality and recovery.\n\n"
            "Highlight the strongest correlations and explain them in terms of variance explained "
            "(e.g., 'X% of the variability in deep sleep can be explained by activity level'). "
            "Note day-to-day patterns, anomalies, and intensity of activity relative to recommendations.\n\n"
            "Each insight should be 1-2 sentences, specific rather than general, with precise values "
            "and percentages. Avoid vague statements. Where possible, suggest what the causal relationship "
            "might be, even if correlation doesn't prove causation.\n\n"
            "Statistics:\n" + json.dumps(stats.model_dump(), indent=2)
        )
        return await self.generate_narrative(prompt)
