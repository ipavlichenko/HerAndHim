"""Typed outbound request purposes."""

from enum import Enum


class OutboundPurpose(str, Enum):
    DIALOGUE = "dialogue"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"

    @classmethod
    def parse(cls, value: object) -> "OutboundPurpose":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError("outbound purpose is required")
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown outbound purpose: {value!r}") from exc
