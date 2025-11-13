"""
StoryFi Scoring Package
=======================

Contains scoring modules for evaluating miner responses.

Modules:
- technical: Technical score (JSON validity, schema, response time)
- structure: Structure score (story arc, character relationships, etc.)
- content: Content score (relevance, fluency, originality)

Total Score = Technical (30%) + Structure (40%) + Content (30%)
"""

from .technical import (
    calculate_technical_score,
    validate_json_structure
)

from .structure import (
    calculate_structure_score,
    is_progress_monotonic,
    validate_act_structure
)

from .content import (
    calculate_content_score,
    calculate_originality
)

__all__ = [
    # Technical
    "calculate_technical_score",
    "validate_json_structure",

    # Structure
    "calculate_structure_score",
    "is_progress_monotonic",
    "validate_act_structure",

    # Content
    "calculate_content_score",
    "calculate_originality"
]
