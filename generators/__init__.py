"""
StoryFi Generators Module

This module provides flexible story generation backends for Bittensor miners.
Following the decentralization philosophy, miners can choose any implementation:
- Local GPU models (recommended)
- API services (fallback)
- Custom scripts

Example:
    from generators import GeneratorLoader

    # Loads generator based on config/generator_config.yaml
    loader = GeneratorLoader()
    
    result = await loader.generate({
        "user_input": "Generate a mystery story"
    })
"""

from .base import (
    StoryGenerator,
    GenerationError,
    GeneratorNotInitializedError,
    GeneratorConfigError
)
from .prompt_templates import PromptTemplateManager

__all__ = [
    "StoryGenerator",
    "GenerationError",
    "GeneratorNotInitializedError",
    "GeneratorConfigError",
    "PromptTemplateManager",
]
