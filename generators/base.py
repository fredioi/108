"""
Story Generator Base Class

This module defines the abstract interface for all story generators.
Following Bittensor's decentralization philosophy, miners can implement
any generation method as long as it conforms to this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import time
from .prompt_templates import PromptTemplateManager


class StoryGenerator(ABC):
    """
    Abstract base class for story generators.

    Miners can implement this interface using:
    - Local GPU models (recommended, 1.5x rewards)
    - API services (fallback, 0.5x rewards)
    - Custom scripts (advanced, 1.0x rewards)
    - Any other method that produces valid output

    Validators only judge output quality, not generation method.
    """

    def __init__(self, config: Dict):
        """
        Initialize generator with configuration.

        Args:
            config: Dictionary containing generator-specific configuration
        """
        self.config = config
        self.initialized = False
        self.init_time = None

        # Initialize prompt template manager (optional, can be disabled)
        template_dir = config.get('template_dir')
        self.use_templates = config.get('use_templates', True)

        if self.use_templates:
            try:
                self.template_manager = PromptTemplateManager(template_dir)
            except Exception as e:
                print(f"⚠️  Template system disabled: {e}")
                self.use_templates = False
                self.template_manager = None
        else:
            self.template_manager = None

    @abstractmethod
    async def generate(self, input_data: Dict) -> Dict:
        """
        Generate story content from input data.

        Args:
            input_data: Dictionary containing:
                - user_input: str - User's story request
                - blueprint: Optional[Dict] - Story blueprint
                - characters: Optional[Dict] - Character definitions
                - story_arc: Optional[Dict] - Story arc structure
                - chapter_ids: Optional[List[int]] - Chapter IDs

        Returns:
            Dict containing:
                - generated_content: str - The generated story content
                - model: str - Model identifier
                - mode: str - Generation mode (local/api/custom)
                - generation_time: float - Time taken to generate
                - metadata: Optional[Dict] - Additional metadata

        Raises:
            GenerationError: If generation fails
        """
        pass

    @abstractmethod
    def get_mode(self) -> str:
        """
        Get the generation mode.

        Returns:
            str: One of "local", "api", "custom"
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """
        Get information about the model being used.

        Returns:
            Dict containing:
                - name: str - Model name
                - version: Optional[str] - Model version
                - provider: Optional[str] - Provider (for API mode)
                - parameters: Optional[Dict] - Model parameters
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the generator is healthy and ready.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass

    def is_initialized(self) -> bool:
        """Check if generator is initialized."""
        return self.initialized

    def get_init_time(self) -> Optional[float]:
        """Get initialization time."""
        return self.init_time

    async def warmup(self) -> bool:
        """
        Warm up the generator (optional).
        Useful for loading models into memory.

        Returns:
            bool: True if warmup successful
        """
        return True

    def _build_prompt(self, input_data: Dict) -> str:
        """
        Build generation prompt from input data.

        Uses template system if enabled, otherwise falls back to simple format.

        Args:
            input_data: Input data containing user_input, task_type, etc.

        Returns:
            Formatted prompt string
        """
        task_type = input_data.get('task_type', 'generic')

        # Use template system if enabled
        if self.use_templates and self.template_manager:
            try:
                return self.template_manager.render(task_type, input_data)
            except Exception as e:
                print(f"⚠️  Template rendering failed: {e}, using fallback")

        # Fallback: simple prompt construction
        return self._build_simple_prompt(input_data)

    def _build_simple_prompt(self, input_data: Dict) -> str:
        """
        Build a simple prompt without templates (fallback).

        Args:
            input_data: Input data dict

        Returns:
            Simple formatted prompt
        """
        user_input = input_data.get("user_input", "")
        blueprint = input_data.get("blueprint", {})
        characters = input_data.get("characters", {})
        story_arc = input_data.get("story_arc", {})

        # Build structured prompt
        prompt_parts = [
            "You are a creative story writer for an interactive story game.",
            "Generate engaging, immersive story content based on the following:",
            ""
        ]

        # Add user input
        if user_input:
            prompt_parts.append(f"User Request: {user_input}")
            prompt_parts.append("")

        # Add blueprint if available
        if blueprint:
            prompt_parts.append(f"Story Blueprint: {blueprint}")
            prompt_parts.append("")

        # Add characters if available
        if characters:
            prompt_parts.append(f"Characters: {characters}")
            prompt_parts.append("")

        # Add story arc if available
        if story_arc:
            prompt_parts.append(f"Story Arc: {story_arc}")
            prompt_parts.append("")

        prompt_parts.append("Generated Story:")

        return "\n".join(prompt_parts)


class GenerationError(Exception):
    """Raised when story generation fails."""
    pass


class GeneratorNotInitializedError(Exception):
    """Raised when trying to use uninitialized generator."""
    pass


class GeneratorConfigError(Exception):
    """Raised when generator configuration is invalid."""
    pass
