"""
API Generator

Uses cloud AI APIs for story generation (fallback option).
Supports OpenAI, Anthropic, Google Gemini, and custom endpoints.
"""

import os
import time
import asyncio
from typing import Dict, Optional
from .base import StoryGenerator, GenerationError

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class APIGenerator(StoryGenerator):
    """
    Cloud API generator for story generation.
    
    Supports:
    - OpenAI (GPT-4, GPT-3.5)
    - Google Gemini
    - Anthropic Claude (TODO)
    - Custom endpoints
    
    Note: API generators receive 0.5x reward multiplier.
    Local models are recommended for maximum rewards.
    """

    def __init__(self, config: Dict):
        """
        Initialize API generator.

        Args:
            config: Dict containing:
                - provider: str ("openai", "gemini", "custom")
                - api_key_env: str (environment variable name)
                - model: str (model name)
                - endpoint: Optional[str] (for custom providers)
        """
        super().__init__(config)

        self.provider = config.get("provider", "openai")
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.model = config.get("model", "gpt-4o-mini")
        self.endpoint = config.get("endpoint")

        # Get API key from environment
        self.api_key = os.getenv(api_key_env)
        
        if not self.api_key:
            print(f"⚠️  Warning: {api_key_env} not found in environment")
            print(f"   API Generator will not be available")
            self.available = False
            return

        self.available = True

        # Initialize client based on provider
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise GeneratorConfigError("openai library not installed")
            # Support custom endpoint (for OpenAI-compatible APIs like 智谱AI)
            if self.endpoint:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.endpoint)
            else:
                self.client = AsyncOpenAI(api_key=self.api_key)
        
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise GeneratorConfigError("google-generativeai not installed")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)

        self.initialized = True
        print(f"✅ API Generator initialized: {self.provider}/{self.model}")

    async def generate(self, input_data: Dict) -> Dict:
        """Generate story content using API."""
        if not self.available:
            raise GenerationError("API Generator not available (missing API key)")

        start_time = time.time()

        try:
            if self.provider == "openai":
                result = await self._generate_openai(input_data)
            elif self.provider == "gemini":
                result = await self._generate_gemini(input_data)
            else:
                raise GenerationError(f"Unsupported provider: {self.provider}")

            generation_time = time.time() - start_time

            return {
                "generated_content": result,
                "model": self.model,
                "mode": "api",
                "generation_time": generation_time,
                "metadata": {
                    "provider": self.provider
                }
            }

        except Exception as e:
            raise GenerationError(f"API generation failed: {str(e)}")

    async def _generate_openai(self, input_data: Dict) -> str:
        """Generate using OpenAI API."""
        messages = self._build_messages(input_data)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            max_tokens=2048
        )

        return response.choices[0].message.content

    async def _generate_gemini(self, input_data: Dict) -> str:
        """Generate using Gemini API."""
        prompt = self._build_prompt(input_data)

        # Run in thread pool (Gemini SDK is sync)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.client.generate_content,
            prompt
        )

        return response.text

    def _build_messages(self, input_data: Dict) -> list:
        """Build OpenAI-style messages."""
        user_input = input_data.get("user_input", "")
        blueprint = input_data.get("blueprint", {})

        content = f"Generate engaging story content based on:\n\n"
        content += f"User Request: {user_input}\n\n"
        
        if blueprint:
            content += f"Blueprint: {blueprint}\n\n"

        return [
            {
                "role": "system",
                "content": "You are a creative story writer for an interactive story game."
            },
            {
                "role": "user",
                "content": content
            }
        ]

    def _build_prompt(self, input_data: Dict) -> str:
        """Build simple prompt."""
        user_input = input_data.get("user_input", "")
        blueprint = input_data.get("blueprint", {})

        prompt = "You are a creative story writer.\n\n"
        prompt += f"User Request: {user_input}\n\n"
        
        if blueprint:
            prompt += f"Blueprint: {blueprint}\n\n"

        prompt += "Generate engaging story content:"

        return prompt

    def get_mode(self) -> str:
        """Return 'api'."""
        return "api"

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "name": self.model,
            "version": None,
            "provider": self.provider,
            "parameters": {}
        }

    async def health_check(self) -> bool:
        """Check if API is available."""
        return self.available


class GeneratorConfigError(Exception):
    """Raised when generator configuration is invalid."""
    pass
