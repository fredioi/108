"""
vLLM Generator

Ultra-fast local GPU inference using vLLM.
Provides 2-3x faster generation compared to transformers library.

Requirements:
    pip install vllm

Hardware:
    - NVIDIA GPU with CUDA
    - 16GB+ VRAM for Llama-3.1-8B
    - 40GB+ VRAM for Llama-3.1-70B
"""

import time
import asyncio
from typing import Dict, Any, Optional, List

from .base import StoryGenerator, GenerationError

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class vLLMGenerator(StoryGenerator):
    """
    vLLM-based local model generator.
    
    Features:
    - PagedAttention for efficient memory usage
    - Continuous batching for high throughput
    - 2-3x faster than transformers
    - Multi-GPU support out of the box
    
    Recommended for:
    - High-volume generation
    - Multiple concurrent requests
    - Production deployments
    """

    def __init__(self, config: Dict):
        """
        Initialize vLLM generator.

        Args:
            config: Dict containing:
                - model_name: str (HuggingFace model name)
                - tensor_parallel_size: int (number of GPUs, default: 1)
                - gpu_memory_utilization: float (0.0-1.0, default: 0.9)
                - max_model_len: int (context length, default: None)
                - quantization: str ("awq", "squeezeLLM", or None)
                - dtype: str ("auto", "float16", "bfloat16")
        """
        super().__init__(config)

        if not VLLM_AVAILABLE:
            raise GeneratorConfigError(
                "vLLM not installed. Install with: pip install vllm"
            )

        self.model_name = config.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9)
        self.max_model_len = config.get("max_model_len")
        self.quantization = config.get("quantization")
        self.dtype = config.get("dtype", "auto")

        # Sampling parameters
        self.temperature = config.get("temperature", 0.8)
        self.top_p = config.get("top_p", 0.9)
        self.max_tokens = config.get("max_tokens", 2048)

        self.llm = None

        print(f"🔄 Initializing vLLM Generator")
        print(f"   Model: {self.model_name}")
        print(f"   GPUs: {self.tensor_parallel_size}")
        print(f"   GPU Memory: {self.gpu_memory_utilization*100:.0f}%")
        if self.quantization:
            print(f"   Quantization: {self.quantization}")

    async def _load_model(self):
        """Load model (async wrapper for sync loading)."""
        start_time = time.time()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

        self.init_time = time.time() - start_time
        self.initialized = True

        print(f"✅ vLLM model loaded in {self.init_time:.2f}s")

    def _load_model_sync(self):
        """Synchronous model loading."""
        # Build vLLM kwargs
        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "trust_remote_code": True,
        }

        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len

        if self.quantization:
            llm_kwargs["quantization"] = self.quantization

        # Initialize vLLM
        self.llm = LLM(**llm_kwargs)
        
        print(f"   Model loaded with {self.tensor_parallel_size} GPU(s)")

    async def generate(self, input_data: Dict) -> Dict:
        """
        Generate story content using vLLM.

        Args:
            input_data: Dict containing user_input, blueprint, etc.

        Returns:
            Dict with generated_content, model info, timing, etc.
        """
        if not self.initialized:
            print("🔄 Model not initialized, loading now...")
            await self._load_model()

        start_time = time.time()

        try:
            # Build prompt
            prompt = self._build_prompt(input_data)

            # Generate (run in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            generated_text = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt
            )

            generation_time = time.time() - start_time

            return {
                "generated_content": generated_text,
                "model": self.model_name,
                "mode": "vllm",
                "generation_time": generation_time,
                "metadata": {
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "quantization": self.quantization,
                    "prompt_length": len(prompt)
                }
            }

        except Exception as e:
            raise GenerationError(f"vLLM generation failed: {str(e)}")

    def _generate_sync(self, prompt: str) -> str:
        """Synchronous generation (called in thread pool)."""
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        # Generate
        outputs = self.llm.generate([prompt], sampling_params)

        # Extract text
        generated_text = outputs[0].outputs[0].text

        return generated_text

    def _build_prompt(self, input_data: Dict) -> str:
        """Build generation prompt from input data."""
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

    def get_mode(self) -> str:
        """Return 'vllm'."""
        return "vllm"

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "name": self.model_name,
            "version": None,
            "provider": "vLLM",
            "parameters": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "quantization": self.quantization,
                "dtype": self.dtype
            }
        }

    async def health_check(self) -> bool:
        """Check if model is loaded and working."""
        if not self.initialized:
            return False

        try:
            # Try a simple generation
            test_result = await self.generate({
                "user_input": "test"
            })
            return len(test_result.get("generated_content", "")) > 0
        except:
            return False

    async def warmup(self) -> bool:
        """Warm up model by loading it into memory."""
        if not self.initialized:
            await self._load_model()
        return self.initialized


class GeneratorConfigError(Exception):
    """Raised when generator configuration is invalid."""
    pass
