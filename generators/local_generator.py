"""
Local Model Generator

Uses local GPU models for story generation (recommended approach).
Supports HuggingFace Transformers with 4bit quantization for efficient memory usage.

Inspired by SoulX's architecture using local models.
"""

import torch
import time
import asyncio
from typing import Dict, Optional
from .base import StoryGenerator, GenerationError, GeneratorNotInitializedError

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LocalModelGenerator(StoryGenerator):
    """
    Local GPU model generator using HuggingFace Transformers.
    
    Features:
    - 4bit quantization (reduces 16GB model to ~4GB VRAM)
    - Flash Attention 2 support (faster inference)
    - Async generation (non-blocking)
    
    Recommended Models:
    - meta-llama/Llama-3.1-8B-Instruct (8B params, excellent quality)
    - mistralai/Mixtral-8x7B-Instruct-v0.1 (56B params, best quality)
    - Qwen/Qwen2.5-7B-Instruct (7B params, fast)
    """

    def __init__(self, config: Dict):
        """
        Initialize local model generator.

        Args:
            config: Dict containing:
                - model_name: str (e.g., "meta-llama/Llama-3.1-8B-Instruct")
                - device: str ("cuda" or "cpu", default: auto-detect)
                - quantization: str ("4bit", "8bit", or None)
                - max_memory: str (e.g., "16GB")
                - use_flash_attention: bool (default: True if available)
        """
        super().__init__(config)

        if not TRANSFORMERS_AVAILABLE:
            raise GeneratorConfigError(
                "transformers library not installed. "
                "Install with: pip install transformers accelerate bitsandbytes"
            )

        self.model_name = config.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.quantization = config.get("quantization", "4bit")
        self.max_memory = config.get("max_memory", "16GB")
        self.use_flash_attention = config.get("use_flash_attention", True)

        self.model = None
        self.tokenizer = None

        print(f"🔄 Initializing Local Model Generator")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Quantization: {self.quantization}")

    async def _load_model(self):
        """Load model and tokenizer (async wrapper for sync loading)."""
        start_time = time.time()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

        self.init_time = time.time() - start_time
        self.initialized = True

        print(f"✅ Model loaded in {self.init_time:.2f}s")

    def _load_model_sync(self):
        """Synchronous model loading."""
        # Configure quantization
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Load tokenizer
        print(f"   Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print(f"   Loading model...")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        print(f"   Model loaded on {self.device}")

    async def generate(self, input_data: Dict) -> Dict:
        """
        Generate story content using local model.

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
                "mode": "local",
                "generation_time": generation_time,
                "metadata": {
                    "device": self.device,
                    "quantization": self.quantization,
                    "prompt_length": len(prompt)
                }
            }

        except Exception as e:
            raise GenerationError(f"Local generation failed: {str(e)}")

    def _generate_sync(self, prompt: str) -> str:
        """Synchronous generation (called in thread pool)."""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Extract only the generated part (remove prompt)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    # _build_prompt is inherited from StoryGenerator base class
    # It uses the PromptTemplateManager if enabled, with fallback to simple prompts

    def get_mode(self) -> str:
        """Return 'local'."""
        return "local"

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "name": self.model_name,
            "version": None,
            "provider": "HuggingFace",
            "parameters": {
                "device": self.device,
                "quantization": self.quantization,
                "max_memory": self.max_memory,
                "flash_attention": self.use_flash_attention
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
