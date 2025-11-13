"""
Generator Loader

Intelligently loads the best available generator based on configuration.
Implements fallback chain: Local → API → Error
"""

import yaml
import os
from typing import Dict, Optional
from .base import StoryGenerator
# Lazy import LocalModelGenerator to avoid torch dependency when not needed
# from .local_generator import LocalModelGenerator
from .api_generator import APIGenerator
from .custom_generator import CustomGenerator

try:
    from .vllm_generator import vLLMGenerator
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class GeneratorLoader:
    """
    Smart generator loader with fallback chain.
    
    Loading Priority:
    1. Try configured mode (local/api)
    2. If fails, try fallback modes
    3. If all fail, raise error
    
    Example:
        loader = GeneratorLoader()  # Loads from config/generator_config.yaml
        result = await loader.generate({"user_input": "..."})
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize loader.

        Args:
            config_path: Path to YAML config file (default: config/generator_config.yaml)
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config",
                "generator_config.yaml"
            )

        self.config_path = config_path
        self.config = self._load_config()
        self.generator: Optional[StoryGenerator] = None
        self.fallback_attempted = False

        self._load_generator()

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            print(f"⚠️  Config not found: {self.config_path}")
            print(f"   Using default configuration")
            return self._default_config()

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            "generator": {
                "mode": "api",  # Default to API for easy setup
                "local": {
                    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                    "device": "cuda",
                    "quantization": "4bit"
                },
                "api": {
                    "provider": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o-mini"
                }
            }
        }

    def _load_generator(self):
        """Load generator with fallback chain."""
        gen_config = self.config.get("generator", {})
        mode = gen_config.get("mode", "api")

        print(f"🔄 Loading generator in {mode} mode...")

        # Try configured mode
        if mode == "local":
            if self._try_load_local(gen_config.get("local", {})):
                return
            print(f"⚠️  Local mode failed, trying API fallback...")
            if self._try_load_api(gen_config.get("api", {})):
                self.fallback_attempted = True
                return

        elif mode == "api":
            if self._try_load_api(gen_config.get("api", {})):
                return
            print(f"⚠️  API mode failed, trying local fallback...")
            if self._try_load_local(gen_config.get("local", {})):
                self.fallback_attempted = True
                return

        elif mode == "vllm":
            if self._try_load_vllm(gen_config.get("vllm", {})):
                return
            print(f"⚠️  vLLM mode failed, trying local fallback...")
            if self._try_load_local(gen_config.get("local", {})):
                self.fallback_attempted = True
                return

        elif mode == "custom":
            if self._try_load_custom(gen_config.get("custom", {})):
                return
            print(f"⚠️  Custom mode failed, trying API fallback...")
            if self._try_load_api(gen_config.get("api", {})):
                self.fallback_attempted = True
                return

        # All failed - use dummy generator for testing
        print(f"⚠️  All generator modes failed, using DUMMY generator for testing")
        if self._try_load_dummy():
            self.fallback_attempted = True
            return

        raise RuntimeError(
            "❌ No generator available!\n"
            "   - Local mode: Not available or failed to load\n"
            "   - API mode: No API key or failed\n"
            "   - Custom mode: Not configured or failed\n"
            "   - Dummy mode: Failed to load (this should never happen)\n"
            "   \n"
            "   Please configure at least one generator mode in config/generator_config.yaml"
        )

    def _try_load_local(self, config: Dict) -> bool:
        """Try to load local generator."""
        try:
            # Lazy import to avoid torch dependency when not needed
            from .local_generator import LocalModelGenerator
            self.generator = LocalModelGenerator(config)
            print(f"✅ Loaded LOCAL generator")
            return True
        except Exception as e:
            print(f"   Local generator failed: {e}")
            return False

    def _try_load_api(self, config: Dict) -> bool:
        """Try to load API generator."""
        try:
            gen = APIGenerator(config)
            if gen.available:
                self.generator = gen
                print(f"✅ Loaded API generator")
                return True
            else:
                print(f"   API generator not available (no API key)")
                return False
        except Exception as e:
            print(f"   API generator failed: {e}")
            return False

    def _try_load_vllm(self, config: Dict) -> bool:
        """Try to load vLLM generator."""
        if not VLLM_AVAILABLE:
            print(f"   vLLM not available (not installed)")
            return False

        try:
            self.generator = vLLMGenerator(config)
            print(f"✅ Loaded vLLM generator")
            return True
        except Exception as e:
            print(f"   vLLM generator failed: {e}")
            return False

    def _try_load_custom(self, config: Dict) -> bool:
        """Try to load custom generator."""
        try:
            self.generator = CustomGenerator(config)
            print(f"✅ Loaded CUSTOM generator")
            return True
        except Exception as e:
            print(f"   Custom generator failed: {e}")
            return False

    def _try_load_dummy(self) -> bool:
        """Try to load dummy generator (always succeeds, for testing)."""
        try:
            from .dummy_generator import DummyGenerator
            self.generator = DummyGenerator()
            print(f"✅ Loaded DUMMY generator (test mode)")
            return True
        except Exception as e:
            print(f"   Dummy generator failed: {e}")
            return False

    async def generate(self, input_data: Dict) -> Dict:
        """
        Generate story content using loaded generator.

        Args:
            input_data: Input data for generation

        Returns:
            Generated result dict
        """
        if not self.generator:
            raise RuntimeError("No generator loaded")

        return await self.generator.generate(input_data)

    def get_mode(self) -> str:
        """Get current generator mode."""
        if not self.generator:
            return "none"
        return self.generator.get_mode()

    def get_model_info(self) -> Dict:
        """Get current model info."""
        if not self.generator:
            return {}
        return self.generator.get_model_info()

    async def health_check(self) -> bool:
        """Check if generator is healthy."""
        if not self.generator:
            return False
        return await self.generator.health_check()

    def is_fallback(self) -> bool:
        """Check if using fallback generator."""
        return self.fallback_attempted
