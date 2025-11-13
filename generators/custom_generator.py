"""
Custom Generator

Supports custom story generation implementations via:
- External scripts (Python, Node.js, etc.)
- HTTP endpoints
- Command-line tools

This allows miners to use any generation method they want.
"""

import asyncio
import json
import subprocess
import time
from typing import Dict, Any, Optional

import aiohttp

from .base import StoryGenerator, GenerationError


class CustomGenerator(StoryGenerator):
    """
    Custom generator that supports external scripts and HTTP endpoints.
    
    Supports two modes:
    1. Script mode: Executes external script with JSON input/output
    2. HTTP mode: Sends POST request to HTTP endpoint
    
    Script mode example:
        script_path: "./custom/generate.py"
        Input via stdin, output via stdout
    
    HTTP mode example:
        endpoint: "http://localhost:8000/generate"
        POST request with JSON body
    """

    def __init__(self, config: Dict):
        """
        Initialize custom generator.

        Args:
            config: Dict containing:
                - script_path: Optional[str] - Path to generation script
                - endpoint: Optional[str] - HTTP endpoint URL
                - timeout: Optional[int] - Timeout in seconds (default: 60)
                - env_vars: Optional[Dict] - Environment variables for script
        """
        super().__init__(config)

        self.script_path = config.get("script_path")
        self.endpoint = config.get("endpoint")
        self.timeout = config.get("timeout", 60)
        self.env_vars = config.get("env_vars", {})

        if not self.script_path and not self.endpoint:
            raise GeneratorConfigError(
                "Custom generator requires either 'script_path' or 'endpoint'"
            )

        self.mode = "script" if self.script_path else "http"
        self.initialized = True

        print(f"✅ Custom Generator initialized: {self.mode} mode")
        if self.script_path:
            print(f"   Script: {self.script_path}")
        if self.endpoint:
            print(f"   Endpoint: {self.endpoint}")

    async def generate(self, input_data: Dict) -> Dict:
        """
        Generate story content using custom implementation.

        Args:
            input_data: Dict containing user_input, blueprint, etc.

        Returns:
            Dict with generated_content, model info, timing, etc.
        """
        start_time = time.time()

        try:
            if self.mode == "script":
                generated_text = await self._generate_script(input_data)
            else:  # http
                generated_text = await self._generate_http(input_data)

            generation_time = time.time() - start_time

            return {
                "generated_content": generated_text,
                "model": "custom",
                "mode": "custom",
                "generation_time": generation_time,
                "metadata": {
                    "custom_mode": self.mode,
                    "script_path": self.script_path,
                    "endpoint": self.endpoint
                }
            }

        except Exception as e:
            raise GenerationError(f"Custom generation failed: {str(e)}")

    async def _generate_script(self, input_data: Dict) -> str:
        """
        Generate using external script.
        
        Protocol:
        1. Script receives JSON via stdin
        2. Script outputs JSON with "content" field to stdout
        """
        # Prepare input JSON
        input_json = json.dumps(input_data, ensure_ascii=False)

        # Build environment
        env = {**self.env_vars}

        # Run script in thread pool (blocking operation)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run_script_sync,
            input_json,
            env
        )

        return result

    def _run_script_sync(self, input_json: str, env: Dict) -> str:
        """Synchronous script execution (called in thread pool)."""
        try:
            # Execute script
            process = subprocess.run(
                [self.script_path],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**env}
            )

            if process.returncode != 0:
                raise GenerationError(
                    f"Script failed with code {process.returncode}: {process.stderr}"
                )

            # Parse output
            output = json.loads(process.stdout)
            
            if "content" not in output:
                raise GenerationError("Script output missing 'content' field")

            return output["content"]

        except subprocess.TimeoutExpired:
            raise GenerationError(f"Script timeout after {self.timeout}s")
        except json.JSONDecodeError as e:
            raise GenerationError(f"Script output is not valid JSON: {e}")
        except FileNotFoundError:
            raise GenerationError(f"Script not found: {self.script_path}")

    async def _generate_http(self, input_data: Dict) -> str:
        """
        Generate using HTTP endpoint.
        
        Protocol:
        1. POST JSON to endpoint
        2. Expect JSON response with "content" field
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    json=input_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        text = await response.text()
                        raise GenerationError(
                            f"HTTP {response.status}: {text}"
                        )

                    result = await response.json()
                    
                    if "content" not in result:
                        raise GenerationError("HTTP response missing 'content' field")

                    return result["content"]

        except aiohttp.ClientError as e:
            raise GenerationError(f"HTTP request failed: {e}")
        except asyncio.TimeoutError:
            raise GenerationError(f"HTTP request timeout after {self.timeout}s")

    def get_mode(self) -> str:
        """Return 'custom'."""
        return "custom"

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "name": "custom",
            "version": None,
            "provider": "custom",
            "parameters": {
                "mode": self.mode,
                "script_path": self.script_path,
                "endpoint": self.endpoint,
                "timeout": self.timeout
            }
        }

    async def health_check(self) -> bool:
        """Check if custom generator is available."""
        if self.mode == "script":
            # Check if script exists and is executable
            import os
            if not os.path.exists(self.script_path):
                return False
            if not os.access(self.script_path, os.X_OK):
                return False
            return True
        
        else:  # http
            # Try a simple ping to endpoint
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.endpoint.replace("/generate", "/health"),
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        return response.status == 200
            except:
                return False


class GeneratorConfigError(Exception):
    """Raised when generator configuration is invalid."""
    pass
