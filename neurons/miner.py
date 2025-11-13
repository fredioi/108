"""
StoryFi Bittensor Miner
=======================

This miner listens for story generation requests from Validators
and responds with AI-generated content using configurable generation backends.

Supports:
- Local GPU models (recommended, 1.5x rewards)
- Cloud APIs (fallback, 0.5x rewards)
- Custom implementations

Usage:
    python neurons/miner.py \
        --netuid 42 \
        --wallet.name my_miner \
        --wallet.hotkey default \
        --logging.info
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from typing import Dict, Any, Optional, Tuple

import bittensor as bt
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template.protocol import StoryGenerationSynapse
from template.utils import Timer, compute_hash
from generators.loader import GeneratorLoader

# Load environment variables
load_dotenv()


class StoryMiner:
    """
    StoryFi Miner that generates stories using configurable backends.

    The miner:
    1. Listens for requests from Validators via Bittensor network
    2. Processes 4 types of tasks: blueprint, characters, story_arc, chapters
    3. Uses GeneratorLoader to support multiple generation methods
    4. Returns JSON-formatted story content

    Supports (following Bittensor's decentralization philosophy):
    - Local GPU models (recommended)
    - Cloud APIs (fallback)
    - Custom implementations
    """

    def __init__(self, config: bt.config):
        """
        Initialize the miner.

        Args:
            config: Bittensor configuration object
        """
        self.config = config
        bt.logging.info("Initializing StoryFi Miner...")

        # Initialize Bittensor components
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = bt.metagraph(netuid=self.config.netuid, network=self.subtensor.network)

        # Initialize Generator (replaces hardcoded OpenAI)
        bt.logging.info("Loading story generator...")
        self.generator = GeneratorLoader()

        generator_mode = self.generator.get_mode()
        model_info = self.generator.get_model_info()

        bt.logging.info(f"✅ Generator Mode: {generator_mode}")
        bt.logging.info(f"✅ Model: {model_info.get('name', 'unknown')}")
        if self.generator.is_fallback():
            bt.logging.warning(f"⚠️  Using fallback generator")

        # Statistics
        self.requests_processed = 0
        self.total_generation_time = 0.0
        self.errors = 0

        bt.logging.info(f"✅ Wallet: {self.wallet.hotkey.ss58_address}")
        bt.logging.info(f"✅ Netuid: {self.config.netuid}")

    def setup_axon(self):
        """Setup and start the axon server."""
        bt.logging.info("Setting up axon...")

        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        # Log axon info for debugging
        bt.logging.info(f"📡 Axon IP: {self.axon.external_ip}")
        bt.logging.info(f"📡 Axon Port: {self.axon.external_port}")

        # Attach forward function
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority
        )

        # Start axon
        self.axon.start()

        # Register to network
        self.subtensor.serve_axon(
            netuid=self.config.netuid,
            axon=self.axon
        )

        bt.logging.info(f"✅ Axon registered: {self.axon.external_ip}:{self.axon.external_port}")
        bt.logging.info(f"✅ Registered to subnet {self.config.netuid}")

    async def forward(self, synapse: StoryGenerationSynapse) -> StoryGenerationSynapse:
        """
        Process incoming request from Validator.

        Args:
            synapse: Request synapse containing task information

        Returns:
            Synapse with generated content filled in
        """
        try:
            bt.logging.info(f"📨 Received {synapse.task_type} request")

            with Timer() as t:
                # Debug: Log incoming synapse details
                bt.logging.debug(f"🔍 MINER DEBUG: Processing {synapse.task_type} request")
                bt.logging.debug(f"   Synapse type: {type(synapse)}")
                bt.logging.debug(f"   Protocol version: {getattr(synapse, 'protocol_version', 'unknown')}")
                bt.logging.debug(f"   User input: {synapse.user_input[:100]}...")
                bt.logging.debug(f"   Blueprint present: {synapse.blueprint is not None}")
                bt.logging.debug(f"   Characters present: {synapse.characters is not None}")
                bt.logging.debug(f"   Story arc present: {synapse.story_arc is not None}")
                bt.logging.debug(f"   Chapter IDs: {synapse.chapter_ids}")

                # Build input_data from synapse fields (Protocol v3.1.0)
                input_data = {
                    "user_input": synapse.user_input,
                    "blueprint": synapse.blueprint,
                    "characters": synapse.characters,
                    "story_arc": synapse.story_arc,
                    "chapter_ids": synapse.chapter_ids,
                    "task_type": synapse.task_type  # Pass task type to generator
                }

                # Debug: Log input_data structure
                bt.logging.debug(f"🔍 MINER DEBUG: Input data structure:")
                bt.logging.debug(f"   Keys: {list(input_data.keys())}")
                for key, value in input_data.items():
                    if value is not None:
                        if isinstance(value, dict):
                            bt.logging.debug(f"   {key}: dict with {len(value)} keys: {list(value.keys())}")
                        elif isinstance(value, list):
                            bt.logging.debug(f"   {key}: list with {len(value)} items")
                        else:
                            bt.logging.debug(f"   {key}: {type(value).__name__} = {str(value)[:100]}...")
                    else:
                        bt.logging.debug(f"   {key}: None")

                # Debug: Log generator info
                bt.logging.debug(f"🔍 MINER DEBUG: Generator mode: {self.generator.get_mode()}")
                bt.logging.debug(f"🔍 MINER DEBUG: Generator model info: {self.generator.get_model_info()}")

                # Use unified generator (supports local models, APIs, etc.)
                bt.logging.debug(f"🔍 MINER DEBUG: Calling generator.generate()...")
                result = await self.generator.generate(input_data)
                bt.logging.debug(f"🔍 MINER DEBUG: Generator returned result type: {type(result)}")
                bt.logging.debug(f"🔍 MINER DEBUG: Generator result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")

                # Extract generated content from generator response
                generated_content = result.get("generated_content", "")
                bt.logging.debug(f"🔍 MINER DEBUG: Generated content length: {len(generated_content)}")
                bt.logging.debug(f"🔍 MINER DEBUG: Generated content preview: {generated_content[:200]}...")

                # Try to parse as JSON if task expects structured output
                try:
                    if generated_content:
                        # Clean markdown wrappers if present
                        content = generated_content.strip()
                        if content.startswith("```json"):
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif content.startswith("```"):
                            content = content.split("```")[1].split("```")[0].strip()

                        # Parse JSON
                        output_data = json.loads(content)

                        # Validate format matches task type (Protocol v3.2.0)
                        # ALL tasks must return Dict (JSON object), never List (JSON array)
                        # - blueprint: returns {title, genre, setting, ...}
                        # - characters: returns {characters: [...]}
                        # - story_arc: returns {title, chapters, arcs, ...}
                        # - chapters: returns {chapters: [...]}
                        if isinstance(output_data, list):
                            bt.logging.warning(
                                f"⚠️  {synapse.task_type} task returned array instead of object. "
                                f"LLM misunderstood the prompt format."
                            )
                            # Wrap in error object with helpful context
                            output_data = {
                                "error": f"Format mismatch: {synapse.task_type} must return JSON object, not array",
                                "hint": "Check your prompt templates - all tasks require object format {...}",
                                "raw_output": output_data
                            }
                    else:
                        output_data = {"error": "Empty response from generator"}
                except json.JSONDecodeError as e:
                    # If not JSON, return raw text with error info
                    bt.logging.warning(f"⚠️  JSON parsing failed for {synapse.task_type}: {e}")
                    bt.logging.debug(f"   Raw content (first 500 chars): {generated_content[:500]}")
                    output_data = {
                        "error": f"JSON parsing failed: {str(e)}",
                        "generated_text": generated_content,
                        "hint": "LLM did not return valid JSON format"
                    }

            # Fill response fields (Protocol v3.2.0)
            synapse.output_data = output_data
            synapse.generation_time = t.elapsed
            synapse.miner_version = "3.2.0"  # Match protocol version

            # Populate model_info for transparency (Protocol v3.2.0)
            synapse.model_info = {
                "mode": self.generator.get_mode(),
                "name": self.generator.get_model_info().get("name", "unknown"),
                "version": self.generator.get_model_info().get("version"),
                "provider": self.generator.get_model_info().get("provider"),
                "parameters": self.generator.get_model_info().get("parameters", {})
            }

            # Update statistics
            self.requests_processed += 1
            self.total_generation_time += t.elapsed

            bt.logging.success(
                f"✅ Generated {synapse.task_type} in {t.elapsed:.2f}s "
                f"(output: {len(json.dumps(result, ensure_ascii=False))} chars)"
            )

            return synapse

        except Exception as e:
            self.errors += 1
            bt.logging.error(f"❌ Error processing request: {e}")
            bt.logging.error(traceback.format_exc())

            synapse.output_data = {"error": str(e)}
            synapse.generation_time = 0.0
            synapse.miner_version = "2.0.0"

            # Include model_info even in error case for transparency
            synapse.model_info = {
                "mode": self.generator.get_mode(),
                "name": self.generator.get_model_info().get("name", "unknown"),
                "version": self.generator.get_model_info().get("version"),
                "provider": self.generator.get_model_info().get("provider"),
                "parameters": self.generator.get_model_info().get("parameters", {})
            }

            return synapse

    # Note: All generation logic is now handled by GeneratorLoader
    # No need for task-specific generate_* functions
    # The generator handles prompt building based on task_type

    def blacklist(self, synapse: StoryGenerationSynapse) -> Tuple[bool, str]:
        """
        Determine if request should be blacklisted.

        Args:
            synapse: Incoming request

        Returns:
            Tuple of (should_blacklist, reason)
        """
        # Accept all requests for now
        # Can add blacklisting logic later (e.g., known malicious validators)
        return False, ""

    def priority(self, synapse: StoryGenerationSynapse) -> float:
        """
        Determine priority for request processing.

        Args:
            synapse: Incoming request

        Returns:
            Priority score (higher = more priority)
        """
        # Give higher priority to validators with higher stake
        validator_hotkey = synapse.validator_hotkey
        if validator_hotkey and validator_hotkey in self.metagraph.hotkeys:
            uid = self.metagraph.hotkeys.index(validator_hotkey)
            stake = self.metagraph.S[uid].item()
            return stake
        return 0.0

    async def run(self):
        """Main run loop."""
        bt.logging.info("🚀 Starting miner...")

        # Setup axon
        self.setup_axon()

        # Keep alive and print stats
        try:
            while True:
                await asyncio.sleep(60)

                # Print statistics
                avg_time = (
                    self.total_generation_time / self.requests_processed
                    if self.requests_processed > 0
                    else 0.0
                )

                bt.logging.info(
                    f"📊 Stats: "
                    f"Requests={self.requests_processed}, "
                    f"AvgTime={avg_time:.2f}s, "
                    f"Errors={self.errors}"
                )

                # Resync metagraph
                self.metagraph.sync(subtensor=self.subtensor)

        except KeyboardInterrupt:
            bt.logging.info("🛑 Shutting down miner...")
            self.axon.stop()


def get_config():
    """Get configuration from command line arguments."""
    parser = argparse.ArgumentParser()

    # Bittensor arguments
    parser.add_argument("--netuid", type=int, default=108, help="Subnet netuid (StoryFi subnet ID)")
    parser.add_argument("--wallet.name", type=str, default="miner", help="Wallet name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", help="Wallet hotkey")
    parser.add_argument("--subtensor.network", type=str, default="finney", help="Bittensor network (finney=mainnet, test=testnet)")
    parser.add_argument("--subtensor.chain_endpoint", type=str, default=None, help="Subtensor chain endpoint")
    parser.add_argument("--logging.info", action="store_true", help="Enable info logging")
    parser.add_argument("--logging.debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--axon.port", type=int, default=8091, help="Axon port")
    parser.add_argument("--axon.external_ip", type=str, default=None, help="External IP address (required for cloud/NAT servers)")

    # Parse and add bittensor config
    config = bt.config(parser)

    return config


def main():
    """Main entry point."""
    config = get_config()

    # Setup logging
    bt.logging.set_trace(config.logging.debug)
    bt.logging.set_debug(config.logging.debug)
    bt.logging.set_info(config.logging.info)

    # Create and run miner
    miner = StoryMiner(config)
    asyncio.run(miner.run())


if __name__ == "__main__":
    main()
