"""
StoryFi Bittensor Validator
============================

This validator:
1. Generates tasks for miners
2. Queries miners with different task types
3. Evaluates responses using scoring system
4. Calculates and sets weights on-chain
5. Implements anti-cheating mechanisms

Usage:
    python neurons/validator.py \
        --netuid 42 \
        --wallet.name my_validator \
        --wallet.hotkey default \
        --logging.info
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

import bittensor as bt
import torch
import yaml
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template.protocol import (
    StoryGenerationSynapse,
    create_blueprint_synapse,
    create_characters_synapse,
    create_story_arc_synapse,
    create_chapters_synapse
)
from template.utils import (
    Timer,
    exponential_moving_average,
    normalize_weights
)
from scoring import (
    calculate_technical_score,
    calculate_structure_score,
    calculate_content_score
)

# Load environment variables
load_dotenv()


class StoryValidator:
    """
    StoryFi Validator that evaluates miners and distributes rewards.

    The validator:
    1. Periodically queries miners with story generation tasks
    2. Scores responses using 3-part system (Technical + Structure + Content)
    3. Updates EMA scores and calculates weights
    4. Sets weights on-chain every N queries
    5. Detects and blacklists cheating miners
    """

    def __init__(self, config: bt.config):
        """Initialize the validator."""
        self.config = config
        bt.logging.info("Initializing StoryFi Validator...")

        # Initialize Bittensor components
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = bt.metagraph(netuid=self.config.netuid, network=self.subtensor.network)
        self.dendrite = bt.dendrite(wallet=self.wallet)

        # Configuration
        self.query_interval = int(os.getenv("VALIDATOR_QUERY_INTERVAL", "12"))
        self.timeout = int(os.getenv("VALIDATOR_TIMEOUT", "60"))
        self.ema_alpha = float(os.getenv("EMA_ALPHA", "0.1"))
        self.temperature = float(os.getenv("SOFTMAX_TEMPERATURE", "2.0"))
        self.weight_update_frequency = int(os.getenv("WEIGHT_UPDATE_FREQ", "100"))

        # Task distribution (blueprint:40%, characters:25%, story_arc:25%, chapters:10%)
        self.task_distribution = {
            "blueprint": 0.40,
            "characters": 0.25,
            "story_arc": 0.25,
            "chapters": 0.10
        }

        # State
        self.scores = {}  # {miner_uid: ema_score}
        self.history = deque(maxlen=1000)  # Historical responses for plagiarism detection
        self.blacklist = set()  # Blacklisted miner UIDs
        self.violations = {}  # {miner_uid: violation_count}

        # Statistics
        self.total_queries = 0
        self.successful_queries = 0
        self.total_rewards = 0.0

        # Sample data for task generation
        self.sample_prompts = [
            "一个关于赛博朋克黑客的故事",
            "一个关于太空探险的故事",
            "一个古代武侠传奇故事",
            "一个末日生存的故事",
            "一个都市悬疑推理故事",
            "一个奇幻魔法世界的故事",
            "一个时间旅行的故事",
            "一个AI觉醒的故事"
        ]

        # Load model quality policy (Protocol v3.2.0)
        self.model_policy = self._load_model_policy()

        bt.logging.info(f"✅ Wallet: {self.wallet.hotkey.ss58_address}")
        bt.logging.info(f"✅ Netuid: {self.config.netuid}")
        bt.logging.info(f"✅ Query interval: {self.query_interval}s")
        bt.logging.info(f"✅ Model quality policy loaded")

    def _load_model_policy(self) -> Dict[str, Any]:
        """Load model quality policy from YAML file."""
        policy_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "validators/config/model_policy.yaml"
        )

        try:
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy = yaml.safe_load(f)
                return policy
        except Exception as e:
            bt.logging.warning(f"Failed to load model policy: {e}, using defaults")
            # Return default policy if file not found
            return {
                "quality_policy": {
                    "min_quality_score": 0.6,
                    "mode_multipliers": {
                        "local": 1.5,
                        "vllm": 1.5,
                        "custom": 1.0,
                        "api": 0.5,
                        "unknown": 0.2
                    },
                    "recommended_models": [],
                    "blacklisted_models": [],
                    "penalties": {
                        "no_model_info": 0.5
                    }
                }
            }

    def apply_model_quality_multiplier(
        self,
        base_score: float,
        model_info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply model quality policy to adjust score (Protocol v3.2.0).

        Args:
            base_score: Base score from content/structure/technical evaluation (0-100)
            model_info: Model information from miner

        Returns:
            Tuple of (adjusted_score, multiplier_breakdown)
        """
        policy = self.model_policy.get("quality_policy", {})
        multiplier_info = {
            "mode_multiplier": 1.0,
            "model_bonus": 1.0,
            "penalty": 1.0,
            "final_multiplier": 1.0
        }

        # Check if model_info is missing or unknown
        if not model_info or model_info.get("mode") == "unknown":
            penalty = policy.get("penalties", {}).get("no_model_info", 0.5)
            multiplier_info["penalty"] = penalty
            multiplier_info["final_multiplier"] = penalty
            final_score = base_score * penalty
            bt.logging.warning(f"⚠️  No model info provided, applying {penalty}x penalty")
            return final_score, multiplier_info

        mode = model_info.get("mode", "unknown")
        model_name = model_info.get("name", "unknown")

        # 1. Apply mode multiplier
        mode_multipliers = policy.get("mode_multipliers", {})
        mode_mult = mode_multipliers.get(mode, 1.0)
        multiplier_info["mode_multiplier"] = mode_mult

        # 2. Check blacklist (instant disqualification)
        blacklist = policy.get("blacklisted_models", [])
        if any(blacklisted in model_name for blacklisted in blacklist):
            bt.logging.error(f"🚫 Blacklisted model detected: {model_name}")
            return 0.0, multiplier_info

        # 3. Check recommended models for bonus
        model_bonus = 1.0
        for rec_model in policy.get("recommended_models", []):
            if rec_model["name"] in model_name:
                model_bonus = rec_model.get("bonus", 1.0)
                multiplier_info["model_bonus"] = model_bonus
                bt.logging.info(f"✨ Recommended model {model_name}, {model_bonus}x bonus")
                break

        # Calculate final multiplier
        final_multiplier = mode_mult * model_bonus
        multiplier_info["final_multiplier"] = final_multiplier

        # Apply multiplier
        final_score = base_score * final_multiplier

        # Apply minimum quality threshold
        min_quality = policy.get("min_quality_score", 0.6)
        normalized_score = base_score / 100.0
        if normalized_score < min_quality:
            bt.logging.warning(
                f"⚠️  Score {base_score:.2f} below minimum quality {min_quality*100:.2f}, "
                f"setting to 0"
            )
            return 0.0, multiplier_info

        return final_score, multiplier_info

    def select_task_type(self) -> str:
        """Randomly select a task type based on distribution."""
        return random.choices(
            list(self.task_distribution.keys()),
            weights=list(self.task_distribution.values())
        )[0]

    def create_task(self, task_type: str) -> Tuple[StoryGenerationSynapse, Dict[str, Any]]:
        """
        Create a task synapse with mock context.

        Args:
            task_type: Type of task to create

        Returns:
            Tuple of (synapse, context)
        """
        user_input = random.choice(self.sample_prompts)
        context = {"user_input": user_input}

        if task_type == "blueprint":
            synapse = create_blueprint_synapse(user_input)

        elif task_type == "characters":
            # Mock blueprint for characters task
            mock_blueprint = {
                "title": "测试故事",
                "genre": "科幻",
                "setting": "未来世界",
                "core_conflict": "人与AI的冲突",
                "themes": ["科技", "伦理"],
                "tone": "紧张",
                "target_audience": "成人"
            }
            synapse = create_characters_synapse(mock_blueprint, user_input)
            context["blueprint"] = mock_blueprint

        elif task_type == "story_arc":
            # Mock blueprint and characters
            mock_blueprint = {
                "title": "测试故事",
                "genre": "科幻",
                "setting": "未来世界",
                "core_conflict": "人与AI的冲突",
                "themes": ["科技", "伦理"],
                "tone": "紧张",
                "target_audience": "成人"
            }
            mock_characters = {
                "characters": [
                    {"id": "protagonist", "name": "主角", "archetype": "英雄"},
                    {"id": "ally", "name": "盟友", "archetype": "助手"},
                    {"id": "rival", "name": "对手", "archetype": "反派"},
                    {"id": "mentor", "name": "导师", "archetype": "智者"},
                    {"id": "wildcard", "name": "变数", "archetype": "神秘人"}
                ]
            }
            synapse = create_story_arc_synapse(mock_blueprint, mock_characters, user_input)
            context["blueprint"] = mock_blueprint
            context["characters"] = mock_characters

        elif task_type == "chapters":
            # Mock complete context
            mock_blueprint = {"title": "测试故事"}
            mock_characters = {"characters": []}
            mock_story_arc = {
                "title": "测试故事",
                "chapters": [{"id": i} for i in range(1, 13)]
            }
            chapter_ids = [1]  # v3.0.0: 使用整数而不是字符串
            synapse = create_chapters_synapse(
                mock_blueprint, mock_characters, mock_story_arc,
                chapter_ids, user_input
            )
            context["blueprint"] = mock_blueprint
            context["characters"] = mock_characters
            context["story_arc"] = mock_story_arc

        return synapse, context

    async def query_miners(
        self,
        synapse: StoryGenerationSynapse,
        miners: List[bt.AxonInfo]
    ) -> List[StoryGenerationSynapse]:
        """
        Query multiple miners with a task.

        Args:
            synapse: Task synapse
            miners: List of miner axons

        Returns:
            List of responses
        """
        try:
            responses = await self.dendrite.forward(
                axons=miners,
                synapse=synapse,
                timeout=self.timeout
            )
            return responses
        except Exception as e:
            bt.logging.error(f"Error querying miners: {e}")
            return [synapse] * len(miners)  # Return empty responses on error

    def score_response(
        self,
        response: StoryGenerationSynapse,
        context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score a miner's response using 3-part scoring system with model quality multiplier (Protocol v3.2.0).

        Scoring components:
        1. Technical Score (30 points): Validity, performance, format
        2. Structure Score (40 points): Completeness, field quality
        3. Content Score (30 points): Creativity, coherence, relevance
        4. Model Quality Multiplier: Based on model mode, name, and quality policy

        Args:
            response: Miner's response synapse
            context: Task context

        Returns:
            Tuple of (final_score, breakdown) where final_score includes model quality multiplier
        """
        # Initialize breakdown
        breakdown = {
            "technical": 0.0,
            "structure": 0.0,
            "content": 0.0,
            "total": 0.0
        }

        # Get output data (v3.0.0: direct Dict access)
        data = response.output_data
        if not data:
            return 0.0, breakdown

        # Get required fields
        required_fields = response.get_required_output_fields()

        # 1. Technical Score (30 points)
        # v3.0.0: convert output_data to JSON string for technical scoring
        output_json_str = json.dumps(data, ensure_ascii=False)
        tech_score, tech_breakdown = calculate_technical_score(
            output_json_str,
            response.generation_time,
            response.task_type,
            required_fields
        )
        breakdown["technical"] = tech_score
        breakdown["technical_breakdown"] = tech_breakdown

        # 2. Structure Score (40 points)
        struct_score, struct_breakdown = calculate_structure_score(
            data,
            response.task_type
        )
        breakdown["structure"] = struct_score
        breakdown["structure_breakdown"] = struct_breakdown

        # 3. Content Score (30 points)
        content_score, content_breakdown = calculate_content_score(
            data,
            context,
            response.task_type,
            history=[h["output_data"] for h in list(self.history)[-20:]],
            use_embeddings=False  # Set to True when OpenAI embeddings are available
        )
        breakdown["content"] = content_score
        breakdown["content_breakdown"] = content_breakdown

        # Base score (0-100)
        base_score = tech_score + struct_score + content_score
        breakdown["base_score"] = base_score

        # Apply model quality multiplier (Protocol v3.2.0)
        model_info = response.model_info if hasattr(response, 'model_info') else {}
        final_score, multiplier_breakdown = self.apply_model_quality_multiplier(
            base_score,
            model_info
        )

        # Add multiplier info to breakdown
        breakdown["model_quality"] = multiplier_breakdown
        breakdown["total"] = final_score

        return final_score, breakdown

    def detect_plagiarism(
        self,
        response: StoryGenerationSynapse,
        all_responses: List[StoryGenerationSynapse]
    ) -> Tuple[bool, str, float]:
        """
        Detect plagiarism in response (Protocol v3.0.0).

        Args:
            response: Current response to check
            all_responses: All responses in current batch

        Returns:
            Tuple of (is_plagiarism, type, similarity)
        """
        # First check if response is None or doesn't have output_data
        if response is None or not hasattr(response, 'output_data'):
            return False, "", 0.0

        current_data = response.output_data
        if not current_data:
            return False, "", 0.0

        # Check against current batch (cross-miner copying)
        for other in all_responses:
            if other == response:
                continue

            # Skip if other response is None or invalid
            if other is None or not hasattr(other, 'output_data'):
                continue

            other_data = other.output_data
            if not other_data:
                continue

            similarity = self.calculate_similarity(current_data, other_data)

            if similarity > 0.95:
                return True, "cross_miner_copy", similarity

        # Check against history (template reuse)
        for historical in list(self.history)[-50:]:
            historical_data = historical.get("output_data")
            if not historical_data:
                continue

            similarity = self.calculate_similarity(current_data, historical_data)

            if similarity > 0.90:
                return True, "template_reuse", similarity

        return False, "original", 0.0

    def calculate_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate simple similarity between two data objects."""
        str1 = json.dumps(data1, ensure_ascii=False, sort_keys=True)
        str2 = json.dumps(data2, ensure_ascii=False, sort_keys=True)

        # Character-level bigrams
        def get_bigrams(s: str) -> set:
            return set(s[i:i+2] for i in range(len(s) - 1))

        bigrams1 = get_bigrams(str1)
        bigrams2 = get_bigrams(str2)

        if not bigrams1 or not bigrams2:
            return 0.0

        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)

        return intersection / union if union > 0 else 0.0

    def _is_axon_valid(self, axon: bt.AxonInfo) -> bool:
        """
        Check if an axon is valid for connection.

        Filters out:
        - 0.0.0.0 IP addresses (unregistered/invalid)
        - Invalid ports
        - Missing hotkeys

        Args:
            axon: AxonInfo object to check

        Returns:
            True if axon is valid, False otherwise
        """
        # Check if IP is not None or empty
        if not axon.ip:
            return False

        # Check if IP is not 0.0.0.0
        if axon.ip == "0.0.0.0":
            bt.logging.debug(f"Filtered axon with 0.0.0.0 IP (hotkey: {axon.hotkey[:8]}...)")
            return False

        # Check if port is valid (not 0 and in valid range)
        if axon.port <= 0 or axon.port > 65535:
            bt.logging.debug(f"Filtered axon with invalid port {axon.port}")
            return False

        # Check if hotkey is not empty
        if not axon.hotkey:
            bt.logging.debug("Filtered axon with missing hotkey")
            return False

        return True

    def update_ema_scores(self, new_scores: Dict[int, float]):
        """Update EMA scores for miners."""
        for uid, score in new_scores.items():
            if uid not in self.scores:
                self.scores[uid] = score
            else:
                self.scores[uid] = exponential_moving_average(
                    score,
                    self.scores[uid],
                    self.ema_alpha
                )

    def calculate_weights(self) -> Dict[int, float]:
        """
        Calculate weights using 3-factor system (inspired by SoulX):
        - 15% Stake weight (防止新矿工作弊)
        - 75% Quality score (当前表现)
        - 10% Historical score (长期稳定性)
        """
        if not self.scores:
            return {}

        # Calculate composite scores with stake consideration
        composite_scores = {}

        for uid, quality_score in self.scores.items():
            # 1. Get stake weight
            try:
                stake = self.metagraph.S[uid].item()
                max_stake = max(self.metagraph.S).item() if len(self.metagraph.S) > 0 else 1.0
                stake_weight = stake / max_stake if max_stake > 0 else 0.0
            except Exception as e:
                bt.logging.warning(f"Failed to get stake for UID {uid}: {e}")
                stake_weight = 0.0

            # 2. Get historical score (从历史记录计算平均)
            historical_scores = [
                h["score"] for h in list(self.history)[-50:]
                if h.get("uid") == uid
            ]
            historical_avg = (
                sum(historical_scores) / len(historical_scores)
                if historical_scores else quality_score
            )
            historical_score = historical_avg / 100.0  # Normalize to 0-1

            # 3. Normalize quality score to 0-1
            normalized_quality = quality_score / 100.0

            # 4. Composite score (15% stake + 75% quality + 10% history)
            composite = (
                0.15 * stake_weight +
                0.75 * normalized_quality +
                0.10 * historical_score
            )

            composite_scores[uid] = composite

        # Apply temperature (增加竞争差异)
        incentives = {
            uid: score ** self.temperature
            for uid, score in composite_scores.items()
        }

        # Normalize
        weights = normalize_weights(incentives)

        # Apply minimum weight (防止完全归零)
        min_weight = 0.001
        weights = {uid: max(w, min_weight) for uid, w in weights.items()}

        # Re-normalize
        weights = normalize_weights(weights)

        return weights

    async def set_weights(self):
        """Set weights on blockchain."""
        try:
            weights_dict = self.calculate_weights()

            if not weights_dict:
                bt.logging.warning("No weights to set")
                return

            # Log weight calculation details
            bt.logging.info("Weight calculation details:")
            sorted_weights = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
            for uid, weight in sorted_weights[:10]:  # Show top 10
                # Get miner info
                try:
                    axon = self.metagraph.axons[uid]
                    stake = self.metagraph.S[uid].item()
                    ema_score = self.scores.get(uid, 0)
                    bt.logging.info(
                        f"  UID {uid}: weight={weight:.4f} "
                        f"(score={ema_score:.2f}, stake={stake:.2f}τ, "
                        f"ip={axon.ip}:{axon.port})"
                    )
                except Exception as e:
                    bt.logging.debug(f"  UID {uid}: weight={weight:.4f} (error getting details: {e})")

            if len(sorted_weights) > 10:
                bt.logging.info(f"  ... and {len(sorted_weights) - 10} more miners")

            # Convert to lists
            uids = list(weights_dict.keys())
            weights = [weights_dict[uid] for uid in uids]

            # Convert to tensors
            uids_tensor = torch.tensor(uids, dtype=torch.int64)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)

            bt.logging.info(f"Submitting weights to chain...")
            bt.logging.debug(f"  UIDs: {uids[:20]}{'...' if len(uids) > 20 else ''}")
            bt.logging.debug(f"  Weights sum: {sum(weights):.4f}")

            # Set weights
            success, message = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids_tensor,
                weights=weights_tensor,
                wait_for_inclusion=False,
                wait_for_finalization=False
            )

            if success:
                bt.logging.success(f"✅ Weights set successfully: {len(uids)} miners")
                bt.logging.info(f"   Transaction broadcast to chain")
            else:
                bt.logging.error(f"❌ Failed to set weights: {message}")

        except Exception as e:
            bt.logging.error(f"Error setting weights: {e}")
            bt.logging.error(traceback.format_exc())

    async def run_step(self):
        """Run one validation step."""
        try:
            # 1. Select task type
            task_type = self.select_task_type()
            bt.logging.info(f"🎯 Task type: {task_type}")

            # 2. Create task
            synapse, context = self.create_task(task_type)

            # 3. Select miners (top 70% + random 30%)
            all_miners = self.metagraph.axons
            available_miners = [
                (i, axon) for i, axon in enumerate(all_miners)
                if i not in self.blacklist and self._is_axon_valid(axon)
            ]

            # Log filtered miners count
            filtered_count = len(all_miners) - len(available_miners) - len(self.blacklist)
            if filtered_count > 0:
                bt.logging.debug(f"Filtered out {filtered_count} invalid axons (0.0.0.0 or invalid config)")

            if not available_miners:
                bt.logging.warning("No available miners")
                await asyncio.sleep(self.query_interval)
                return

            # Select top performers + random exploration
            num_miners = min(10, len(available_miners))
            sorted_miners = sorted(
                available_miners,
                key=lambda x: self.scores.get(x[0], 0),
                reverse=True
            )

            top_k = int(num_miners * 0.7)
            explore_k = num_miners - top_k

            selected = sorted_miners[:top_k]
            if len(sorted_miners) > top_k:
                selected += random.sample(sorted_miners[top_k:], min(explore_k, len(sorted_miners) - top_k))

            selected_uids = [uid for uid, _ in selected]
            selected_axons = [axon for _, axon in selected]

            bt.logging.info(f"📡 Querying {len(selected_axons)} miners: {selected_uids}")

            # Log miner IPs for debugging
            for uid, axon in zip(selected_uids, selected_axons):
                bt.logging.debug(f"  → UID {uid}: {axon.ip}:{axon.port}")

            # 4. Query miners
            bt.logging.debug(f"?? VALIDATOR DEBUG: Querying {len(selected_axons)} miners with {synapse.task_type} task")
            bt.logging.debug(f"🔍 VALIDATOR DEBUG: Synapse being sent:")
            bt.logging.debug(f"   Type: {type(synapse)}")
            bt.logging.debug(f"   Task type: {synapse.task_type}")
            bt.logging.debug(f"   User input: {synapse.user_input[:100]}...")
            bt.logging.debug(f"   Protocol version: {getattr(synapse, 'protocol_version', 'unknown')}")
            
            with Timer() as t:
                responses = await self.query_miners(synapse, selected_axons)

            bt.logging.info(f"⏱️  Query completed in {t.elapsed:.2f}s")
            bt.logging.debug(f"🔍 VALIDATOR DEBUG: Received {len(responses)} responses")
            bt.logging.debug(f"🔍 VALIDATOR DEBUG: Response types: {[type(r).__name__ for r in responses]}")

            # 5. Score responses
            scores = {}
            for uid, response, axon in zip(selected_uids, responses, selected_axons):
                bt.logging.debug(f"\n{'='*60}")
                bt.logging.debug(f"Evaluating Miner UID {uid} ({axon.ip}:{axon.port})")
                bt.logging.debug(f"{'='*60}")
                bt.logging.debug(f"🔍 VALIDATOR DEBUG: Response {uid} details:")
                bt.logging.debug(f"   Type: {type(response)}")
                bt.logging.debug(f"   Is None: {response is None}")
                
                if response is not None:
                    bt.logging.debug(f"   Has output_data: {hasattr(response, 'output_data')}")
                    if hasattr(response, 'output_data'):
                        bt.logging.debug(f"   output_data type: {type(response.output_data)}")
                        bt.logging.debug(f"   output_data is None: {response.output_data is None}")
                        if response.output_data is not None:
                            bt.logging.debug(f"   output_data keys: {list(response.output_data.keys()) if isinstance(response.output_data, dict) else 'not a dict'}")
                    
                    bt.logging.debug(f"   Has generation_time: {hasattr(response, 'generation_time')}")
                    bt.logging.debug(f"   Has task_type: {hasattr(response, 'task_type')}")
                    bt.logging.debug(f"   Has model_info: {hasattr(response, 'model_info')}")
                    
                    # Log all attributes for debugging
                    all_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
                    bt.logging.debug(f"   All attributes: {all_attrs}")

                # Check and log response status (INFO level for operators to see without --logging.debug)
                if response is None:
                    bt.logging.warning(
                        f"⚠️  Miner {uid} ({axon.ip}:{axon.port}): Response is None "
                        f"(connection failed, timeout, or invalid response)"
                    )
                    scores[uid] = 0.0
                    continue

                # Check if response is a dict (serialization error)
                if isinstance(response, dict):
                    bt.logging.warning(f"⚠️  Miner {uid}: Response is dict instead of StoryGenerationSynapse object")
                    bt.logging.warning(f"   This indicates a serialization/deserialization error")
                    bt.logging.warning(f"   Response keys: {list(response.keys()) if response else 'empty'}")
                    
                    # Try to extract data from dict response
                    # Handle two cases:
                    # 1. dict contains "output_data" field (proper serialization)
                    # 2. dict directly contains the JSON structure (serialization error)
                    
                    try:
                        # Create a mock synapse object from dict
                        class MockSynapse:
                            def __init__(self, data):
                                # Case 1: dict has output_data field
                                if "output_data" in data:
                                    self.output_data = data.get("output_data")
                                    self.generation_time = data.get("generation_time", 0.0)
                                    self.task_type = data.get("task_type", "unknown")
                                    self.model_info = data.get("model_info", {})
                                    self.miner_version = data.get("miner_version", "unknown")
                                else:
                                    # Case 2: dict IS the output_data (serialization error)
                                    self.output_data = data
                                    self.generation_time = data.get("generation_time", 0.0)
                                    self.task_type = self._infer_task_type(data)
                                    self.model_info = data.get("model_info", {})
                                    self.miner_version = data.get("miner_version", "unknown")
                            
                            def _infer_task_type(self, data):
                                """Infer task type from data structure"""
                                if "title" in data and "genre" in data and "setting" in data:
                                    return "blueprint"
                                elif "characters" in data:
                                    return "characters"
                                elif "chapters" in data and "arcs" in data:
                                    return "story_arc"
                                elif "chapters" in data and any("content" in ch for ch in data.get("chapters", [])):
                                    return "chapters"
                                else:
                                    return "unknown"
                            
                            def get_required_output_fields(self):
                                """Get required output fields for the task type"""
                                if self.task_type == "blueprint":
                                    return ["title", "genre", "setting", "core_conflict", "themes", "tone", "target_audience"]
                                elif self.task_type == "characters":
                                    return ["characters"]
                                elif self.task_type == "story_arc":
                                    return ["title", "description", "chapters", "arcs", "themes", "hooks"]
                                elif self.task_type == "chapters":
                                    return ["chapters"]
                                else:
                                    return []
                        
                        mock_response = MockSynapse(response)
                        
                        # Check if output_data is valid
                        if mock_response.output_data is not None:
                            # Score the mock response
                            score, breakdown = self.score_response(mock_response, context)
                            scores[uid] = score
                            bt.logging.info(f"📊 Miner {uid} (dict): {score:.2f} points")
                            continue
                        else:
                            bt.logging.warning(f"   output_data is None in dict response")
                    except Exception as e:
                        bt.logging.error(f"   Failed to score dict response: {e}")
                        import traceback
                        bt.logging.debug(traceback.format_exc())
                    
                    scores[uid] = 0.0
                    continue

                if not hasattr(response, 'output_data'):
                    bt.logging.warning(
                        f"⚠️  Miner {uid} ({axon.ip}:{axon.port}): Response missing output_data attribute "
                        f"(response type: {type(response).__name__})"
                    )
                    bt.logging.info(f"   Available attributes: {[a for a in dir(response) if not a.startswith('_')][:10]}")
                    scores[uid] = 0.0
                    continue

                if response.output_data is None:
                    bt.logging.warning(
                        f"⚠️  Miner {uid} ({axon.ip}:{axon.port}): output_data is None "
                        f"(gen_time: {getattr(response, 'generation_time', 'unknown')}s)"
                    )
                    scores[uid] = 0.0
                    continue

                # Additional debug details (only with --logging.debug)
                bt.logging.debug(f"Response type: {type(response)}")
                bt.logging.debug(f"output_data type: {type(response.output_data)}")
                bt.logging.debug(f"output_data value: {response.output_data}")

                # Log response metadata
                bt.logging.debug(f"Response metadata:")
                bt.logging.debug(f"  - Generation time: {response.generation_time:.2f}s")
                bt.logging.debug(f"  - Miner version: {getattr(response, 'miner_version', 'unknown')}")
                bt.logging.debug(f"  - Model info: {getattr(response, 'model_info', {})}")

                # Log output data preview
                output_preview = str(response.output_data)[:200] + "..." if len(str(response.output_data)) > 200 else str(response.output_data)
                bt.logging.debug(f"  - Output preview: {output_preview}")

                # Check plagiarism
                is_plagiarism, plag_type, similarity = self.detect_plagiarism(response, responses)

                if is_plagiarism:
                    bt.logging.warning(f"⚠️  Miner {uid}: Plagiarism detected ({plag_type}, similarity={similarity:.2f})")
                    self.violations[uid] = self.violations.get(uid, 0) + 1

                    if self.violations[uid] >= 3:
                        self.blacklist.add(uid)
                        bt.logging.error(f"🚫 Miner {uid} blacklisted (3 violations)")

                    scores[uid] = 0.0
                    continue

                # Score response
                score, breakdown = self.score_response(response, context)
                scores[uid] = score

                # Detailed scoring breakdown
                bt.logging.info(f"📊 Miner {uid} ({axon.ip}): {score:.2f} points")
                bt.logging.debug(f"  Technical Score: {breakdown['technical']:.1f}/30")
                if 'technical_breakdown' in breakdown:
                    for k, v in breakdown['technical_breakdown'].items():
                        bt.logging.debug(f"    - {k}: {v}")

                bt.logging.debug(f"  Structure Score: {breakdown['structure']:.1f}/40")
                if 'structure_breakdown' in breakdown:
                    for k, v in breakdown['structure_breakdown'].items():
                        bt.logging.debug(f"    - {k}: {v}")

                bt.logging.debug(f"  Content Score: {breakdown['content']:.1f}/30")
                if 'content_breakdown' in breakdown:
                    for k, v in breakdown['content_breakdown'].items():
                        bt.logging.debug(f"    - {k}: {v}")

                bt.logging.debug(f"  Base Score: {breakdown.get('base_score', 0):.2f}/100")

                if 'model_quality' in breakdown:
                    bt.logging.debug(f"  Model Quality Multiplier: {breakdown['model_quality'].get('final_multiplier', 1.0):.2f}x")

                bt.logging.debug(f"  Final Score: {score:.2f}")

                # Record history (v3.0.0: store output_data instead of output_json)
                self.history.append({
                    "uid": uid,
                    "task_type": task_type,
                    "output_data": response.output_data,
                    "score": score,
                    "timestamp": time.time()
                })

            # 6. Update EMA scores
            self.update_ema_scores(scores)

            # 7. Update statistics
            self.total_queries += 1
            self.successful_queries += len([s for s in scores.values() if s > 0])
            self.total_rewards += sum(scores.values())

            # 8. Set weights periodically
            if self.total_queries % self.weight_update_frequency == 0:
                bt.logging.info(f"\n{'='*60}")
                bt.logging.info(f"Setting weights (query #{self.total_queries})")
                bt.logging.info(f"{'='*60}")
                await self.set_weights()

        except Exception as e:
            bt.logging.error(f"Error in run_step: {e}")
            bt.logging.error(traceback.format_exc())

    async def run(self):
        """Main run loop."""
        bt.logging.info("🚀 Starting validator...")

        # Sync metagraph
        self.metagraph.sync(subtensor=self.subtensor)
        bt.logging.info(f"📡 Metagraph synced: {len(self.metagraph.axons)} miners")

        try:
            while True:
                await self.run_step()

                # Print statistics
                if self.total_queries % 10 == 0:
                    avg_score = (
                        self.total_rewards / self.successful_queries
                        if self.successful_queries > 0
                        else 0.0
                    )

                    bt.logging.info(
                        f"\n📈 Statistics:\n"
                        f"  Total queries: {self.total_queries}\n"
                        f"  Successful: {self.successful_queries}\n"
                        f"  Avg score: {avg_score:.2f}\n"
                        f"  Blacklisted: {len(self.blacklist)}\n"
                        f"  Active miners: {len(self.scores)}"
                    )

                # Sync metagraph periodically
                if self.total_queries % 100 == 0:
                    self.metagraph.sync(subtensor=self.subtensor)

                # Wait before next query
                await asyncio.sleep(self.query_interval)

        except KeyboardInterrupt:
            bt.logging.info("🛑 Shutting down validator...")

            # Final weight update
            await self.set_weights()


def get_config():
    """Get configuration from command line arguments."""
    parser = argparse.ArgumentParser()

    # Add Bittensor standard arguments
    bt.subtensor.add_args(parser)
    bt.wallet.add_args(parser)
    bt.logging.add_args(parser)

    # Add custom arguments
    parser.add_argument("--netuid", type=int, default=108, help="Subnet netuid (StoryFi subnet ID)")

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

    # Create and run validator
    validator = StoryValidator(config)
    asyncio.run(validator.run())


if __name__ == "__main__":
    main()
