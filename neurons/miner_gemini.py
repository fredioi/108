"""
StoryFi Bittensor Miner (Google Gemini 版本)
============================================

使用 Google Gemini API 而不是 OpenAI
- 成本降低 80 倍
- 免费额度每天 1500 次请求
- 速度更快

Usage:
    python neurons/miner_gemini.py \
        --netuid 108 \
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

# Google Gemini
import google.generativeai as genai

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from template.protocol import StoryGenerationSynapse
from template.utils import Timer, compute_hash

# Load environment variables
load_dotenv()


class StoryMinerGemini:
    """
    StoryFi Miner 使用 Google Gemini API
    """

    def __init__(self, config: bt.config):
        """初始化 Miner"""
        self.config = config
        bt.logging.info("Initializing StoryFi Miner (Gemini)...")

        # Initialize Bittensor components
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = bt.metagraph(netuid=self.config.netuid, network=self.subtensor.network)

        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        # Configuration
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
        self.model = genai.GenerativeModel(self.model_name)
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "3000"))

        # Statistics
        self.requests_processed = 0
        self.total_generation_time = 0.0
        self.errors = 0

        bt.logging.info(f"✅ Wallet: {self.wallet.hotkey.ss58_address}")
        bt.logging.info(f"✅ Model: {self.model_name}")
        bt.logging.info(f"✅ Netuid: {self.config.netuid}")

    def setup_axon(self):
        """Setup and start the axon server."""
        bt.logging.info("Setting up axon...")

        self.axon = bt.axon(wallet=self.wallet, config=self.config)

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

        bt.logging.info(f"✅ Axon started on port {self.axon.external_port}")
        bt.logging.info(f"✅ Registered to subnet {self.config.netuid}")

    async def forward(self, synapse: StoryGenerationSynapse) -> StoryGenerationSynapse:
        """处理来自 Validator 的请求 (Protocol v3.0.0)"""
        try:
            bt.logging.info(f"📨 Received {synapse.task_type} request (protocol v{synapse.protocol_version})")
            bt.logging.info(f"   User input: {synapse.user_input[:50]}...")

            with Timer() as t:
                # 路由到不同的生成函数
                if synapse.task_type == "blueprint":
                    result = await self.generate_blueprint(synapse)
                elif synapse.task_type == "characters":
                    result = await self.generate_characters(synapse)
                elif synapse.task_type == "story_arc":
                    result = await self.generate_story_arc(synapse)
                elif synapse.task_type == "chapters":
                    result = await self.generate_chapters(synapse)
                else:
                    result = {"error": f"Unknown task type: {synapse.task_type}"}

            # 填充响应 (v3.0.0: 直接设置 output_data Dict)
            synapse.output_data = result
            synapse.generation_time = t.elapsed
            synapse.miner_version = "3.0.0"

            # 更新统计
            self.requests_processed += 1
            self.total_generation_time += t.elapsed

            output_size = len(json.dumps(result)) if result else 0
            bt.logging.success(
                f"✅ Generated {synapse.task_type} in {t.elapsed:.2f}s "
                f"(output: {output_size} bytes)"
            )

            return synapse

        except Exception as e:
            self.errors += 1
            bt.logging.error(f"❌ Error processing request: {e}")
            bt.logging.error(traceback.format_exc())

            synapse.output_data = {"error": str(e)}
            synapse.generation_time = 0.0
            return synapse

    async def generate_with_gemini(self, prompt: str) -> Dict[str, Any]:
        """使用 Gemini 生成内容"""
        try:
            # Gemini 是同步的，需要在 executor 中运行
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                )
            )

            content = response.text.strip()

            # 清理可能的 markdown 格式
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            # 解析 JSON
            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            bt.logging.error(f"JSON 解析失败: {e}")
            bt.logging.error(f"原始内容: {content[:500]}")
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "generated_text": content,
                "hint": "LLM did not return valid JSON format"
            }
        except Exception as e:
            bt.logging.error(f"Gemini API 错误: {e}")
            return {"error": str(e)}

    async def generate_blueprint(self, synapse: StoryGenerationSynapse) -> Dict[str, Any]:
        """生成故事蓝图"""
        user_input = synapse.user_input

        prompt = f"""你是故事设计师。根据用户输入创建故事蓝图。

用户输入: {user_input}

生成一个完整的故事蓝图，包含以下要素：

**必须以JSON格式输出**，不要有任何Markdown格式，不要有```json```标记，直接输出纯JSON。

JSON结构：
{{
  "title": "故事标题（5-30字）",
  "genre": "类型（科幻/奇幻/悬疑/爱情/历史等）",
  "setting": "背景设定（时间、地点、世界观，50-200字）",
  "core_conflict": "核心冲突（主要矛盾，30-150字）",
  "themes": ["主题1", "主题2", "主题3"],
  "tone": "基调（轻松/严肃/黑暗/温馨/紧张等）",
  "target_audience": "目标读者（青少年/成人/全年龄等）"
}}

要求：
1. title必须吸引人且与用户输入相关
2. setting要具体且有画面感
3. core_conflict要有张力和深度
4. themes包含2-5个主题
5. 所有内容必须用中文

只输出JSON，不要有任何其他文字。"""

        return await self.generate_with_gemini(prompt)

    async def generate_characters(self, synapse: StoryGenerationSynapse) -> Dict[str, Any]:
        """生成5个角色"""
        blueprint = synapse.blueprint
        user_input = synapse.user_input

        prompt = f"""你是角色设计师。基于故事蓝图创建5个独特的角色。

故事蓝图:
{json.dumps(blueprint, ensure_ascii=False, indent=2)}

原始输入: {user_input}

生成5个角色，必须包含以下ID：
1. protagonist（主角）
2. ally（盟友）
3. rival（对手）
4. mentor（导师）
5. wildcard（不可预测的角色）

**必须以JSON格式输出**，不要有任何Markdown格式。

JSON结构：
{{
  "characters": [
    {{
      "id": "protagonist",
      "name": "角色名字（2-10字）",
      "archetype": "角色原型（英雄/反叛者/智者等）",
      "background": "背景故事（50-200字）",
      "motivation": "动机（为什么做这件事）",
      "skills": ["技能1", "技能2", "技能3"],
      "personality_traits": ["性格1", "性格2", "性格3"],
      "relationships": {{
        "ally": "与盟友的关系",
        "rival": "与对手的关系"
      }}
    }},
    ... （其他4个角色）
  ]
}}

要求：
1. 5个角色必须性格迥异
2. 每个角色的background要有深度
3. relationships要描述角色间的互动
4. 所有内容必须用中文

只输出JSON，不要有任何其他文字。"""

        return await self.generate_with_gemini(prompt)

    async def generate_story_arc(self, synapse: StoryGenerationSynapse) -> Dict[str, Any]:
        """生成12章故事弧"""
        blueprint = synapse.blueprint
        characters = synapse.characters
        user_input = synapse.user_input

        prompt = f"""你是故事结构设计师。基于蓝图和角色创建12章故事弧。

故事蓝图:
{json.dumps(blueprint, ensure_ascii=False, indent=2)}

角色信息:
{json.dumps(characters, ensure_ascii=False, indent=2)}

原始输入: {user_input}

**必须严格按照以下JSON格式输出**，不要有任何Markdown格式，不要有```json```标记：

{{
  "title": "{blueprint.get('title', '故事标题')}",
  "description": "故事整体描述（150-300字）",
  "chapters": [
    {{
      "id": 1,
      "title": "第一章标题",
      "description": "章节描述（80-150字）",
      "storyProgress": 0.08,
      "characterFocus": ["protagonist"]
    }},
    {{
      "id": 2,
      "title": "第二章标题",
      "description": "章节描述",
      "storyProgress": 0.17,
      "characterFocus": ["protagonist", "ally"]
    }},
    ... 继续到第12章 (storyProgress: 1.0)
  ],
  "arcs": {{
    "act1": {{"chapters": [1, 2, 3], "description": "第一幕：设定和触发事件"}},
    "act2a": {{"chapters": [4, 5, 6], "description": "第二幕上：上升行动"}},
    "act2b": {{"chapters": [7, 8, 9], "description": "第二幕下：中点转折"}},
    "act3": {{"chapters": [10, 11, 12], "description": "第三幕：高潮和解决"}}
  }},
  "themes": {{
    "primary": "主要主题",
    "secondary": ["次要主题1", "次要主题2"]
  }},
  "hooks": {{
    "opening": "开场钩子（第1章）",
    "midpoint": "中点钩子（第6章）",
    "climax": "高潮钩子（第11章）"
  }}
}}

**严格要求**：
1. 必须有完整的12个章节
2. storyProgress必须递增：0.08 → 0.17 → 0.25 → 0.33 → 0.42 → 0.50 → 0.58 → 0.67 → 0.75 → 0.83 → 0.92 → 1.0
3. arcs必须包含act1, act2a, act2b, act3，每幕正好3章
4. 每章description要有情节推进
5. characterFocus要轮换不同角色
6. 所有内容必须用中文

只输出纯JSON，不要有任何其他文字。"""

        return await self.generate_with_gemini(prompt)

    async def generate_chapters(self, synapse: StoryGenerationSynapse) -> Dict[str, Any]:
        """生成章节内容"""
        blueprint = synapse.blueprint
        characters = synapse.characters
        story_arc = synapse.story_arc
        chapter_ids = synapse.chapter_ids
        user_input = synapse.user_input

        # Get chapter info from story_arc
        chapters_info = {ch["id"]: ch for ch in story_arc["chapters"]}
        selected_chapters = [chapters_info[cid] for cid in chapter_ids if cid in chapters_info]

        prompt = f"""你是章节内容作家。基于故事弧创作详细的章节内容。

需要生成的章节信息:
{json.dumps(selected_chapters, ensure_ascii=False, indent=2)}

**必须严格按照以下JSON格式输出**，不要有任何Markdown格式，不要有```json```标记：

{{
  "chapters": [
    {{
      "id": 1,
      "title": "章节标题",
      "content": "章节完整正文内容（1000-3000字）。必须包含：场景描写、人物对话、心理活动、动作描述。要有画面感，要生动具体，让读者身临其境...",
      "choices": [
        {{
          "text": "选项1：做什么事情",
          "nextChapter": 2,
          "consequences": {{
            "mood": "+10",
            "relationship_protagonist": "+5",
            "resource_gold": "-20"
          }}
        }},
        {{
          "text": "选项2：做另一件事",
          "nextChapter": 3,
          "consequences": {{
            "mood": "-5",
            "relationship_ally": "+10",
            "resource_gold": "+50"
          }}
        }}
      ]
    }}
  ]
}}

**严格要求**：
1. content字段必须是1000-3000字的完整章节内容
2. 每个章节必须有2-4个choices
3. 每个choice必须有：
   - text: 选项文字（10-30字）
   - nextChapter: 下一章ID（整数）
   - consequences: 后果对象，包含至少2个属性
4. consequences的keys可以是：
   - mood（心情）: "+10", "-5" 等
   - relationship_xxx（关系）: "protagonist", "ally", "rival" 等
   - resource_xxx（资源）: "gold", "health", "reputation" 等
5. 不同选项的consequences必须有明显差异
6. 所有内容必须用中文

只输出纯JSON，不要有任何其他文字。"""

        return await self.generate_with_gemini(prompt)

    def blacklist(self, synapse: StoryGenerationSynapse) -> Tuple[bool, str]:
        """黑名单检查"""
        # 检查协议版本
        if synapse.protocol_version != "3.0.0":
            return True, f"Incompatible protocol version: {synapse.protocol_version}, expected 3.0.0"

        return False, ""

    def priority(self, synapse: StoryGenerationSynapse) -> float:
        """优先级计算"""
        validator_hotkey = synapse.validator_hotkey
        if validator_hotkey and validator_hotkey in self.metagraph.hotkeys:
            uid = self.metagraph.hotkeys.index(validator_hotkey)
            stake = self.metagraph.S[uid].item()
            return stake
        return 0.0

    async def run(self):
        """主运行循环"""
        bt.logging.info("🚀 Starting miner (Gemini)...")

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

    # Add Bittensor standard arguments
    bt.subtensor.add_args(parser)
    bt.wallet.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)

    # Add custom arguments
    parser.add_argument("--netuid", type=int, default=108, help="Subnet netuid")

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
    miner = StoryMinerGemini(config)
    asyncio.run(miner.run())


if __name__ == "__main__":
    main()
