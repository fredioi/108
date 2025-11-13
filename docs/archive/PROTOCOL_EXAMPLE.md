# StoryFi Protocol 使用示例

## 基本概念

`StoryGenerationSynapse` 是 Miner 和 Validator 之间通信的载体（类似HTTP请求/响应）。

**流程**：
```
Validator 创建 Synapse → 发送给 Miner → Miner 填充结果 → 返回给 Validator
```

## 安装依赖

```bash
pip install bittensor pydantic
```

## 1. 创建 Synapse（Validator 侧）

### 方法 1：使用 Helper 函数（推荐）

```python
from template.protocol import (
    create_blueprint_synapse,
    create_characters_synapse,
    create_story_arc_synapse,
    create_chapters_synapse
)

# Task 1: Blueprint
synapse = create_blueprint_synapse(
    user_input="一个关于赛博朋克黑客的故事"
)

print(synapse.task_type)  # "blueprint"
print(synapse.input_data)  # {"user_input": "...", "max_tokens": 1000}
```

### 方法 2：手动创建

```python
from template.protocol import StoryGenerationSynapse

synapse = StoryGenerationSynapse(
    task_type="blueprint",
    input_data={
        "user_input": "一个关于赛博朋克黑客的故事",
        "max_tokens": 1000
    }
)
```

## 2. 发送 Synapse 给 Miners（Validator 侧）

```python
import bittensor as bt

# 初始化 Dendrite（Validator 的客户端）
wallet = bt.wallet(name="validator")
dendrite = bt.dendrite(wallet=wallet)

# 获取 Metagraph（矿工列表）
metagraph = bt.metagraph(netuid=42)  # 你的子网 UID

# 选择前 10 个矿工
miners = metagraph.axons[:10]

# 发送查询
responses = await dendrite.query(
    axons=miners,
    synapse=synapse,
    timeout=60
)

# 处理响应
for i, response in enumerate(responses):
    print(f"Miner {i}:")
    print(f"  Generated in: {response.generation_time:.2f}s")
    print(f"  Output length: {len(response.output_json)}")
```

## 3. 处理 Synapse（Miner 侧）

Miner 需要实现 `forward` 函数来处理请求：

```python
import bittensor as bt
from template.protocol import StoryGenerationSynapse
import json
import time

class StoryMiner:
    def __init__(self):
        self.wallet = bt.wallet(name="miner")
        self.axon = bt.axon(wallet=self.wallet)

        # 注册 forward 函数
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority
        )

    async def forward(self, synapse: StoryGenerationSynapse) -> StoryGenerationSynapse:
        """
        处理来自 Validator 的请求

        Args:
            synapse: 包含任务信息的 Synapse

        Returns:
            填充了结果的 Synapse
        """
        start_time = time.time()

        # 根据任务类型调用不同的生成函数
        if synapse.task_type == "blueprint":
            result = await self.generate_blueprint(synapse.input_data)
        elif synapse.task_type == "characters":
            result = await self.generate_characters(synapse.input_data)
        elif synapse.task_type == "story_arc":
            result = await self.generate_story_arc(synapse.input_data)
        elif synapse.task_type == "chapters":
            result = await self.generate_chapters(synapse.input_data)
        else:
            result = {"error": "Unknown task type"}

        # 填充响应字段
        synapse.output_json = json.dumps(result, ensure_ascii=False)
        synapse.generation_time = time.time() - start_time

        return synapse

    async def generate_blueprint(self, input_data: dict) -> dict:
        """生成 Blueprint"""
        user_input = input_data["user_input"]

        # 调用 AI API（OpenAI/Claude）
        result = await call_openai_api(
            prompt=f"生成故事蓝图：{user_input}",
            system_prompt="你是故事设计师..."
        )

        # 返回符合格式的 JSON
        return {
            "title": "数字迷城",
            "genre": "赛博朋克",
            "setting": "2084年，新东京都市",
            "core_conflict": "黑客组织与AI跨国公司的对抗",
            "themes": ["自由意志", "数字监控", "人机边界"],
            "tone": "黑暗、紧张、反乌托邦",
            "target_audience": "成人科幻读者"
        }

    # ... 其他生成函数
```

## 4. 完整示例：Pipeline 测试

```python
import asyncio
from template.protocol import (
    create_blueprint_synapse,
    create_characters_synapse,
    create_story_arc_synapse,
    create_chapters_synapse
)
from template.utils import validate_json, Timer

async def test_pipeline():
    """测试完整的 4 阶段流水线"""

    # 模拟 Miner 的生成函数
    async def mock_generate(synapse):
        await asyncio.sleep(1)  # 模拟 AI 生成时间

        if synapse.task_type == "blueprint":
            synapse.output_json = json.dumps({
                "title": "测试故事",
                "genre": "科幻",
                "setting": "未来世界",
                "core_conflict": "人与AI的冲突",
                "themes": ["科技", "伦理"],
                "tone": "紧张",
                "target_audience": "成人"
            }, ensure_ascii=False)

        elif synapse.task_type == "characters":
            synapse.output_json = json.dumps({
                "characters": [
                    {
                        "id": "protagonist",
                        "name": "李明",
                        "archetype": "英雄",
                        "background": "AI工程师",
                        "motivation": "拯救世界",
                        "skills": ["编程", "黑客"],
                        "personality_traits": ["勇敢", "智慧"],
                        "relationships": {"ally": "信任"}
                    },
                    # ... 其他 4 个角色
                ]
            }, ensure_ascii=False)

        # ... 其他任务类型

        synapse.generation_time = 1.0
        return synapse

    # Stage 1: Blueprint
    print("=" * 60)
    print("Stage 1: Generating Blueprint")
    print("=" * 60)

    with Timer() as t:
        blueprint_synapse = create_blueprint_synapse("一个关于AI的故事")
        blueprint_synapse = await mock_generate(blueprint_synapse)

    is_valid, blueprint = validate_json(blueprint_synapse.output_json)
    print(f"✅ Blueprint generated in {t.elapsed:.2f}s")
    print(f"   Valid JSON: {is_valid}")
    print(f"   Title: {blueprint.get('title')}")

    # Stage 2: Characters
    print("\n" + "=" * 60)
    print("Stage 2: Generating Characters")
    print("=" * 60)

    with Timer() as t:
        characters_synapse = create_characters_synapse(blueprint, "一个关于AI的故事")
        characters_synapse = await mock_generate(characters_synapse)

    is_valid, characters = validate_json(characters_synapse.output_json)
    print(f"✅ Characters generated in {t.elapsed:.2f}s")
    print(f"   Valid JSON: {is_valid}")
    print(f"   Character count: {len(characters.get('characters', []))}")

    # Stage 3: Story Arc
    print("\n" + "=" * 60)
    print("Stage 3: Generating Story Arc")
    print("=" * 60)

    with Timer() as t:
        arc_synapse = create_story_arc_synapse(blueprint, characters, "一个关于AI的故事")
        arc_synapse = await mock_generate(arc_synapse)

    is_valid, arc = validate_json(arc_synapse.output_json)
    print(f"✅ Story Arc generated in {t.elapsed:.2f}s")
    print(f"   Valid JSON: {is_valid}")

    # Stage 4: Chapters
    print("\n" + "=" * 60)
    print("Stage 4: Generating Chapters")
    print("=" * 60)

    with Timer() as t:
        chapters_synapse = create_chapters_synapse(
            blueprint, characters, arc,
            chapter_ids=["chapter_1", "chapter_2"],
            user_input="一个关于AI的故事"
        )
        chapters_synapse = await mock_generate(chapters_synapse)

    is_valid, chapters = validate_json(chapters_synapse.output_json)
    print(f"✅ Chapters generated in {t.elapsed:.2f}s")
    print(f"   Valid JSON: {is_valid}")

    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
```

**运行输出**：
```
============================================================
Stage 1: Generating Blueprint
============================================================
✅ Blueprint generated in 1.00s
   Valid JSON: True
   Title: 测试故事

============================================================
Stage 2: Generating Characters
============================================================
✅ Characters generated in 1.00s
   Valid JSON: True
   Character count: 5

============================================================
Stage 3: Generating Story Arc
============================================================
✅ Story Arc generated in 1.00s
   Valid JSON: True

============================================================
Stage 4: Generating Chapters
============================================================
✅ Chapters generated in 1.00s
   Valid JSON: True

============================================================
✅ Pipeline complete!
============================================================
```

## 5. 字段验证示例

```python
from template.protocol import StoryGenerationSynapse
from template.utils import validate_required_fields

# 创建 Synapse
synapse = StoryGenerationSynapse(
    task_type="blueprint",
    input_data={"user_input": "测试"}
)

# 获取必需字段
required_fields = synapse.get_required_fields()
print(required_fields)
# ['title', 'genre', 'setting', 'core_conflict', 'themes', 'tone', 'target_audience']

# 验证输出
output = {
    "title": "测试",
    "genre": "科幻"
    # 缺少其他字段
}

is_valid, missing = validate_required_fields(output, required_fields)
print(f"Valid: {is_valid}")
print(f"Missing: {missing}")
# Valid: False
# Missing: ['setting', 'core_conflict', 'themes', 'tone', 'target_audience']
```

## 6. 错误处理

```python
from template.protocol import StoryGenerationSynapse
from pydantic import ValidationError

# 错误的 task_type
try:
    synapse = StoryGenerationSynapse(
        task_type="invalid_task",  # ❌ 无效
        input_data={}
    )
except ValidationError as e:
    print(f"错误: {e}")
    # 错误: task_type must be one of ['blueprint', 'characters', 'story_arc', 'chapters']

# 缺少必需的 input_data 字段
try:
    synapse = StoryGenerationSynapse(
        task_type="characters",
        input_data={"user_input": "测试"}  # ❌ 缺少 blueprint
    )
except ValidationError as e:
    print(f"错误: {e}")
    # 错误: characters task requires 'blueprint' field
```

## 7. 下一步

Protocol 定义完成后，可以开始实现：

1. **Miner** (`neurons/miner.py`)：
   - 实现 `forward()` 函数处理 Synapse
   - 调用 OpenAI/Claude API 生成内容
   - 返回符合格式的 JSON

2. **Validator** (`neurons/validator.py`)：
   - 创建不同类型的 Synapse
   - 发送给 Miners
   - 评分并分配权重

完整代码结构参考：[STORYFI_BITTENSOR_SUBNET_PROJECT_PLAN.md](./STORYFI_BITTENSOR_SUBNET_PROJECT_PLAN.md)
