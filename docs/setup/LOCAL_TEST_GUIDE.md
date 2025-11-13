# StoryFi 本地测试完整指南

## 🎯 目标

在本地环境测试 StoryFi 子网的所有功能：
1. ✅ Protocol 通信
2. ✅ Miner AI 生成
3. ✅ Validator 评分系统
4. ✅ 权重计算
5. ✅ 反作弊机制

**不需要**：真实的子网、TAO、钱包

---

## 📋 前置要求

### 1. 系统要求
- macOS / Linux / Windows
- Python 3.9+
- 4GB+ RAM
- 稳定的网络（用于 OpenAI API）

### 2. 需要准备的
- ✅ OpenAI API Key
- ✅ 终端/命令行工具
- ✅ 代码编辑器（可选，用于查看日志）

---

## 🚀 快速开始（5 步）

### Step 1: 环境准备（2 分钟）

```bash
# 进入项目目录
cd /Users/xinyueyu/storyfi/storyfi-subnet

# 检查 Python 版本
python3 --version
# 应该显示 Python 3.9 或更高

# 安装依赖
pip3 install -r requirements.txt

# 如果遇到权限问题，使用：
pip3 install --user -r requirements.txt
```

**预期输出**：
```
Successfully installed bittensor-6.x.x pydantic-2.x.x ...
```

---

### Step 2: 配置环境变量（1 分钟）

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置
nano .env
```

**填入你的 OpenAI API Key**：
```bash
# 必填
OPENAI_API_KEY=sk-your-openai-api-key-here

# 可选（本地测试用）
OPENAI_MODEL=gpt-4-turbo-preview
MAX_TOKENS=3000
TEMPERATURE=0.7

# 本地测试配置
NETUID=108
WALLET_NAME=test_miner
WALLET_HOTKEY=default
```

保存并退出（`Ctrl+X`, `Y`, `Enter`）

---

### Step 3: 测试 Protocol（2 分钟）

创建测试脚本：

```bash
# 创建测试文件
cat > test_protocol.py << 'EOF'
"""测试 Protocol 通信协议"""
import json
from template.protocol import (
    create_blueprint_synapse,
    create_characters_synapse,
    create_story_arc_synapse
)
from template.utils import validate_json, Timer

print("=" * 60)
print("测试 1: Blueprint Synapse")
print("=" * 60)

# 创建 Blueprint 任务
synapse = create_blueprint_synapse("一个关于太空探险的故事")
print(f"✅ Task type: {synapse.task_type}")
print(f"✅ Input data: {synapse.input_data}")
print(f"✅ Required fields: {synapse.get_required_fields()}")

# 模拟 Miner 响应
synapse.output_json = json.dumps({
    "title": "星际迷航",
    "genre": "科幻",
    "setting": "2234年，人类已经殖民火星",
    "core_conflict": "发现外星文明信号，但无法破译",
    "themes": ["探索", "交流", "未知"],
    "tone": "神秘而充满希望",
    "target_audience": "科幻爱好者"
}, ensure_ascii=False)
synapse.generation_time = 2.5

# 验证
is_valid, data = validate_json(synapse.output_json)
print(f"✅ JSON valid: {is_valid}")
print(f"✅ Output preview: {data['title']}")
print()

print("=" * 60)
print("测试 2: Characters Synapse")
print("=" * 60)

# 创建 Characters 任务
blueprint = data
synapse2 = create_characters_synapse(blueprint, "一个关于太空探险的故事")
print(f"✅ Task type: {synapse2.task_type}")
print(f"✅ Has blueprint: {'blueprint' in synapse2.input_data}")
print()

print("=" * 60)
print("✅ Protocol 测试通过！")
print("=" * 60)
EOF

# 运行测试
python3 test_protocol.py
```

**预期输出**：
```
============================================================
测试 1: Blueprint Synapse
============================================================
✅ Task type: blueprint
✅ Input data: {'user_input': '一个关于太空探险的故事', 'max_tokens': 1000}
✅ Required fields: ['title', 'genre', 'setting', ...]
✅ JSON valid: True
✅ Output preview: 星际迷航

============================================================
测试 2: Characters Synapse
============================================================
✅ Task type: characters
✅ Has blueprint: True

============================================================
✅ Protocol 测试通过！
============================================================
```

---

### Step 4: 测试评分系统（3 分钟）

```bash
cat > test_scoring.py << 'EOF'
"""测试评分系统"""
import json
from scoring import (
    calculate_technical_score,
    calculate_structure_score,
    calculate_content_score
)

print("=" * 60)
print("测试评分系统")
print("=" * 60)

# 测试数据
response_json = json.dumps({
    "title": "星际迷航",
    "genre": "科幻",
    "setting": "2234年，人类已经殖民火星，发现了古老的外星遗迹",
    "core_conflict": "破译外星文明信号，但发现他们可能是敌对的",
    "themes": ["探索", "信任", "生存"],
    "tone": "紧张而神秘",
    "target_audience": "成人科幻读者"
}, ensure_ascii=False)

generation_time = 2.5
task_type = "blueprint"
required_fields = ["title", "genre", "setting", "core_conflict", "themes", "tone", "target_audience"]

# 1. Technical Score
tech_score, tech_breakdown = calculate_technical_score(
    response_json,
    generation_time,
    task_type,
    required_fields
)
print(f"\n📊 Technical Score: {tech_score:.1f}/30")
print(f"   - JSON valid: {tech_breakdown['json_valid']:.1f}")
print(f"   - Schema complete: {tech_breakdown['schema_complete']:.1f}")
print(f"   - Response time: {tech_breakdown['response_time']:.1f}")

# 2. Structure Score
data = json.loads(response_json)
struct_score, struct_breakdown = calculate_structure_score(data, task_type)
print(f"\n📊 Structure Score: {struct_score:.1f}/40")
print(f"   - Field completeness: {struct_breakdown['field_completeness']:.1f}")
print(f"   - Content length: {struct_breakdown['content_length']:.1f}")
print(f"   - Themes count: {struct_breakdown['themes_count']:.1f}")

# 3. Content Score
context = {"user_input": "一个关于太空探险的故事"}
content_score, content_breakdown = calculate_content_score(
    data, context, task_type, history=[], use_embeddings=False
)
print(f"\n📊 Content Score: {content_score:.1f}/30")
print(f"   - Relevance: {content_breakdown['relevance']:.1f}")
print(f"   - Fluency: {content_breakdown['fluency']:.1f}")
print(f"   - Originality: {content_breakdown['originality']:.1f}")

# Total
total = tech_score + struct_score + content_score
print(f"\n{'=' * 60}")
print(f"🎯 Total Score: {total:.1f}/100")
print(f"{'=' * 60}")

if total >= 70:
    print("✅ 高质量响应！")
elif total >= 50:
    print("⚠️ 中等质量")
else:
    print("❌ 低质量，需要改进")
EOF

python3 test_scoring.py
```

**预期输出**：
```
============================================================
测试评分系统
============================================================

📊 Technical Score: 30.0/30
   - JSON valid: 10.0
   - Schema complete: 10.0
   - Response time: 10.0

📊 Structure Score: 40.0/40
   - Field completeness: 20.0
   - Content length: 10.0
   - Themes count: 10.0

📊 Content Score: 20.3/30
   - Relevance: 9.0
   - Fluency: 6.3
   - Originality: 5.0

============================================================
🎯 Total Score: 90.3/100
============================================================
✅ 高质量响应！
```

---

### Step 5: 测试 Miner 生成（5 分钟）

这个测试需要调用 OpenAI API，会产生少量费用（约 $0.05）。

```bash
cat > test_miner_generation.py << 'EOF'
"""测试 Miner AI 生成功能"""
import asyncio
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

async def test_generation():
    # 初始化 OpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=" * 60)
    print("测试 Miner AI 生成 - Blueprint")
    print("=" * 60)

    user_input = "一个关于赛博朋克黑客的故事"

    prompt = f"""你是故事设计师。根据用户输入创建故事蓝图。

用户输入: {user_input}

生成JSON，包含：title, genre, setting, core_conflict, themes, tone, target_audience

直接输出JSON，不要markdown格式。"""

    print(f"📝 用户输入: {user_input}")
    print(f"⏳ 调用 OpenAI API...")

    response = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "你是故事设计师。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )

    content = response.choices[0].message.content.strip()

    # 解析JSON
    if content.startswith("```json"):
        content = content.split("```json")[1].split("```")[0].strip()
    elif content.startswith("```"):
        content = content.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(content)
        print(f"\n✅ 生成成功！")
        print(f"\n📖 生成的故事蓝图：")
        print(json.dumps(data, ensure_ascii=False, indent=2))

        # 验证字段
        required = ["title", "genre", "setting", "core_conflict", "themes", "tone", "target_audience"]
        missing = [f for f in required if f not in data]

        if not missing:
            print(f"\n✅ 所有必需字段都存在")
        else:
            print(f"\n⚠️ 缺少字段: {missing}")

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        print(f"原始内容:\n{content}")

if __name__ == "__main__":
    asyncio.run(test_generation())
EOF

python3 test_miner_generation.py
```

**预期输出**：
```
============================================================
测试 Miner AI 生成 - Blueprint
============================================================
📝 用户输入: 一个关于赛博朋克黑客的故事
⏳ 调用 OpenAI API...

✅ 生成成功！

📖 生成的故事蓝图：
{
  "title": "数字幽灵",
  "genre": "赛博朋克",
  "setting": "2084年新东京，霓虹灯下的数字迷宫",
  "core_conflict": "一个黑客发现跨国公司的阴谋，试图揭露真相",
  "themes": ["自由与控制", "虚拟与现实", "人性与科技"],
  "tone": "黑暗、紧张、反乌托邦",
  "target_audience": "成人科幻读者"
}

✅ 所有必需字段都存在
```

---

## 🎉 如果所有测试都通过

恭喜！你的代码完全正常工作！

**你已经验证了**：
- ✅ Protocol 通信协议正常
- ✅ 评分系统计算正确
- ✅ Miner 能够生成高质量内容
- ✅ OpenAI API 集成成功

---

## 🐛 常见问题

### 问题 1: ModuleNotFoundError

```bash
ModuleNotFoundError: No module named 'bittensor'
```

**解决**：
```bash
pip3 install bittensor pydantic openai python-dotenv
```

### 问题 2: OpenAI API Key 无效

```
Error: Invalid API key
```

**解决**：
1. 检查 `.env` 文件中的 API Key
2. 确保没有多余的空格或引号
3. 访问 https://platform.openai.com/api-keys 创建新的 Key

### 问题 3: JSON 解析失败

```
json.JSONDecodeError: Expecting value
```

**解决**：
- OpenAI 有时返回包含 markdown 的内容
- 代码已经包含了自动清理逻辑
- 如果仍然失败，检查 prompt 是否明确要求 JSON 格式

### 问题 4: Import 错误

```
ImportError: cannot import name 'calculate_technical_score'
```

**解决**：
```bash
# 确保在正确的目录
cd /Users/xinyueyu/storyfi/storyfi-subnet

# 检查文件是否存在
ls -la scoring/

# 确保 __init__.py 存在
touch scoring/__init__.py
```

---

## 📊 测试清单

完成以下测试后打勾：

### 基础测试
- [ ] Python 环境检查
- [ ] 依赖安装成功
- [ ] .env 配置完成
- [ ] OpenAI API Key 有效

### Protocol 测试
- [ ] Blueprint Synapse 创建成功
- [ ] Characters Synapse 创建成功
- [ ] Story Arc Synapse 创建成功
- [ ] Chapters Synapse 创建成功
- [ ] JSON 验证通过

### 评分系统测试
- [ ] Technical Score 计算正确
- [ ] Structure Score 计算正确
- [ ] Content Score 计算正确
- [ ] Total Score 在合理范围（50-100）

### AI 生成测试
- [ ] Blueprint 生成成功
- [ ] 生成内容为有效 JSON
- [ ] 所有必需字段存在
- [ ] 内容质量合格

---

## 🚀 下一步：完整集成测试

所有单元测试通过后，我们可以进行完整的 Miner-Validator 集成测试。

这需要：
1. 启动模拟的 Miner（不需要真实网络）
2. 启动模拟的 Validator
3. 测试完整的请求-响应-评分流程

**准备好了吗？** 完成上面的测试后告诉我，我会指导你进行集成测试！

---

## 📝 测试日志

记录你的测试结果：

```
测试日期：___________
Python 版本：___________
OpenAI 模型：___________

测试结果：
- Protocol: [ ] 通过 [ ] 失败
- Scoring: [ ] 通过 [ ] 失败
- Generation: [ ] 通过 [ ] 失败

遇到的问题：
___________
___________

解决方案：
___________
___________
```

---

**现在就开始第一个测试吧！** 🎯

从 Step 1 开始，一步步执行，遇到任何问题随时告诉我！
