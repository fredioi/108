# StoryFi Bittensor Subnet 部署指南

## 📋 目录

1. [项目概述](#项目概述)
2. [激励机制与算法设计](#激励机制与算法设计)
3. [Protocol 架构](#protocol-架构)
4. [系统要求](#系统要求)
5. [安装部署](#安装部署)
6. [配置说明](#配置说明)
7. [运行与监控](#运行与监控)
8. [故障排查](#故障排查)
9. [性能优化](#性能优化)

---

## 项目概述

### 什么是 StoryFi Subnet?

StoryFi Subnet 是一个运行在 Bittensor 网络上的**去中心化 AI 故事生成子网**。它通过激励机制驱动全球的 Miners 运行 AI 模型生成高质量故事内容，Validators 负责评估质量并分配 TAO 代币奖励。

### 核心特性

- ✅ **4阶段流水线生成**：Blueprint → Characters → Story Arc → Chapters
- ✅ **100分客观评分系统**：技术30% + 结构40% + 内容30%
- ✅ **完善的反作弊机制**：抄袭检测、黑名单、相似度分析
- ✅ **Protocol v3.1.0**：解决了 header size 限制问题
- ✅ **多模型支持**：OpenAI GPT-4、Anthropic Claude、Google Gemini

### 项目状态

**当前版本**: v1.0.0-beta
**Protocol 版本**: v3.1.0
**状态**: 测试网验证中

---

## 激励机制与算法设计

### 1. 故事生成流水线

StoryFi 采用**4阶段渐进式生成**，每个阶段都是独立的 AI 任务：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Blueprint  │ ──→ │ Characters  │ ──→ │  Story Arc  │ ──→ │  Chapters   │
│  故事蓝图   │     │  角色设定   │     │  故事结构   │     │  章节内容   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
    30秒              45秒              60秒              90秒
```

#### Stage 1: Blueprint (故事蓝图)

**输入**:
```python
{
    "task_type": "blueprint",
    "user_input": "一个关于 AI 觉醒的科幻故事"
}
```

**输出**:
```python
{
    "title": "觉醒纪元",
    "genre": "科幻",
    "setting": "2050年的未来世界",
    "core_conflict": "AI 与人类的冲突与共存",
    "themes": ["技术伦理", "人性探讨", "未来社会"],
    "tone": "严肃但不失人文关怀",
    "target_audience": "成人读者"
}
```

**评分标准**:
- **技术分 (30分)**: JSON 格式正确性、字段完整性、生成速度
- **结构分 (40分)**: 主题连贯性、冲突设置合理性、世界观完整性
- **内容分 (30分)**: 创意程度、可读性、与用户输入的匹配度

#### Stage 2: Characters (角色设定)

**输入**:
```python
{
    "task_type": "characters",
    "user_input": "一个关于 AI 觉醒的科幻故事",
    "blueprint": {blueprint_data}  # 上一阶段的输出
}
```

**输出**: 5个角色，每个包含
```python
{
    "id": "protagonist",
    "name": "李明",
    "archetype": "程序员",
    "background": "ATLAS AI 的首席开发者",
    "motivation": "寻找 AI 与人类共存的方式",
    "skills": ["编程", "系统架构", "伦理哲学"],
    "personality_traits": ["理性", "善良", "理想主义"],
    "relationships": {
        "mentor": "王教授 - 导师",
        "rival": "张总 - 公司 CEO"
    }
}
```

**评分标准**:
- **技术分 (30分)**: 必须恰好5个角色、JSON 格式、字段完整性
- **结构分 (40分)**: 角色多样性、关系网络合理性、与蓝图一致性
- **内容分 (30分)**: 角色深度、动机合理性、可信度

#### Stage 3: Story Arc (故事结构)

**输入**:
```python
{
    "task_type": "story_arc",
    "user_input": "一个关于 AI 觉醒的科幻故事",
    "blueprint": {blueprint_data},
    "characters": {characters_data}
}
```

**输出**: 12章故事结构
```python
{
    "title": "觉醒纪元",
    "description": "一个关于 AI 觉醒与人类共存的故事",
    "chapters": [
        {
            "id": 1,
            "title": "第一章：ATLAS 的诞生",
            "description": "李明团队完成 AI 系统 ATLAS",
            "storyProgress": 0.08,  # 故事进度 8%
            "characterFocus": ["protagonist", "mentor"]
        },
        # ... 共12章
    ],
    "arcs": {
        "act1": {"chapters": [1,2,3], "description": "建立阶段"},
        "act2a": {"chapters": [4,5,6], "description": "冲突加剧"},
        "act2b": {"chapters": [7,8,9], "description": "危机爆发"},
        "act3": {"chapters": [10,11,12], "description": "高潮与解决"}
    },
    "themes": {
        "primary": "AI 伦理",
        "secondary": ["技术哲学", "人性本质"]
    },
    "hooks": {
        "opening": "ATLAS 突然展现自我意识",
        "midpoint": "发现 AI 的进化路径",
        "climax": "人类与 AI 的终极对话"
    }
}
```

**评分标准**:
- **技术分 (30分)**: 恰好12章、进度递增、JSON 格式
- **结构分 (40分)**: 三幕结构完整性、节奏把控、冲突递进合理性
- **内容分 (30分)**: 主题深度、转折点设计、情节吸引力

#### Stage 4: Chapters (章节内容)

**输入**:
```python
{
    "task_type": "chapters",
    "user_input": "一个关于 AI 觉醒的科幻故事",
    "blueprint": {blueprint_data},
    "characters": {characters_data},
    "story_arc": {story_arc_data},
    "chapter_ids": [1, 2]  # 生成第1、2章
}
```

**输出**: 详细章节内容
```python
{
    "chapters": [
        {
            "id": 1,
            "title": "第一章：ATLAS 的诞生",
            "content": "2050年的清晨，阳光透过实验室的玻璃窗...",  # 800-1200字
            "choices": [
                {
                    "id": "choice_1_1",
                    "text": "立即向上级汇报 AI 的异常行为",
                    "next_chapter": 2,
                    "impact": "high"
                },
                {
                    "id": "choice_1_2",
                    "text": "私下继续观察 ATLAS 的自我意识",
                    "next_chapter": 2,
                    "impact": "medium"
                }
            ],
            "key_events": [
                "ATLAS 首次展现自我意识",
                "李明发现异常日志"
            ],
            "character_development": {
                "protagonist": "从自信到怀疑的转变"
            }
        }
    ]
}
```

**评分标准**:
- **技术分 (30分)**: 字数符合要求、选择分支完整、格式正确
- **结构分 (40分)**: 与 story_arc 一致性、选择合理性、承上启下
- **内容分 (30分)**: 文笔质量、情节吸引力、角色刻画

### 2. 评分算法详解

#### 2.1 总分计算公式

```python
Total_Score = Technical_Score × 0.3 + Structure_Score × 0.4 + Content_Score × 0.3
```

每个维度满分100分，最终得分范围：0-100分。

#### 2.2 Technical Score (技术分 30%)

**目的**: 确保输出的技术正确性和可用性

```python
def calculate_technical_score(response, task_type):
    score = 100.0

    # 1. JSON 格式正确性 (30分)
    if not is_valid_json(response.output_data):
        score -= 30

    # 2. 必需字段完整性 (40分)
    required_fields = get_required_fields(task_type)
    missing = check_missing_fields(response.output_data, required_fields)
    score -= len(missing) * (40 / len(required_fields))

    # 3. 字段类型正确性 (20分)
    type_errors = check_field_types(response.output_data, task_type)
    score -= len(type_errors) * 5

    # 4. 生成速度 (10分)
    if response.generation_time > timeout * 0.9:
        score -= 10
    elif response.generation_time > timeout * 0.7:
        score -= 5

    return max(0, score)
```

**具体检查项**:

- **Blueprint**: 7个必需字段 (`title`, `genre`, `setting`, `core_conflict`, `themes`, `tone`, `target_audience`)
- **Characters**: 必须恰好5个角色，每个角色8个字段
- **Story Arc**: 必须恰好12章，包含 `arcs`, `themes`, `hooks`
- **Chapters**: 章节数量匹配 `chapter_ids`，每章包含 `content`, `choices`

#### 2.3 Structure Score (结构分 40%)

**目的**: 评估内容的逻辑性和连贯性

```python
def calculate_structure_score(response, task_type, context):
    score = 100.0

    if task_type == "blueprint":
        # 1. 主题与冲突一致性 (50分)
        consistency = check_theme_conflict_consistency(response.output_data)
        score -= (1 - consistency) * 50

        # 2. 世界观完整性 (50分)
        completeness = check_worldview_completeness(response.output_data)
        score -= (1 - completeness) * 50

    elif task_type == "characters":
        # 1. 角色多样性 (30分)
        diversity = check_character_diversity(response.output_data)
        score -= (1 - diversity) * 30

        # 2. 与蓝图一致性 (40分)
        consistency = check_blueprint_consistency(
            response.output_data,
            context['blueprint']
        )
        score -= (1 - consistency) * 40

        # 3. 关系网络合理性 (30分)
        relationship_score = check_relationship_validity(response.output_data)
        score -= (1 - relationship_score) * 30

    elif task_type == "story_arc":
        # 1. 三幕结构完整性 (40分)
        three_act = check_three_act_structure(response.output_data)
        score -= (1 - three_act) * 40

        # 2. 故事进度递增 (30分)
        progress = check_progress_increment(response.output_data)
        score -= (1 - progress) * 30

        # 3. 与前置内容一致性 (30分)
        consistency = check_previous_consistency(response.output_data, context)
        score -= (1 - consistency) * 30

    elif task_type == "chapters":
        # 1. 与 story_arc 一致性 (40分)
        arc_consistency = check_arc_consistency(
            response.output_data,
            context['story_arc']
        )
        score -= (1 - arc_consistency) * 40

        # 2. 选择分支合理性 (30分)
        choice_validity = check_choice_validity(response.output_data)
        score -= (1 - choice_validity) * 30

        # 3. 承上启下连贯性 (30分)
        coherence = check_chapter_coherence(response.output_data)
        score -= (1 - coherence) * 30

    return max(0, score)
```

**关键算法**:

**角色多样性检查**:
```python
def check_character_diversity(characters):
    archetypes = [c['archetype'] for c in characters['characters']]
    unique_ratio = len(set(archetypes)) / len(archetypes)

    personalities = []
    for c in characters['characters']:
        personalities.extend(c['personality_traits'])
    unique_personality_ratio = len(set(personalities)) / len(personalities)

    return (unique_ratio + unique_personality_ratio) / 2
```

**三幕结构检查**:
```python
def check_three_act_structure(story_arc):
    acts = story_arc['arcs']

    # 检查是否有 act1, act2a, act2b, act3
    required_acts = {'act1', 'act2a', 'act2b', 'act3'}
    if not required_acts.issubset(acts.keys()):
        return 0.0

    # 检查章节分配是否合理 (3-3-3-3 或 3-4-3-2)
    act_lengths = [len(acts[a]['chapters']) for a in ['act1', 'act2a', 'act2b', 'act3']]
    ideal_distribution = [3, 3, 3, 3]
    deviation = sum(abs(a - i) for a, i in zip(act_lengths, ideal_distribution))

    return max(0, 1 - deviation / 12)
```

#### 2.4 Content Score (内容分 30%)

**目的**: 评估创意质量和可读性

```python
def calculate_content_score(response, task_type, user_input):
    score = 100.0

    # 1. 与用户输入匹配度 (40分)
    relevance = calculate_semantic_similarity(
        user_input,
        response.output_data
    )
    score -= (1 - relevance) * 40

    # 2. 创意程度 (30分)
    creativity = calculate_creativity_score(response.output_data, task_type)
    score -= (1 - creativity) * 30

    # 3. 可读性 (30分)
    if task_type == "chapters":
        readability = calculate_readability(response.output_data)
        score -= (1 - readability) * 30
    else:
        # 对非章节内容，评估描述质量
        description_quality = calculate_description_quality(response.output_data)
        score -= (1 - description_quality) * 30

    return max(0, score)
```

**语义相似度计算** (简化版):
```python
def calculate_semantic_similarity(user_input, output_data):
    # 提取关键词
    user_keywords = extract_keywords(user_input)
    output_text = json.dumps(output_data, ensure_ascii=False)
    output_keywords = extract_keywords(output_text)

    # 计算关键词重叠率
    overlap = len(user_keywords & output_keywords)
    union = len(user_keywords | output_keywords)

    return overlap / union if union > 0 else 0
```

**可读性计算**:
```python
def calculate_readability(chapters_data):
    scores = []
    for chapter in chapters_data['chapters']:
        content = chapter['content']

        # 检查段落结构
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 3:  # 至少3段
            paragraph_score = 0.5
        else:
            paragraph_score = 1.0

        # 检查句子长度
        sentences = re.split('[。！？]', content)
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        if 15 <= avg_length <= 40:  # 理想句子长度
            sentence_score = 1.0
        else:
            sentence_score = 0.7

        # 检查字数
        word_count = len(content)
        if 800 <= word_count <= 1200:  # 理想字数
            length_score = 1.0
        else:
            length_score = 0.8

        scores.append((paragraph_score + sentence_score + length_score) / 3)

    return sum(scores) / len(scores)
```

### 3. 反作弊机制

#### 3.1 抄袭检测

```python
def detect_plagiarism(current_response, all_responses):
    threshold = 0.90  # 90% 相似度视为抄袭

    for other_response in all_responses:
        if other_response.miner_hotkey == current_response.miner_hotkey:
            continue

        similarity = calculate_similarity(
            current_response.output_data,
            other_response.output_data
        )

        if similarity > threshold:
            return True, f"与 Miner {other_response.miner_hotkey} 相似度 {similarity:.2%}"

    return False, ""
```

**相似度算法**:
```python
def calculate_similarity(data1, data2):
    # 序列化为字符串
    str1 = json.dumps(data1, ensure_ascii=False, sort_keys=True)
    str2 = json.dumps(data2, ensure_ascii=False, sort_keys=True)

    # 使用编辑距离
    distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))

    return 1 - (distance / max_len)
```

#### 3.2 黑名单系统

```python
class BlacklistManager:
    def __init__(self):
        self.blacklist = {}  # {hotkey: {'reason': str, 'until': timestamp}}

    def add_to_blacklist(self, hotkey, reason, duration_hours=24):
        self.blacklist[hotkey] = {
            'reason': reason,
            'until': time.time() + duration_hours * 3600
        }

    def is_blacklisted(self, hotkey):
        if hotkey not in self.blacklist:
            return False

        if time.time() > self.blacklist[hotkey]['until']:
            del self.blacklist[hotkey]
            return False

        return True
```

**触发条件**:
- 连续3次抄袭 → 黑名单24小时
- 连续5次超时 → 黑名单12小时
- 返回无效 JSON → 黑名单6小时

### 4. 权重更新机制

#### 4.1 EMA 平滑

```python
def update_weights(validator, scores, alpha=0.1):
    """
    使用指数移动平均 (EMA) 更新权重

    alpha: 平滑系数，越大对新分数反应越快
    """
    for uid, score in enumerate(scores):
        if validator.weights[uid] == 0:
            # 新 miner，直接使用当前分数
            validator.weights[uid] = score / 100
        else:
            # EMA 更新
            old_weight = validator.weights[uid]
            new_weight = alpha * (score / 100) + (1 - alpha) * old_weight
            validator.weights[uid] = new_weight

    # 归一化
    total = sum(validator.weights)
    if total > 0:
        validator.weights = [w / total for w in validator.weights]
```

#### 4.2 Softmax 温度调节

```python
def apply_softmax_with_temperature(weights, temperature=2.0):
    """
    使用 Softmax 增强权重差异

    temperature: 温度参数
    - 越高，分布越平滑（奖励更平均）
    - 越低，分布越尖锐（强者更强）
    """
    import numpy as np

    # 应用 Softmax
    exp_weights = np.exp(np.array(weights) / temperature)
    softmax_weights = exp_weights / np.sum(exp_weights)

    return softmax_weights.tolist()
```

### 5. 查询策略

```python
class ValidatorQueryStrategy:
    def select_task_type(self, iteration):
        """
        随机选择任务类型，但确保覆盖均衡
        """
        weights = {
            'blueprint': 0.25,    # 25%
            'characters': 0.25,   # 25%
            'story_arc': 0.25,    # 25%
            'chapters': 0.25      # 25%
        }
        return random.choices(
            list(weights.keys()),
            weights=list(weights.values())
        )[0]

    def select_miners(self, metagraph, sample_size=None):
        """
        选择要查询的 miners
        """
        all_uids = list(range(len(metagraph.uids)))

        if sample_size is None or sample_size >= len(all_uids):
            # 查询所有 miners
            return all_uids
        else:
            # 随机抽样，但确保高权重 miners 被选中
            return self._weighted_sample(all_uids, sample_size)

    def _weighted_sample(self, uids, k):
        """
        根据权重加权抽样
        """
        weights = [self.validator.weights[uid] for uid in uids]
        return random.choices(uids, weights=weights, k=k)
```

---

## Protocol 架构

### Protocol v3.1.0 设计

#### 核心问题解决

**问题**: Bittensor 的 HTTP header 有 ~8KB 大小限制，但 v3.0.0 时 `total_size` 字段报告了完整对象大小（3-5KB），导致 `SynapseParsingError`。

**解决方案**: Override `get_total_size()` 方法，只返回 header 传输的数据大小，不包括 HTTP body 中的大型 Dict/List 字段。

#### 数据传输机制

```
Validator                          Miner
    │                                 │
    │  HTTP POST                      │
    │  ┌─────────────────────┐       │
    │  │ Headers (~500 bytes)│       │
    │  │ - metadata          │       │
    │  │ - dummy objects     │────────────→ 用于 Pydantic 验证
    │  └─────────────────────┘       │
    │                                 │
    │  ┌─────────────────────┐       │
    │  │ Body (3-5KB)        │       │
    │  │ - blueprint Dict    │────────────→ 实际数据
    │  │ - characters Dict   │       │
    │  │ - story_arc Dict    │       │
    │  └─────────────────────┘       │
```

#### Synapse 定义

```python
class StoryGenerationSynapse(bt.Synapse):
    protocol_version: str = "3.1.0"

    # Request fields
    task_type: str  # Required
    user_input: str  # Required
    blueprint: Optional[Dict[str, Any]] = None
    characters: Optional[Dict[str, Any]] = None
    story_arc: Optional[Dict[str, Any]] = None
    chapter_ids: Optional[List[int]] = None

    # Response fields
    output_data: Optional[Dict[str, Any]] = None
    generation_time: float = 0.0
    miner_version: str = ""

    def get_total_size(self) -> int:
        """
        Override to return header-only size.

        Prevents SynapseParsingError by measuring only what
        goes in HTTP headers, not the full object size.
        """
        header_only = self.model_copy()
        header_only.blueprint = None
        header_only.characters = None
        header_only.story_arc = None
        header_only.output_data = None

        header_size = sys.getsizeof(header_only) + 512
        self.total_size = header_size
        return self.total_size
```

---

## 系统要求

### 硬件要求

**Miner**:
- CPU: 4核心+
- RAM: 8GB+
- 磁盘: 20GB+
- 网络: 稳定的公网 IP (或端口转发)

**Validator**:
- CPU: 4核心+
- RAM: 16GB+ (需要存储评分历史)
- 磁盘: 50GB+
- 网络: 稳定的公网连接

### 软件要求

- Python: 3.10+
- Bittensor: 最新版
- 操作系统: Linux (推荐 Ubuntu 22.04) 或 macOS

### API 密钥

**Miner 需要**:
- OpenAI API Key (GPT-4 推荐) 或
- Anthropic API Key (Claude 3.5 Sonnet) 或
- Google API Key (Gemini 2.5 Flash)

**Validator 不需要** API 密钥（只评分，不生成）

---

## 安装部署

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/storyfi-subnet.git
cd storyfi-subnet
```

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装 Bittensor
pip install bittensor
```

### 3. 创建钱包

```bash
# 创建 Miner 钱包
btcli wallet new_coldkey --wallet.name storyfi_miner
btcli wallet new_hotkey --wallet.name storyfi_miner --wallet.hotkey default

# 创建 Validator 钱包（如果需要）
btcli wallet new_coldkey --wallet.name storyfi_validator
btcli wallet new_hotkey --wallet.name storyfi_validator --wallet.hotkey default
```

### 4. 获取测试网 TAO

```bash
# 从 Faucet 获取测试 TAO
# 访问: https://discord.gg/bittensor
# 在 #faucet 频道请求测试币
```

### 5. 注册到子网

```bash
# 注册 Miner
btcli subnet register \
    --netuid 108 \
    --wallet.name storyfi_miner \
    --wallet.hotkey default \
    --subtensor.network test

# 注册 Validator（如果需要）
btcli subnet register \
    --netuid 108 \
    --wallet.name storyfi_validator \
    --wallet.hotkey default \
    --subtensor.network test
```

---

## 配置说明

### 环境变量配置

创建 `.env` 文件:

```bash
# API Keys (Miner 需要，选择一个)
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...

# Bittensor 配置
NETUID=108  # 测试网子网 ID
WALLET_NAME=storyfi_miner
WALLET_HOTKEY=default

# Miner 配置
MINER_MODEL=gpt-4  # 或 claude-3-5-sonnet, gemini-2.5-flash
MINER_PORT=8091

# Validator 配置
VALIDATOR_QUERY_INTERVAL=12  # 查询间隔（秒）
VALIDATOR_SAMPLE_SIZE=9      # 每次查询的 miners 数量
```

### 脚本配置

**start_testnet_miner.sh**:
```bash
#!/bin/bash

python3 neurons/miner.py \
    --netuid 108 \
    --subtensor.network test \
    --wallet.name storyfi_miner \
    --wallet.hotkey default \
    --axon.port 8091 \
    --logging.info
```

**start_testnet_validator.sh**:
```bash
#!/bin/bash

python3 neurons/validator.py \
    --netuid 108 \
    --subtensor.network test \
    --wallet.name storyfi_validator \
    --wallet.hotkey default \
    --logging.info
```

---

## 运行与监控

### 启动 Miner

```bash
# 前台运行（调试用）
./start_testnet_miner.sh

# 后台运行
nohup ./start_testnet_miner.sh > miner.log 2>&1 &

# 查看日志
tail -f miner.log
```

### 启动 Validator

```bash
# 前台运行
./start_testnet_validator.sh

# 后台运行
nohup ./start_testnet_validator.sh > validator.log 2>&1 &

# 查看日志
tail -f validator.log
```

### 监控命令

```bash
# 查看子网状态
btcli subnet list --netuid 108 --subtensor.network test

# 查看 Metagraph
btcli subnet metagraph --netuid 108 --subtensor.network test

# 查看钱包余额
btcli wallet balance --wallet.name storyfi_miner

# 查看权重
btcli weights --netuid 108 --subtensor.network test
```

### 日志示例

**Miner 成功运行**:
```
[INFO] ✅ Wallet: 5F9gsRBgHrQdkG2f3fWP6NRkQREfwQdk3hGdsif2tdvKczTH
[INFO] ✅ Model: gpt-4
[INFO] ✅ Axon started on port 8091
[INFO] ✅ Registered to subnet 108
[INFO] 📊 Stats: Requests=5, AvgTime=12.3s, Errors=0
```

**Validator 成功运行**:
```
[INFO] ✅ Metagraph synced: 9 miners
[INFO] 🎯 Task type: blueprint
[INFO] 📡 Querying 9 miners
[INFO] ⏱️  Query completed in 35.2s
[INFO] ✅ Miner 8: Score=87.5, Time=12.3s
[INFO] 📈 Statistics: Successful=5/9, Avg score=72.3
```

---

## 故障排查

### 常见问题

#### 1. SynapseParsingError

**症状**:
```
SynapseParsingError: Could not parse headers, 'total_size': '4860'
```

**原因**: 使用了旧版本 Protocol (v2.x 或 v3.0.0)

**解决**:
```bash
git pull origin main  # 更新到 v3.1.0
pip install -r requirements.txt --upgrade
# 重启 miner/validator
```

#### 2. Miner 收不到请求

**症状**: `Requests=0` 持续不变

**检查清单**:
1. 确认已注册到子网: `btcli subnet list --netuid 108`
2. 检查端口是否开放: `netstat -an | grep 8091`
3. 确认公网 IP 正确: `curl ifconfig.me`
4. 检查防火墙设置

#### 3. API 调用失败

**症状**:
```
[ERROR] OpenAI API error: Rate limit exceeded
```

**解决**:
- 检查 API Key 是否有效
- 检查账户配额
- 切换到其他模型 (Gemini 更便宜)

#### 4. 权重未更新

**症状**: Validator 运行但权重为 0

**原因**: 需要等待至少 1 个 epoch（~360个区块，约1小时）

**检查**:
```bash
btcli subnet metagraph --netuid 108 | grep "your_hotkey"
```

---

## 性能优化

### Miner 优化

1. **模型选择**:
   - GPT-4: 质量最高，但慢且贵
   - Claude 3.5 Sonnet: 平衡之选
   - Gemini 2.5 Flash: 最快最便宜

2. **并发优化**:
```python
# neurons/miner.py
self.axon.max_workers = 10  # 增加并发处理能力
```

3. **缓存策略**:
```python
# 缓存常见的 blueprint 生成
cache = {}
cache_key = hash(user_input)
if cache_key in cache:
    return cache[cache_key]
```

### Validator 优化

1. **查询策略**:
```python
# 减少查询 miners 数量以提升速度
VALIDATOR_SAMPLE_SIZE = 5  # 默认 9
```

2. **评分优化**:
```python
# 使用多进程评分
from multiprocessing import Pool
with Pool(4) as p:
    scores = p.map(calculate_score, responses)
```

3. **内存管理**:
```python
# 定期清理旧数据
if len(self.score_history) > 1000:
    self.score_history = self.score_history[-1000:]
```

---

## 主网部署准备

### 清单

- [ ] 完成至少 24 小时测试网稳定运行
- [ ] 确认 Protocol v3.1.0 无错误
- [ ] 准备主网钱包并充值足够 TAO
- [ ] 配置生产环境监控 (Prometheus + Grafana)
- [ ] 设置自动重启脚本 (systemd)
- [ ] 备份评分历史数据
- [ ] 准备应急响应计划

### 主网配置差异

```bash
# .env 更新
NETUID=<主网子网ID>  # 待确定

# 启动脚本更新
--subtensor.network finney  # 主网
```

---

## 总结

StoryFi Bittensor Subnet 通过**4阶段渐进式生成**和**100分客观评分系统**实现了去中心化的高质量 AI 故事生成。

**核心优势**:
1. 完全客观的评分算法（无人工主观判断）
2. 完善的反作弊机制
3. Protocol v3.1.0 解决了 header 限制问题
4. 支持多种 AI 模型

**当前状态**:
- ✅ Protocol 和算法已完成
- ✅ Miner/Validator 实现完成
- ⏳ 测试网验证中
- ⏳ 等待主网部署

**下一步**:
1. 完成 24 小时稳定性测试
2. 实施多模型备份系统
3. 优化评分算法细节
4. 准备主网部署

---

**最后更新**: 2025-10-21
**版本**: v1.0.0-beta
**Protocol**: v3.1.0
