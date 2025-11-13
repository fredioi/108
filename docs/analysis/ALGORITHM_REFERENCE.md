# StoryFi Bittensor Subnet - 算法路径参考文档

## 📋 目录

1. [核心文件路径](#核心文件路径)
2. [Miner 算法说明](#miner-算法说明)
3. [Validator 算法说明](#validator-算法说明)
4. [评分系统详解](#评分系统详解)
5. [Protocol 通信机制](#protocol-通信机制)
6. [快速导航](#快速导航)

---

## 核心文件路径

### 1. Protocol 通信协议

**路径**: `template/protocol.py`

**核心类**:
```python
class StoryGenerationSynapse(bt.Synapse)
```

**功能**:
- 定义 Validator 和 Miner 之间的通信格式
- 支持 4 种任务类型：blueprint, characters, story_arc, chapters
- Protocol v3.1.0 修复了 SynapseParsingError

**关键方法**:
- `get_total_size()`: 计算 HTTP Headers 大小（v3.1.0 核心修复）
- `validate_input_fields()`: 验证输入字段完整性
- `get_required_output_fields()`: 获取必需输出字段

**使用示例**:
```python
from template.protocol import create_blueprint_synapse

synapse = create_blueprint_synapse("一个关于AI觉醒的科幻故事")
response = await dendrite.query(axon, synapse)
```

---

### 2. Miner 实现

**路径**: `neurons/miner_gemini.py`

**核心类**:
```python
class StoryFiMiner
```

**入口点**: `main()` → `miner.run()`

**核心流程**:

#### 2.1 初始化 (lines 147-220)
```python
def __init__(self, config=None):
    # 1. 加载配置
    self.config = self.config or get_config()

    # 2. 初始化 Bittensor 组件
    self.wallet = bt.wallet(config=self.config)
    self.subtensor = bt.subtensor(config=self.config)
    self.metagraph = self.subtensor.metagraph(self.config.netuid)

    # 3. 初始化 AI 模型
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    self.model = genai.GenerativeModel("gemini-2.5-flash")

    # 4. 设置 Axon（接收请求）
    self.axon = bt.axon(wallet=self.wallet, config=self.config)
```

#### 2.2 故事生成算法

##### Blueprint 生成 (lines 222-279)
```python
async def generate_blueprint(self, synapse: StoryGenerationSynapse) -> StoryGenerationSynapse:
    """
    生成故事蓝图：世界观、主题、基调

    输入: synapse.user_input (用户故事需求)
    输出: synapse.output_data (Dict)
        - title: 故事标题
        - genre: 类型
        - setting: 背景设定
        - core_conflict: 核心冲突
        - themes: 主题列表
        - tone: 叙事基调
        - target_audience: 目标受众
    """

    # 构建 Prompt
    prompt = f"""
基于用户输入：{synapse.user_input}

生成一个完整的故事蓝图，包含：
1. 标题（吸引人的故事名称）
2. 类型（科幻/奇幻/悬疑等）
3. 背景设定（世界观描述）
4. 核心冲突（主要矛盾）
5. 主题（3-5个主题关键词）
6. 叙事基调（严肃/轻松/黑暗等）
7. 目标受众（青少年/成人/全年龄）

返回 JSON 格式。
"""

    # 调用 Gemini API
    response = self.model.generate_content(prompt)
    output_data = json.loads(response.text)

    # 填充响应
    synapse.output_data = output_data
    synapse.generation_time = time.time() - start_time
    synapse.miner_version = "1.0.0"

    return synapse
```

##### Characters 生成 (lines 281-345)
```python
async def generate_characters(self, synapse: StoryGenerationSynapse) -> StoryGenerationSynapse:
    """
    生成角色档案：5个主要角色

    输入:
        - synapse.user_input: 用户需求
        - synapse.blueprint: 故事蓝图

    输出: synapse.output_data (Dict)
        - characters: List[Dict] (5个角色)
            - name: 角色名称
            - role: 角色定位
            - background: 背景故事
            - personality: 性格特点
            - goals: 目标动机
            - relationships: 与其他角色的关系
    """

    prompt = f"""
基于故事蓝图：
{json.dumps(synapse.blueprint, ensure_ascii=False, indent=2)}

生成5个主要角色，每个角色包含：
1. 名称
2. 角色定位（主角/反派/配角等）
3. 背景故事（300字）
4. 性格特点（5个关键词）
5. 目标动机
6. 与其他角色的关系

返回 JSON 格式：{{"characters": [...]}}
"""

    # ... 调用 AI 生成 ...
```

##### Story Arc 生成 (lines 347-419)
```python
async def generate_story_arc(self, synapse: StoryGenerationSynapse) -> StoryGenerationSynapse:
    """
    生成故事结构：12章大纲

    输入:
        - synapse.blueprint: 故事蓝图
        - synapse.characters: 角色列表

    输出: synapse.output_data (Dict)
        - title: 故事总标题
        - description: 故事简介
        - chapters: List[Dict] (12章)
            - chapter_id: 章节编号 (1-12)
            - title: 章节标题
            - summary: 章节概要
            - key_events: 关键事件列表
            - character_focus: 本章重点角色
        - arcs: List[str] (3个故事弧)
            - Act 1: Setup (章节1-4)
            - Act 2: Confrontation (章节5-8)
            - Act 3: Resolution (章节9-12)
        - themes: 主题演进
        - hooks: 悬念设置
    """

    prompt = f"""
基于故事蓝图和角色：
蓝图：{json.dumps(synapse.blueprint, ensure_ascii=False)}
角色：{json.dumps(synapse.characters, ensure_ascii=False)}

生成12章故事大纲，遵循三幕结构：
- Act 1 (章节1-4): 世界观建立、角色引入、冲突萌芽
- Act 2 (章节5-8): 冲突升级、角色成长、危机爆发
- Act 3 (章节9-12): 高潮对决、冲突解决、结局收尾

每章包含：标题、概要、关键事件、重点角色

返回 JSON 格式。
"""

    # ... 调用 AI 生成 ...
```

##### Chapters 生成 (lines 421-480)
```python
async def generate_chapters(self, synapse: StoryGenerationSynapse) -> StoryGenerationSynapse:
    """
    生成详细章节内容：带选择的交互式章节

    输入:
        - synapse.blueprint: 故事蓝图
        - synapse.characters: 角色列表
        - synapse.story_arc: 故事结构
        - synapse.chapter_ids: 要生成的章节ID列表 (e.g., [1, 2, 3])

    输出: synapse.output_data (Dict)
        - chapters: List[Dict]
            - chapter_id: 章节编号
            - title: 章节标题
            - content: 章节正文 (2000-3000字)
            - choices: List[Dict] (3-4个选择)
                - choice_id: 选择编号
                - text: 选择文本
                - consequence: 后果描述
                - next_chapter: 下一章ID
    """

    prompt = f"""
基于完整故事信息：
蓝图：{synapse.blueprint}
角色：{synapse.characters}
故事结构：{synapse.story_arc}

为章节 {synapse.chapter_ids} 生成详细内容：

每章包含：
1. 标题
2. 正文（2000-3000字，包含对话、描写、心理活动）
3. 3-4个玩家选择（每个选择影响剧情走向）

返回 JSON 格式。
"""

    # ... 调用 AI 生成 ...
```

#### 2.3 请求处理流程 (lines 433-455)

```python
async def run(self):
    """Miner 主循环"""

    # 1. 附加请求处理器
    self.axon.attach(
        forward_fn=self.forward_blueprint,
        blacklist_fn=self.blacklist,
        priority_fn=self.priority
    ).attach(
        forward_fn=self.forward_characters,
        # ...
    ).attach(
        forward_fn=self.forward_story_arc,
        # ...
    ).attach(
        forward_fn=self.forward_chapters,
        # ...
    )

    # 2. 启动 Axon（监听端口）
    self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

    # 3. 主循环：定期同步 Metagraph
    while True:
        await asyncio.sleep(60)  # 每分钟同步一次
        self.metagraph.sync(subtensor=self.subtensor)
        bt.logging.info(f"📊 Stats: Requests={self.request_count}, AvgTime={avg_time:.2f}s, Errors={self.error_count}")
```

---

### 3. Validator 实现

**路径**: `neurons/validator.py`

**核心类**:
```python
class StoryFiValidator
```

**入口点**: `main()` → `validator.run()`

**核心流程**:

#### 3.1 初始化 (lines 98-152)
```python
def __init__(self, config=None):
    # 1. 加载配置
    self.config = self.config or get_config()

    # 2. 初始化 Bittensor 组件
    self.wallet = bt.wallet(config=self.config)
    self.subtensor = bt.subtensor(config=self.config)
    self.metagraph = self.subtensor.metagraph(self.config.netuid)
    self.dendrite = bt.dendrite(wallet=self.wallet)

    # 3. 初始化评分系统
    self.moving_averaged_scores = torch.zeros(self.metagraph.n)  # EMA scores

    # 4. 反作弊系统
    self.response_history = {}  # 存储历史响应用于抄袭检测
    self.blacklist = set()  # 黑名单
```

#### 3.2 查询流程 (lines 250-330)

```python
async def query_miners(self, task_type: str, user_input: str):
    """
    查询所有 Miners 并评分

    Args:
        task_type: "blueprint" | "characters" | "story_arc" | "chapters"
        user_input: 用户故事需求

    Returns:
        Dict: {
            "responses": List[response],
            "scores": torch.Tensor,
            "best_miner_uid": int
        }
    """

    # 1. 创建 Synapse
    if task_type == "blueprint":
        synapse = create_blueprint_synapse(user_input)
    elif task_type == "characters":
        synapse = create_characters_synapse(blueprint, user_input)
    # ...

    # 2. 选择要查询的 Miners（随机或全部）
    miner_uids = self.get_query_uids()

    # 3. 并发查询所有 Miners
    responses = await self.dendrite.forward(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        timeout=60
    )

    # 4. 评分
    scores = torch.zeros(len(responses))
    for i, response in enumerate(responses):
        if response is None or not hasattr(response, 'output_data'):
            scores[i] = 0.0
            continue

        # 调用评分算法
        score = self.score_response(response, task_type)
        scores[i] = score

    # 5. 更新 EMA scores
    self.update_scores(scores, miner_uids)

    return {
        "responses": responses,
        "scores": scores,
        "best_miner_uid": miner_uids[scores.argmax()]
    }
```

#### 3.3 评分算法 (lines 154-248)

```python
def score_response(self, response: StoryGenerationSynapse, task_type: str) -> float:
    """
    100分制评分系统

    组成：
        - Technical Score: 30分
        - Structure Score: 40分
        - Content Score: 30分

    Returns:
        float: 0-100分
    """

    technical_score = self.calculate_technical_score(response, task_type)  # 0-30
    structure_score = self.calculate_structure_score(response, task_type)  # 0-40
    content_score = self.calculate_content_score(response, task_type)      # 0-30

    total_score = technical_score + structure_score + content_score

    # 反作弊检测
    if self.is_plagiarism(response):
        total_score *= 0.1  # 抄袭惩罚：只保留10%分数

    return max(0.0, min(100.0, total_score))  # 限制在 0-100
```

##### 技术评分 (30分)
```python
def calculate_technical_score(self, response, task_type) -> float:
    """
    评估技术质量

    评分项：
    1. JSON 格式正确性 (10分)
    2. 必需字段完整性 (10分)
    3. 响应速度 (10分)
        - <10s: 满分
        - 10-30s: 8分
        - 30-60s: 5分
        - >60s: 0分
    """
    score = 0.0

    # 1. JSON 格式
    try:
        data = response.output_data
        if isinstance(data, dict):
            score += 10.0
    except:
        return 0.0

    # 2. 字段完整性
    required_fields = response.get_required_output_fields()
    missing = [f for f in required_fields if f not in data]
    if not missing:
        score += 10.0
    else:
        score += 10.0 * (1 - len(missing) / len(required_fields))

    # 3. 响应速度
    gen_time = response.generation_time
    if gen_time < 10:
        score += 10.0
    elif gen_time < 30:
        score += 8.0
    elif gen_time < 60:
        score += 5.0

    return score
```

##### 结构评分 (40分)
```python
def calculate_structure_score(self, response, task_type) -> float:
    """
    评估内容结构质量

    评分项（根据 task_type 不同）：

    blueprint:
        - 世界观完整性 (15分)
        - 冲突设计合理性 (15分)
        - 主题深度 (10分)

    characters:
        - 角色数量正确 (10分) - 必须5个
        - 角色差异化 (15分) - 性格/背景不重复
        - 关系网络合理性 (15分)

    story_arc:
        - 章节数量正确 (10分) - 必须12章
        - 三幕结构完整 (15分)
        - 剧情连贯性 (15分)

    chapters:
        - 字数达标 (10分) - 2000-3000字
        - 选择设计 (15分) - 3-4个有意义的选择
        - 后果差异化 (15分) - 不同选择导致不同结果
    """

    score = 0.0
    data = response.output_data

    if task_type == "blueprint":
        # 世界观完整性
        if "setting" in data and len(data["setting"]) > 200:
            score += 15.0

        # 冲突设计
        if "core_conflict" in data and len(data["core_conflict"]) > 100:
            score += 15.0

        # 主题深度
        if "themes" in data and len(data["themes"]) >= 3:
            score += 10.0

    elif task_type == "characters":
        characters = data.get("characters", [])

        # 数量正确
        if len(characters) == 5:
            score += 10.0

        # 差异化（名字不重复，性格不同）
        names = [c.get("name") for c in characters]
        if len(set(names)) == 5:
            score += 7.5

        personalities = [str(c.get("personality")) for c in characters]
        if len(set(personalities)) >= 4:  # 允许部分重叠
            score += 7.5

        # 关系网络
        has_relationships = sum(1 for c in characters if "relationships" in c)
        score += 15.0 * (has_relationships / 5)

    elif task_type == "story_arc":
        chapters = data.get("chapters", [])

        # 数量正确
        if len(chapters) == 12:
            score += 10.0

        # 三幕结构
        if "arcs" in data and len(data["arcs"]) == 3:
            score += 15.0

        # 连贯性（每章有前后关联）
        coherent = sum(1 for ch in chapters if "key_events" in ch and len(ch["key_events"]) > 0)
        score += 15.0 * (coherent / 12)

    elif task_type == "chapters":
        chapters = data.get("chapters", [])

        for chapter in chapters:
            content = chapter.get("content", "")
            choices = chapter.get("choices", [])

            # 字数
            word_count = len(content)
            if 2000 <= word_count <= 3000:
                score += 10.0 / len(chapters)

            # 选择数量
            if 3 <= len(choices) <= 4:
                score += 15.0 / len(chapters)

            # 选择差异化
            consequences = [c.get("consequence") for c in choices]
            if len(set(consequences)) == len(consequences):
                score += 15.0 / len(chapters)

    return min(40.0, score)
```

##### 内容评分 (30分)
```python
def calculate_content_score(self, response, task_type) -> float:
    """
    评估内容质量（主观维度）

    评分项：
    1. 创意性 (10分)
        - 避免常见套路
        - 设定新颖
        - 角色独特

    2. 可读性 (10分)
        - 语言流畅
        - 描写生动
        - 对话自然

    3. 相关性 (10分)
        - 符合用户需求
        - 匹配故事类型
        - 保持一致性

    实现方式（简化版）：
        - 使用关键词匹配
        - 计算文本多样性（词汇丰富度）
        - 检查与 user_input 的相关性
    """

    score = 0.0
    data = response.output_data
    user_input = response.user_input

    # 1. 创意性（词汇多样性）
    text = json.dumps(data, ensure_ascii=False)
    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    score += 10.0 * unique_ratio

    # 2. 可读性（平均句子长度合理性）
    sentences = text.split("。")
    avg_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
    if 20 < avg_length < 100:  # 合理范围
        score += 10.0
    else:
        score += 5.0

    # 3. 相关性（关键词匹配）
    user_keywords = set(user_input.split())
    content_keywords = set(text.split())
    overlap = len(user_keywords & content_keywords) / max(len(user_keywords), 1)
    score += 10.0 * overlap

    return min(30.0, score)
```

##### 反作弊检测
```python
def is_plagiarism(self, response) -> bool:
    """
    抄袭检测

    方法：
    1. 计算与历史响应的相似度（Levenshtein距离）
    2. 阈值：>90% 相似视为抄袭
    3. 黑名单机制
    """

    current_text = json.dumps(response.output_data, sort_keys=True)

    for historical_text in self.response_history.values():
        similarity = self.calculate_similarity(current_text, historical_text)
        if similarity > 0.90:  # 90% 相似度阈值
            return True

    # 存储当前响应
    self.response_history[response.miner_hotkey] = current_text

    return False

def calculate_similarity(self, text1, text2) -> float:
    """计算文本相似度（简化版 Levenshtein）"""
    import difflib
    return difflib.SequenceMatcher(None, text1, text2).ratio()
```

#### 3.4 权重更新 (lines 332-395)

```python
def update_scores(self, new_scores: torch.Tensor, miner_uids: List[int]):
    """
    更新 EMA scores 并设置链上权重

    EMA公式：
        score_t = alpha * new_score + (1 - alpha) * score_{t-1}
        alpha = 0.1 (平滑系数)

    Softmax with Temperature:
        weights_i = exp(score_i / T) / Σ exp(score_j / T)
        T = 2.0 (温度参数，增加多样性)
    """

    # 1. 更新 EMA scores
    alpha = 0.1
    for i, uid in enumerate(miner_uids):
        self.moving_averaged_scores[uid] = (
            alpha * new_scores[i] +
            (1 - alpha) * self.moving_averaged_scores[uid]
        )

    # 2. Softmax normalization
    temperature = 2.0
    weights = torch.nn.functional.softmax(
        self.moving_averaged_scores / temperature,
        dim=0
    )

    # 3. 设置链上权重（每5分钟一次）
    if self.should_set_weights():
        self.subtensor.set_weights(
            netuid=self.config.netuid,
            wallet=self.wallet,
            uids=self.metagraph.uids,
            weights=weights,
            wait_for_inclusion=False
        )
        bt.logging.info(f"✅ Weights set: {weights}")
```

#### 3.5 主循环 (lines 397-450)

```python
async def run(self):
    """Validator 主循环"""

    step = 0

    while True:
        try:
            # 1. 随机选择任务类型
            task_type = random.choice(["blueprint", "characters", "story_arc", "chapters"])

            # 2. 生成测试输入
            user_input = self.generate_test_input(task_type)

            # 3. 查询 Miners
            bt.logging.info(f"🎯 Task type: {task_type}")
            result = await self.query_miners(task_type, user_input)

            # 4. 记录结果
            bt.logging.info(f"✅ Best miner: {result['best_miner_uid']}, Score: {result['scores'].max():.2f}")

            # 5. 等待下一轮
            await asyncio.sleep(12)  # 12秒间隔

            # 6. 定期同步 Metagraph
            if step % 5 == 0:
                self.metagraph.sync(subtensor=self.subtensor)

            step += 1

        except KeyboardInterrupt:
            bt.logging.info("Validator stopped by user")
            break
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            await asyncio.sleep(60)
```

---

## 评分系统详解

### 完整评分流程

```
用户请求 → Validator 创建 Synapse
    ↓
查询所有 Miners（并发）
    ↓
Miner A: 返回响应 + 生成时间
Miner B: 返回响应 + 生成时间
Miner C: 返回响应 + 生成时间
    ↓
Validator 评分（每个 Miner）:
    ├─ Technical (30分)
    │   ├─ JSON 格式 (10分)
    │   ├─ 字段完整 (10分)
    │   └─ 响应速度 (10分)
    │
    ├─ Structure (40分)
    │   ├─ 数量/格式正确 (10分)
    │   ├─ 内容结构合理 (15分)
    │   └─ 逻辑连贯性 (15分)
    │
    └─ Content (30分)
        ├─ 创意性 (10分)
        ├─ 可读性 (10分)
        └─ 相关性 (10分)
    ↓
反作弊检测:
    - 抄袭检测（>90%相似度）
    - 黑名单过滤
    ↓
EMA 平滑:
    score_t = 0.1 * new_score + 0.9 * score_{t-1}
    ↓
Softmax 归一化:
    weight_i = exp(score_i / 2.0) / Σ exp(score_j / 2.0)
    ↓
链上权重更新（每5分钟）
```

---

## Protocol 通信机制

### 双路传输架构（v3.1.0）

```
Validator                                    Miner
    │                                          │
    ├─ 创建 Synapse                            │
    │  ├─ task_type = "blueprint"             │
    │  └─ user_input = "..."                  │
    │                                          │
    ├─ dendrite.forward()                      │
    │  ├─ HTTP Headers (metadata)             │
    │  │   ├─ task_type: "" (dummy)           │
    │  │   ├─ user_input: "" (dummy)          │
    │  │   └─ total_size: 1024 (header size)  │
    │  │                                       │
    │  └─ HTTP Body (actual data)             │
    │      ├─ task_type: "blueprint"          │
    │      ├─ user_input: "完整内容"          │
    │      ├─ blueprint: {...}                │
    │      └─ characters: {...}               │
    │                                          │
    │ ──────────────────────────────────────> │
    │                                          │
    │                                          ├─ axon 接收请求
    │                                          ├─ 解析 HTTP Body
    │                                          ├─ 调用 forward_blueprint()
    │                                          ├─ 生成故事内容
    │                                          └─ 填充 output_data
    │                                          │
    │ <──────────────────────────────────────── │
    │                                          │
    │  HTTP Body (response):                  │
    │      ├─ output_data: {...}              │
    │      ├─ generation_time: 15.3           │
    │      └─ miner_version: "1.0.0"          │
    │                                          │
    ├─ 接收响应                                │
    ├─ 评分 score_response()                  │
    └─ 更新权重 update_scores()               │
```

### v3.1.0 关键修复

**问题**: `total_size` header 包含完整对象大小（3-5KB），超过 HTTP Header 限制

**解决方案**: 重写 `get_total_size()` 只返回 header 传输的数据大小

```python
def get_total_size(self) -> int:
    # 清空大字段
    header_only = self.model_copy()
    header_only.blueprint = None
    header_only.characters = None
    header_only.story_arc = None
    header_only.output_data = None

    # 只计算小字段 + 元数据
    header_size = sys.getsizeof(header_only) + 512

    return header_size  # ~1KB
```

**效果**:
- v3.0.0: `total_size` = 3658-4860 bytes → SynapseParsingError
- v3.1.0: `total_size` = 637-1024 bytes → ✅ No errors

---

## 快速导航

### 我想看...

**Miner 如何生成故事？**
→ `neurons/miner_gemini.py` lines 222-480

**Validator 如何评分？**
→ `neurons/validator.py` lines 154-248

**评分算法详细计算？**
→ 本文档 [评分系统详解](#评分系统详解)

**Protocol 通信格式？**
→ `template/protocol.py` + 本文档 [Protocol 通信机制](#protocol-通信机制)

**如何部署到测试网？**
→ `DEPLOYMENT_GUIDE.md`

**Protocol v3.1.0 修复了什么？**
→ `PROTOCOL_V3_ANALYSIS.md`

---

## 参数配置总结

### Miner 配置

```bash
# .env
GEMINI_API_KEY=your_api_key_here

# 命令行参数
--netuid 108                    # 子网 ID
--subtensor.network test        # 测试网
--wallet.name storyfi_miner     # 钱包名称
--wallet.hotkey default         # Hotkey
--axon.port 8091                # 监听端口
--logging.info                  # 日志级别
```

### Validator 配置

```bash
# 命令行参数
--netuid 108                    # 子网 ID
--subtensor.network test        # 测试网
--wallet.name storyfi_validator # 钱包名称
--wallet.hotkey default         # Hotkey
--logging.info                  # 日志级别
```

### 评分参数

```python
# Technical Score
JSON_VALID_SCORE = 10.0
FIELDS_COMPLETE_SCORE = 10.0
SPEED_FAST_THRESHOLD = 10.0  # seconds
SPEED_MEDIUM_THRESHOLD = 30.0
SPEED_SLOW_THRESHOLD = 60.0

# Structure Score
# （根据 task_type 不同）

# Content Score
CREATIVITY_SCORE = 10.0
READABILITY_SCORE = 10.0
RELEVANCE_SCORE = 10.0

# Anti-Cheat
PLAGIARISM_THRESHOLD = 0.90  # 90% similarity
PLAGIARISM_PENALTY = 0.1     # 10% of original score

# Weight Update
EMA_ALPHA = 0.1              # Smoothing factor
SOFTMAX_TEMPERATURE = 2.0    # Diversity factor
WEIGHT_UPDATE_INTERVAL = 300 # 5 minutes
```

---

## 文档版本

- **Version**: 1.0.0
- **Protocol**: v3.1.0
- **Last Updated**: 2025-10-23
- **Author**: StoryFi Team

## 相关文档

- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - 完整部署指南
- [PROTOCOL_V3_ANALYSIS.md](./PROTOCOL_V3_ANALYSIS.md) - Protocol 分析报告
- [README.md](./README.md) - 项目概览

---

**如有问题，请查看代码注释或联系开发团队。**
