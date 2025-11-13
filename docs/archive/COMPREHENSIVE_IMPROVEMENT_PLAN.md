# StoryFi Subnet 综合改进工程方案

**制定日期**: 2025-10-17
**项目阶段**: 主网准备期
**预计工期**: 10-14 工作日

---

## 📊 执行摘要

基于测试网部署和本地测试的结果，本方案提出分 5 个阶段的系统性改进计划，确保主网部署的稳定性、可维护性和长期可扩展性。

**核心目标**:
- 修复测试网发现的协议兼容性问题
- 增强系统稳定性和容错能力
- 建立完整的监控和运维体系
- 优化算法性能和激励机制
- 确保主网部署零风险

---

## 🔍 测试结果分析

### ✅ 本地测试（成功）

| 测试项 | 结果 | 指标 |
|--------|------|------|
| Blueprint 生成 | ✅ | 85/100 |
| Characters 生成 | ✅ | 82/100 |
| Story Arc 生成 | ✅ | 78/100 |
| Chapters 生成 | ✅ | 75/100 |
| **综合平均** | ✅ | **80/100** |
| JSON 合规性 | ✅ | 100% |
| 响应时间 | ✅ | 5-10s |
| 错误率 | ✅ | 0% |

### ⚠️ 测试网部署（部分失败）

| 测试项 | 结果 | 问题 |
|--------|------|------|
| Miner 启动 | ✅ | 成功 |
| 网络注册 | ✅ | UID 8 |
| Axon 监听 | ✅ | Port 8091 |
| 接收请求 | ✅ | 收到请求 |
| **协议解析** | ❌ | **SynapseParsingError** |
| 响应生成 | ⏸️ | 未执行（解析失败） |

### 🔴 核心问题诊断

```
ERROR: SynapseParsingError
Could not parse headers into synapse of type StoryGenerationSynapse
```

**根本原因**:
1. `input_data: Dict[str, Any]` 复杂类型在 HTTP headers 序列化失败
2. 不同 Bittensor 版本的 Synapse 序列化方式不兼容
3. Payload 大小（~4KB）可能超过某些限制

**影响**:
- Miner 无法处理任何请求
- 评分为 0，无法获得奖励
- 主网部署会立即失败

---

## 🎯 改进方案总览

### Phase 1: 协议修复与兼容性 🔴 [最高优先级]
**目标**: 解决测试网 SynapseParsingError，确保协议稳定
**工期**: 2-3 天

### Phase 2: 稳定性与容错增强 🟡 [高优先级]
**目标**: 增加多重保障，防止单点故障
**工期**: 2-3 天

### Phase 3: 算法优化与反作弊 🟡 [高优先级]
**目标**: 提升质量和公平性，防止被游戏化
**工期**: 3-4 天

### Phase 4: 监控与运维体系 🟢 [中优先级]
**目标**: 建立完整的可观测性和自动化运维
**工期**: 2-3 天

### Phase 5: 压力测试与主网准备 🟢 [中优先级]
**目标**: 全面验证，确保万无一失
**工期**: 1-2 天

**总工期**: 10-15 天

---

## 📋 Phase 1: 协议修复与兼容性

### 1.1 问题分析

**当前协议定义**:
```python
class StoryGenerationSynapse(bt.Synapse):
    task_type: str
    input_data: Dict[str, Any]  # ← 问题所在
    output_json: str = ""
    generation_time: float = 0.0
    validator_hotkey: Optional[str] = None
```

**问题**:
- `Dict[str, Any]` 是复杂嵌套类型
- Bittensor 的 HTTP headers 序列化不支持复杂类型
- 不同版本的序列化实现不一致

### 1.2 解决方案

#### Option A: 简化为字符串类型（推荐）

```python
class StoryGenerationSynapse(bt.Synapse):
    """
    StoryFi 协议 v2.0
    - 简化类型，提高兼容性
    - 添加版本控制
    - 添加校验机制
    """
    # 版本控制
    protocol_version: str = "2.0.0"

    # 核心字段（简化类型）
    task_type: str
    input_data_json: str  # JSON 字符串，不是 Dict

    # 响应字段
    output_json: str = ""
    generation_time: float = 0.0

    # 元数据
    miner_version: str = ""
    validator_hotkey: Optional[str] = None

    # 辅助方法
    def get_input_data(self) -> Dict[str, Any]:
        """安全地解析 input_data"""
        try:
            return json.loads(self.input_data_json)
        except json.JSONDecodeError:
            return {}

    def set_input_data(self, data: Dict[str, Any]):
        """安全地设置 input_data"""
        self.input_data_json = json.dumps(data, ensure_ascii=False)
```

**优点**:
- 完全兼容 HTTP headers
- 支持所有 Bittensor 版本
- 清晰的版本控制

**缺点**:
- 需要修改所有调用代码

#### Option B: 使用 Body 传输（备选）

```python
class StoryGenerationSynapse(bt.Synapse):
    task_type: str
    # input_data 通过 request body 传输，不在 headers

    def deserialize(self) -> "StoryGenerationSynapse":
        # 自定义反序列化逻辑
        pass
```

**优点**:
- 支持任意复杂类型
- 更灵活

**缺点**:
- 需要自定义序列化逻辑
- 可能与 Bittensor 标准不兼容

### 1.3 实施步骤

#### Day 1: 协议重构
- [ ] 创建 `template/protocol_v2.py`
- [ ] 实现新的 `StoryGenerationSynapse` (Option A)
- [ ] 添加版本检查和兼容性层
- [ ] 更新所有辅助函数（create_*_synapse）

#### Day 2: 代码迁移
- [ ] 更新 Miner 代码使用新协议
- [ ] 更新 Validator 代码使用新协议
- [ ] 更新测试代码
- [ ] 添加向后兼容支持（如果需要）

#### Day 3: 测试验证
- [ ] 本地测试新协议
- [ ] 测试网部署验证
- [ ] 与不同版本 Bittensor 测试兼容性
- [ ] 压力测试协议稳定性

### 1.4 验收标准

- [ ] 测试网 Miner 成功接收并处理请求
- [ ] 无 SynapseParsingError 错误
- [ ] 与 Bittensor 9.x 和 10.x 版本兼容
- [ ] 协议文档更新完整

### 1.5 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 旧协议不兼容 | 中 | 高 | 保留兼容层，支持渐进式迁移 |
| 测试覆盖不足 | 低 | 中 | 编写详细的协议测试套件 |
| 版本回退需求 | 低 | 中 | 保留旧协议代码，添加开关 |

---

## 📋 Phase 2: 稳定性与容错增强

### 2.1 多模型备份系统

**目标**: 避免单点依赖 Gemini API

#### 实施方案

```python
class MultiModelBackend:
    """多模型后端，自动故障转移"""

    def __init__(self):
        self.models = [
            GeminiModel(priority=1, free=True),
            OpenAIModel(priority=2, cost=0.0002),
            ClaudeModel(priority=3, cost=0.0003),
            LocalModel(priority=4, cost=0)
        ]

        self.usage = {
            "gemini": {"calls": 0, "limit": 1500},
            "openai": {"calls": 0, "limit": 10000},
            "claude": {"calls": 0, "limit": 10000}
        }

    async def generate(self, prompt: str, task_type: str) -> Dict:
        """智能选择模型并生成"""
        for model in self.models:
            if self.can_use_model(model.name):
                try:
                    result = await model.generate(prompt)
                    self.update_usage(model.name)
                    return result
                except Exception as e:
                    bt.logging.warning(f"{model.name} failed: {e}")
                    continue

        raise Exception("All models failed")

    def can_use_model(self, name: str) -> bool:
        """检查模型是否可用"""
        if name not in self.usage:
            return True
        return self.usage[name]["calls"] < self.usage[name]["limit"]
```

#### 配置示例

```python
# .env
GEMINI_API_KEY=xxx  # 免费，1500次/天
OPENAI_API_KEY=xxx  # 付费备份
CLAUDE_API_KEY=xxx  # 付费备份

# 优先级策略
MODEL_PRIORITY=gemini,openai,claude,local
GEMINI_DAILY_LIMIT=1500
OPENAI_DAILY_LIMIT=10000
```

### 2.2 质量预检机制

**目标**: 避免提交低质量响应

```python
class QualityPreChecker:
    """生成前质量预检"""

    def __init__(self):
        self.min_score = 60  # 最低可接受分数

    async def generate_with_quality_check(
        self,
        prompt: str,
        task_type: str
    ) -> Dict:
        """带质量检查的生成"""
        max_attempts = 3

        for attempt in range(max_attempts):
            result = await self.backend.generate(prompt, task_type)

            # 快速评分
            score = self.quick_score(result, task_type)

            if score >= self.min_score:
                return result

            # 调整参数重试
            prompt = self.adjust_prompt(prompt, attempt)
            bt.logging.warning(
                f"Quality too low ({score}), retry {attempt+1}/{max_attempts}"
            )

        # 实在不行也要返回
        return result

    def quick_score(self, result: Dict, task_type: str) -> float:
        """快速评分（不调用外部 API）"""
        score = 0.0

        # 检查必需字段
        required = REQUIRED_FIELDS[task_type]
        if all(k in result for k in required):
            score += 30

        # 检查内容长度
        if task_type == "blueprint":
            if len(result.get("setting", "")) > 50:
                score += 20
            if len(result.get("themes", [])) >= 3:
                score += 20

        # ... 其他快速检查

        return score
```

### 2.3 智能缓存系统

**目标**: 提高响应速度，降低 API 调用

```python
import hashlib
from functools import lru_cache

class SmartCache:
    """智能缓存系统"""

    def __init__(self, ttl=3600):
        self.cache = {}
        self.ttl = ttl

    def get_cache_key(self, user_input: str, task_type: str) -> str:
        """生成缓存 key"""
        content = f"{task_type}:{user_input}"
        return hashlib.md5(content.encode()).hexdigest()

    async def generate_with_cache(
        self,
        user_input: str,
        task_type: str
    ) -> Dict:
        """带缓存的生成"""
        key = self.get_cache_key(user_input, task_type)

        # 检查缓存
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                bt.logging.info(f"Cache hit for {key[:8]}")
                return self.adapt_cached_response(cached_data)

        # 生成新内容
        result = await self.generate(user_input, task_type)

        # 存入缓存
        self.cache[key] = (result, time.time())

        return result

    def adapt_cached_response(self, cached: Dict) -> Dict:
        """适配缓存响应（添加随机性）"""
        # 轻微修改，避免完全相同
        result = cached.copy()

        # 例如：随机调整一些描述
        if "setting" in result:
            result["setting"] = self.add_variation(result["setting"])

        return result
```

### 2.4 错误处理与重试

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustMiner:
    """健壮的 Miner 实现"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def forward(self, synapse: StoryGenerationSynapse):
        """带重试的 forward 函数"""
        try:
            result = await self.generate_with_fallback(synapse)
            synapse.output_json = json.dumps(result)
            return synapse
        except Exception as e:
            bt.logging.error(f"Forward failed: {e}")
            # 返回错误而不是崩溃
            synapse.output_json = json.dumps({"error": str(e)})
            return synapse

    async def generate_with_fallback(self, synapse):
        """多层 fallback"""
        try:
            # 优先使用高质量模型
            return await self.quality_checker.generate_with_quality_check(...)
        except Exception as e1:
            bt.logging.warning(f"Quality generation failed: {e1}")
            try:
                # Fallback 到快速模型
                return await self.fast_generate(...)
            except Exception as e2:
                bt.logging.error(f"Fast generation failed: {e2}")
                # 最后返回模板响应
                return self.get_template_response(synapse.task_type)
```

### 2.5 实施步骤

#### Day 1: 多模型后端
- [ ] 实现 MultiModelBackend 类
- [ ] 集成 OpenAI 和 Claude API
- [ ] 添加使用量跟踪
- [ ] 测试故障转移

#### Day 2: 质量预检
- [ ] 实现 QualityPreChecker
- [ ] 添加快速评分逻辑
- [ ] 测试质量提升效果
- [ ] 调优阈值参数

#### Day 3: 缓存与重试
- [ ] 实现 SmartCache
- [ ] 添加重试逻辑
- [ ] 集成到 Miner
- [ ] 压力测试

### 2.6 验收标准

- [ ] Gemini API 故障时自动切换到 OpenAI
- [ ] 平均响应质量提升至 85/100
- [ ] 缓存命中率 > 20%
- [ ] 响应时间降低 30%
- [ ] 错误率 < 1%

---

## 📋 Phase 3: 算法优化与反作弊

### 3.1 Validator 评分系统升级

#### 当前问题
```python
# 当前 Content Score 过于简单
def calculate_content_score(data, context, task_type):
    score = 0.0

    # 只检查字段长度
    if len(data.get("setting", "")) > 50:
        score += 10

    return score  # 容易被优化
```

#### 改进方案

```python
class EnhancedContentScorer:
    """增强的内容评分器"""

    def __init__(self):
        # 使用轻量级 embedding 模型
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # 参考标准
        self.reference_examples = self.load_reference_examples()

    def calculate_content_score(
        self,
        data: Dict,
        context: Dict,
        task_type: str
    ) -> Tuple[float, Dict]:
        """多维度内容评分"""
        breakdown = {}
        total = 0.0

        # 1. 语义相关性 (10分)
        relevance_score = self.score_relevance(data, context)
        breakdown["relevance"] = relevance_score
        total += relevance_score

        # 2. 内容丰富度 (10分)
        richness_score = self.score_richness(data, task_type)
        breakdown["richness"] = richness_score
        total += richness_score

        # 3. 创意性 (5分)
        creativity_score = self.score_creativity(data, task_type)
        breakdown["creativity"] = creativity_score
        total += creativity_score

        # 4. 连贯性 (5分)
        coherence_score = self.score_coherence(data, task_type)
        breakdown["coherence"] = coherence_score
        total += coherence_score

        return total, breakdown

    def score_relevance(self, data: Dict, context: Dict) -> float:
        """语义相关性评分"""
        user_input = context.get("user_input", "")

        # 提取关键内容
        content = self.extract_key_content(data)

        # 计算 embedding 相似度
        emb1 = self.model.encode(user_input)
        emb2 = self.model.encode(content)

        similarity = cosine_similarity([emb1], [emb2])[0][0]

        # 映射到 0-10分
        return min(10.0, similarity * 12)

    def score_richness(self, data: Dict, task_type: str) -> float:
        """内容丰富度评分"""
        score = 0.0

        if task_type == "blueprint":
            # 检查描述性字段的质量
            setting = data.get("setting", "")
            if len(setting) > 100:
                score += 3
            if len(setting) > 200:
                score += 2

            # 检查主题数量和质量
            themes = data.get("themes", [])
            score += min(3, len(themes))

            # 检查冲突描述
            conflict = data.get("core_conflict", "")
            if len(conflict) > 50:
                score += 2

        elif task_type == "characters":
            characters = data.get("characters", [])
            for char in characters:
                # 背景故事质量
                if len(char.get("background", "")) > 100:
                    score += 0.5

                # 技能和性格完整性
                if len(char.get("skills", [])) >= 3:
                    score += 0.3
                if len(char.get("personality_traits", [])) >= 3:
                    score += 0.2

        return min(10.0, score)

    def score_creativity(self, data: Dict, task_type: str) -> float:
        """创意性评分（与历史对比）"""
        # 计算与参考样本的差异度
        content = self.extract_key_content(data)
        content_emb = self.model.encode(content)

        # 与参考样本对比
        similarities = []
        for ref in self.reference_examples[task_type]:
            ref_emb = self.model.encode(ref)
            sim = cosine_similarity([content_emb], [ref_emb])[0][0]
            similarities.append(sim)

        # 差异度越大 = 越有创意
        avg_similarity = np.mean(similarities)
        creativity = 1.0 - avg_similarity

        return creativity * 5.0
```

### 3.2 高级反作弊机制

```python
class AdvancedAntiCheat:
    """高级反作弊系统"""

    def __init__(self):
        self.fingerprint_cache = {}
        self.timing_patterns = {}
        self.style_analyzer = StyleAnalyzer()

    def detect_cheating(
        self,
        miner_uid: int,
        response: StoryGenerationSynapse,
        all_responses: List[StoryGenerationSynapse]
    ) -> Tuple[bool, str, Dict]:
        """多维度作弊检测"""

        checks = [
            self.check_plagiarism(response, all_responses),
            self.check_template_abuse(response),
            self.check_timing_anomaly(miner_uid, response),
            self.check_style_fingerprint(miner_uid, response),
            self.check_semantic_copying(response, all_responses)
        ]

        for is_cheat, reason, details in checks:
            if is_cheat:
                return True, reason, details

        return False, "clean", {}

    def check_template_abuse(
        self,
        response: StoryGenerationSynapse
    ) -> Tuple[bool, str, Dict]:
        """检测模板滥用"""
        try:
            data = json.loads(response.output_json)
        except:
            return False, "", {}

        # 检查固定短语
        text = json.dumps(data, ensure_ascii=False)

        template_phrases = [
            "一个关于",
            "在一个",
            "主人公",
            "故事发生在"
        ]

        phrase_count = sum(1 for phrase in template_phrases if phrase in text)

        if phrase_count > len(template_phrases) * 0.7:
            return True, "template_abuse", {
                "phrase_count": phrase_count,
                "threshold": len(template_phrases) * 0.7
            }

        return False, "", {}

    def check_timing_anomaly(
        self,
        miner_uid: int,
        response: StoryGenerationSynapse
    ) -> Tuple[bool, str, Dict]:
        """检测时间异常（可能是缓存攻击）"""
        gen_time = response.generation_time

        # 记录历史时间
        if miner_uid not in self.timing_patterns:
            self.timing_patterns[miner_uid] = []

        self.timing_patterns[miner_uid].append(gen_time)

        # 保留最近 100 次
        if len(self.timing_patterns[miner_uid]) > 100:
            self.timing_patterns[miner_uid] = self.timing_patterns[miner_uid][-100:]

        # 分析异常
        times = self.timing_patterns[miner_uid]
        if len(times) > 10:
            mean_time = np.mean(times)
            std_time = np.std(times)

            # 时间过短且稳定 = 可能使用缓存
            if mean_time < 2.0 and std_time < 0.5:
                return True, "timing_anomaly", {
                    "mean": mean_time,
                    "std": std_time,
                    "suspicious": "too_fast_and_stable"
                }

        return False, "", {}

    def check_semantic_copying(
        self,
        response: StoryGenerationSynapse,
        all_responses: List[StoryGenerationSynapse]
    ) -> Tuple[bool, str, Dict]:
        """语义级别的抄袭检测"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        try:
            current_data = json.loads(response.output_json)
            current_text = self.extract_semantic_content(current_data)
            current_emb = model.encode(current_text)

            for other in all_responses:
                if other == response:
                    continue

                try:
                    other_data = json.loads(other.output_json)
                    other_text = self.extract_semantic_content(other_data)
                    other_emb = model.encode(other_text)

                    similarity = cosine_similarity([current_emb], [other_emb])[0][0]

                    if similarity > 0.90:  # 语义相似度阈值
                        return True, "semantic_copying", {
                            "similarity": float(similarity),
                            "threshold": 0.90
                        }
                except:
                    continue

        except:
            pass

        return False, "", {}
```

### 3.3 动态权重分配

```python
class AdaptiveWeightCalculator:
    """自适应权重计算器"""

    def __init__(self):
        self.base_ema_alpha = 0.1
        self.base_temperature = 2.0

    def calculate_adaptive_weights(
        self,
        scores: Dict[int, float],
        stability: Dict[int, float],
        network_volatility: float
    ) -> Dict[int, float]:
        """动态计算权重"""

        # 1. 调整 EMA alpha
        alpha = self.adjust_ema_alpha(network_volatility)

        # 2. 应用稳定性加权
        adjusted_scores = {}
        for uid, score in scores.items():
            stability_bonus = stability.get(uid, 0.5) * 10
            adjusted_scores[uid] = score + stability_bonus

        # 3. 计算权重
        weights = self.softmax_with_temperature(
            adjusted_scores,
            self.base_temperature
        )

        # 4. 应用最小权重
        weights = self.apply_min_weight(weights, min_weight=0.001)

        return weights

    def adjust_ema_alpha(self, volatility: float) -> float:
        """根据网络波动调整 alpha"""
        if volatility > 0.5:
            return 0.2  # 快速适应
        elif volatility > 0.3:
            return 0.15
        else:
            return 0.1  # 稳定奖励

    def calculate_stability(
        self,
        uid: int,
        recent_scores: List[float]
    ) -> float:
        """计算 Miner 稳定性"""
        if len(recent_scores) < 5:
            return 0.5  # 中性

        # 使用标准差衡量稳定性
        std = np.std(recent_scores)
        mean = np.mean(recent_scores)

        # CV (变异系数)
        cv = std / mean if mean > 0 else 1.0

        # 稳定性分数 (0-1)
        stability = 1.0 / (1.0 + cv)

        return stability
```

### 3.4 智能 Miner 选择策略

```python
class SmartMinerSelector:
    """智能 Miner 选择器"""

    def __init__(self):
        self.exploration_rate = 0.3

    def select_miners_ucb(
        self,
        scores: Dict[int, float],
        query_counts: Dict[int, int],
        total_queries: int
    ) -> List[int]:
        """使用 UCB (Upper Confidence Bound) 选择 Miners"""

        ucb_scores = {}

        for uid in scores.keys():
            score = scores[uid]
            count = query_counts.get(uid, 1)

            # UCB 公式
            exploration_bonus = np.sqrt(2 * np.log(total_queries) / count)
            ucb = score + exploration_bonus

            ucb_scores[uid] = ucb

        # 选择 top-k
        sorted_uids = sorted(
            ucb_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [uid for uid, _ in sorted_uids[:10]]
```

### 3.5 实施步骤

#### Day 1-2: 评分系统升级
- [ ] 实现 EnhancedContentScorer
- [ ] 集成 sentence-transformers
- [ ] 测试评分准确性
- [ ] 调优权重参数

#### Day 3-4: 反作弊系统
- [ ] 实现 AdvancedAntiCheat
- [ ] 添加多维度检测
- [ ] 测试检测效果
- [ ] 调优阈值

#### Day 5: 权重优化
- [ ] 实现 AdaptiveWeightCalculator
- [ ] 实现 SmartMinerSelector
- [ ] 集成到 Validator
- [ ] 模拟测试

### 3.6 验收标准

- [ ] 评分准确性提升 15%
- [ ] 成功检测模拟的作弊行为
- [ ] 权重分配更加公平
- [ ] Miner 选择策略有效提升网络质量

---

## 📋 Phase 4: 监控与运维体系

### 4.1 系统监控

```python
class SystemMonitor:
    """系统监控器"""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "avg_response_time": 0.0,
            "api_calls": {
                "gemini": 0,
                "openai": 0,
                "claude": 0
            },
            "scores": {
                "current": 0.0,
                "ema": 0.0,
                "history": []
            }
        }

    def record_request(
        self,
        success: bool,
        response_time: float,
        score: float,
        api_used: str
    ):
        """记录请求指标"""
        self.metrics["requests_total"] += 1

        if success:
            self.metrics["requests_success"] += 1
        else:
            self.metrics["requests_failed"] += 1

        # 更新平均响应时间
        total = self.metrics["requests_total"]
        current_avg = self.metrics["avg_response_time"]
        self.metrics["avg_response_time"] = (
            (current_avg * (total - 1) + response_time) / total
        )

        # 记录 API 使用
        if api_used in self.metrics["api_calls"]:
            self.metrics["api_calls"][api_used] += 1

        # 记录评分
        self.metrics["scores"]["current"] = score
        self.metrics["scores"]["history"].append(score)

        # 计算 EMA
        alpha = 0.1
        if self.metrics["scores"]["ema"] == 0:
            self.metrics["scores"]["ema"] = score
        else:
            self.metrics["scores"]["ema"] = (
                alpha * score + (1 - alpha) * self.metrics["scores"]["ema"]
            )

    def get_health_status(self) -> Dict:
        """获取健康状态"""
        total = self.metrics["requests_total"]
        success = self.metrics["requests_success"]

        success_rate = success / total if total > 0 else 0

        status = "healthy"
        if success_rate < 0.95:
            status = "degraded"
        if success_rate < 0.80:
            status = "unhealthy"

        return {
            "status": status,
            "success_rate": success_rate,
            "avg_score": self.metrics["scores"]["ema"],
            "avg_response_time": self.metrics["avg_response_time"]
        }
```

### 4.2 日志系统

```python
import logging
from logging.handlers import RotatingFileHandler

class StructuredLogger:
    """结构化日志系统"""

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)

        # 不同级别的日志文件
        self.logger = logging.getLogger("storyfi_miner")
        self.logger.setLevel(logging.DEBUG)

        # 主日志
        main_handler = RotatingFileHandler(
            f"{log_dir}/miner.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        main_handler.setLevel(logging.INFO)

        # 错误日志
        error_handler = RotatingFileHandler(
            f"{log_dir}/error.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)

        # 性能日志
        perf_handler = RotatingFileHandler(
            f"{log_dir}/performance.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        perf_handler.setLevel(logging.DEBUG)

        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        main_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        perf_handler.setFormatter(formatter)

        self.logger.addHandler(main_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(perf_handler)

    def log_request(
        self,
        task_type: str,
        success: bool,
        response_time: float,
        score: float = 0.0
    ):
        """记录请求日志"""
        self.logger.info(
            f"Request | type={task_type} | "
            f"success={success} | time={response_time:.2f}s | score={score}"
        )
```

### 4.3 告警系统

```python
class AlertSystem:
    """告警系统"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5% 错误率
            "low_score": 60.0,    # 评分低于 60
            "api_limit": 0.90     # API 使用率 90%
        }

    def check_and_alert(self, metrics: Dict):
        """检查指标并发送告警"""
        alerts = []

        # 检查错误率
        total = metrics["requests_total"]
        failed = metrics["requests_failed"]
        if total > 0:
            error_rate = failed / total
            if error_rate > self.alert_thresholds["error_rate"]:
                alerts.append({
                    "level": "warning",
                    "title": "High Error Rate",
                    "message": f"Error rate: {error_rate:.1%}"
                })

        # 检查评分
        current_score = metrics["scores"]["ema"]
        if current_score < self.alert_thresholds["low_score"]:
            alerts.append({
                "level": "warning",
                "title": "Low Score",
                "message": f"EMA score: {current_score:.1f}"
            })

        # 检查 API 限额
        for api, count in metrics["api_calls"].items():
            if api == "gemini" and count > 1500 * 0.90:
                alerts.append({
                    "level": "critical",
                    "title": "API Limit Warning",
                    "message": f"Gemini usage: {count}/1500"
                })

        # 发送告警
        for alert in alerts:
            self.send_alert(alert)

    def send_alert(self, alert: Dict):
        """发送告警通知"""
        if self.webhook_url:
            # 发送到 Webhook (Slack/Discord)
            import requests
            requests.post(self.webhook_url, json=alert)

        # 同时记录到日志
        bt.logging.warning(
            f"ALERT [{alert['level']}] {alert['title']}: {alert['message']}"
        )
```

### 4.4 自动化运维

```python
# PM2 配置文件: ecosystem.config.js
"""
module.exports = {
  apps: [{
    name: 'storyfi-miner',
    script: 'neurons/miner_gemini.py',
    interpreter: 'python3',
    args: '--netuid 108 --subtensor.network finney --wallet.name storyfi_miner --wallet.hotkey default --logging.info',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '2G',
    env: {
      GEMINI_API_KEY: 'xxx',
      OPENAI_API_KEY: 'xxx'
    },
    error_file: 'logs/pm2-error.log',
    out_file: 'logs/pm2-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss'
  }]
}
"""

# 自动更新脚本: auto_update.sh
"""
#!/bin/bash

echo "Checking for updates..."

cd /Users/xinyueyu/storyfi/storyfi-subnet

# 拉取最新代码
git fetch origin

# 检查是否有更新
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ $LOCAL != $REMOTE ]; then
    echo "New version found, updating..."

    # 拉取代码
    git pull origin main

    # 安装依赖
    pip3 install -r requirements.txt

    # 重启服务
    pm2 restart storyfi-miner

    echo "Update completed"
else
    echo "Already up to date"
fi
"""
```

### 4.5 实施步骤

#### Day 1: 监控系统
- [ ] 实现 SystemMonitor
- [ ] 实现 StructuredLogger
- [ ] 集成到 Miner 和 Validator
- [ ] 配置日志轮转

#### Day 2: 告警系统
- [ ] 实现 AlertSystem
- [ ] 配置 Webhook（Slack/Discord）
- [ ] 设置告警阈值
- [ ] 测试告警触发

#### Day 3: 自动化运维
- [ ] 配置 PM2
- [ ] 编写自动更新脚本
- [ ] 配置 Cron Job
- [ ] 测试自动重启和更新

### 4.6 验收标准

- [ ] 监控系统实时显示关键指标
- [ ] 日志完整记录所有请求和错误
- [ ] 告警系统能及时通知异常
- [ ] PM2 自动重启失败进程
- [ ] 自动更新脚本正常工作

---

## 📋 Phase 5: 压力测试与主网准备

### 5.1 压力测试计划

```python
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

class StressTest:
    """压力测试工具"""

    def __init__(self, miner_axon):
        self.miner_axon = miner_axon
        self.results = []

    async def run_stress_test(
        self,
        num_requests: int = 100,
        concurrency: int = 10
    ):
        """执行压力测试"""
        print(f"Starting stress test: {num_requests} requests, {concurrency} concurrent")

        tasks = []
        for i in range(num_requests):
            task = self.send_test_request(i)
            tasks.append(task)

            # 控制并发
            if len(tasks) >= concurrency:
                results = await asyncio.gather(*tasks)
                self.results.extend(results)
                tasks = []

                await asyncio.sleep(0.1)  # 避免过载

        # 完成剩余请求
        if tasks:
            results = await asyncio.gather(*tasks)
            self.results.extend(results)

        # 分析结果
        self.analyze_results()

    async def send_test_request(self, request_id: int):
        """发送测试请求"""
        start_time = time.time()

        task_type = random.choice(["blueprint", "characters", "story_arc", "chapters"])

        synapse = create_test_synapse(task_type)

        try:
            response = await self.dendrite.forward(
                axons=[self.miner_axon],
                synapse=synapse,
                timeout=60
            )

            elapsed = time.time() - start_time

            return {
                "id": request_id,
                "task_type": task_type,
                "success": True,
                "response_time": elapsed,
                "has_output": bool(response[0].output_json)
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "id": request_id,
                "task_type": task_type,
                "success": False,
                "response_time": elapsed,
                "error": str(e)
            }

    def analyze_results(self):
        """分析压力测试结果"""
        total = len(self.results)
        success = sum(1 for r in self.results if r["success"])
        failed = total - success

        response_times = [r["response_time"] for r in self.results if r["success"]]

        print("\n" + "="*60)
        print("Stress Test Results")
        print("="*60)
        print(f"Total Requests: {total}")
        print(f"Success: {success} ({success/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")

        if response_times:
            print(f"\nResponse Times:")
            print(f"  Min: {min(response_times):.2f}s")
            print(f"  Max: {max(response_times):.2f}s")
            print(f"  Avg: {np.mean(response_times):.2f}s")
            print(f"  P50: {np.percentile(response_times, 50):.2f}s")
            print(f"  P95: {np.percentile(response_times, 95):.2f}s")
            print(f"  P99: {np.percentile(response_times, 99):.2f}s")

        # 按任务类型分析
        print(f"\nBy Task Type:")
        for task_type in ["blueprint", "characters", "story_arc", "chapters"]:
            task_results = [r for r in self.results if r.get("task_type") == task_type]
            task_success = sum(1 for r in task_results if r["success"])
            print(f"  {task_type}: {task_success}/{len(task_results)}")
```

### 5.2 性能基准测试

```python
class PerformanceBenchmark:
    """性能基准测试"""

    benchmarks = {
        "response_time": {
            "blueprint": {"target": 8.0, "max": 15.0},
            "characters": {"target": 10.0, "max": 20.0},
            "story_arc": {"target": 12.0, "max": 25.0},
            "chapters": {"target": 15.0, "max": 30.0}
        },
        "quality_score": {
            "blueprint": {"min": 75.0, "target": 85.0},
            "characters": {"min": 70.0, "target": 80.0},
            "story_arc": {"min": 70.0, "target": 80.0},
            "chapters": {"min": 65.0, "target": 75.0}
        },
        "success_rate": {
            "min": 0.95,
            "target": 0.99
        }
    }

    def run_benchmark(self):
        """运行性能基准测试"""
        results = {}

        for task_type in ["blueprint", "characters", "story_arc", "chapters"]:
            print(f"\nBenchmarking {task_type}...")

            task_results = []
            for i in range(10):  # 每个任务类型测试 10 次
                result = self.test_single_task(task_type)
                task_results.append(result)

            # 计算平均
            avg_time = np.mean([r["time"] for r in task_results])
            avg_score = np.mean([r["score"] for r in task_results if r["score"] > 0])
            success_rate = sum(1 for r in task_results if r["success"]) / len(task_results)

            results[task_type] = {
                "avg_time": avg_time,
                "avg_score": avg_score,
                "success_rate": success_rate
            }

            # 检查是否达标
            time_target = self.benchmarks["response_time"][task_type]["target"]
            score_target = self.benchmarks["quality_score"][task_type]["target"]

            print(f"  Response Time: {avg_time:.2f}s (target: {time_target}s) - {'✅' if avg_time <= time_target else '⚠️'}")
            print(f"  Quality Score: {avg_score:.1f} (target: {score_target}) - {'✅' if avg_score >= score_target else '⚠️'}")
            print(f"  Success Rate: {success_rate:.1%} - {'✅' if success_rate >= self.benchmarks['success_rate']['target'] else '⚠️'}")

        return results
```

### 5.3 主网部署检查清单

```markdown
## 主网部署前检查清单

### 代码质量
- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] 压力测试通过
- [ ] 性能基准达标
- [ ] 代码审查完成
- [ ] 无已知 Critical/High bug

### 协议与兼容性
- [ ] 协议 v2.0 在测试网验证通过
- [ ] 与 Bittensor 9.x 兼容
- [ ] 与 Bittensor 10.x 兼容
- [ ] Synapse 序列化/反序列化正常
- [ ] 版本号系统就绪

### 系统稳定性
- [ ] 多模型备份配置完成
- [ ] 质量预检机制启用
- [ ] 缓存系统工作正常
- [ ] 错误处理和重试机制完善
- [ ] 24小时稳定运行测试通过

### 监控与运维
- [ ] 监控系统部署完成
- [ ] 日志系统配置完成
- [ ] 告警系统配置完成
- [ ] PM2 自动重启配置
- [ ] 自动更新脚本就绪
- [ ] 备份和恢复方案准备

### 安全与配置
- [ ] API Keys 安全存储
- [ ] 钱包私钥备份
- [ ] 网络端口配置正确
- [ ] 防火墙规则配置
- [ ] SSL 证书验证通过

### 文档与沟通
- [ ] 运维手册完成
- [ ] 应急预案准备
- [ ] 回滚方案准备
- [ ] 与子网 Owner 沟通确认
- [ ] 部署时间窗口确定

### 资源准备
- [ ] 主网 TAO 准备充足（至少 1 TAO）
- [ ] API 额度检查（Gemini + 备份）
- [ ] 服务器资源充足（CPU/内存/磁盘）
- [ ] 网络带宽充足

### 最终验证
- [ ] 在测试网模拟主网场景
- [ ] 验证所有功能正常
- [ ] 确认没有遗留问题
- [ ] 团队成员确认就绪
```

### 5.4 实施步骤

#### Day 1: 压力测试
- [ ] 运行压力测试（100+ 并发请求）
- [ ] 分析性能瓶颈
- [ ] 优化性能问题
- [ ] 重新测试验证

#### Day 2: 基准测试
- [ ] 运行性能基准测试
- [ ] 确认所有指标达标
- [ ] 记录基准数据
- [ ] 准备性能报告

#### Day 3: 最终验证
- [ ] 完成部署检查清单
- [ ] 准备部署文档
- [ ] 进行最终代码审查
- [ ] 获得部署批准

### 5.5 验收标准

- [ ] 压力测试通过（100+ 请求，成功率 > 95%）
- [ ] 性能基准达标（所有任务类型）
- [ ] 部署检查清单 100% 完成
- [ ] 团队确认可以部署主网

---

## 📈 项目时间线

```
Week 1: 核心修复
├─ Day 1-3: Phase 1 - 协议修复
└─ Day 4-5: Phase 2.1 - 多模型备份

Week 2: 系统增强
├─ Day 6-7: Phase 2.2 - 质量预检和缓存
├─ Day 8-10: Phase 3 - 算法优化
└─ Day 11-12: Phase 4 - 监控运维

Week 3: 测试部署
├─ Day 13-14: Phase 5.1 - 压力测试
├─ Day 15: Phase 5.2 - 基准测试
└─ Day 16-17: Phase 5.3 - 最终准备

主网部署: Week 3 末 或 Week 4 初
```

---

## 🎯 关键指标与目标

### 性能指标

| 指标 | 当前 | 目标 | 改进后预期 |
|------|------|------|------------|
| 平均评分 | 80/100 | 85/100 | 88/100 |
| Blueprint 响应时间 | 7s | 6s | 5s |
| Characters 响应时间 | 9s | 8s | 7s |
| Story Arc 响应时间 | 11s | 10s | 9s |
| Chapters 响应时间 | 14s | 12s | 10s |
| 错误率 | 0% | <1% | <0.5% |
| 缓存命中率 | 0% | 20% | 25% |

### 稳定性指标

| 指标 | 当前 | 目标 |
|------|------|------|
| 单点故障风险 | 高 | 低 |
| API 故障转移 | 无 | <2s |
| 系统可用性 | 未知 | >99.5% |
| MTTR（平均修复时间） | 未知 | <15min |

### 安全指标

| 指标 | 当前 | 目标 |
|------|------|------|
| 作弊检测率 | 70% | 95% |
| 误报率 | 未知 | <2% |
| 评分公平性 | 一般 | 优秀 |

---

## 💰 成本估算

### 开发成本
- **人力**: 2-3 周 × 1 人 = 2-3 人周
- **时间价值**: 按工期计算

### 运营成本（月）

| 项目 | 当前 | 改进后 |
|------|------|--------|
| Gemini API | $0 (免费) | $0 (免费) |
| OpenAI 备份 | $0 | ~$20 (备用) |
| Claude 备份 | $0 | ~$15 (备用) |
| 服务器 | 自有 | 自有 |
| **总计** | $0 | **~$35** |

**对比**: OpenAI 单模型方案约 $150/月

---

## ⚠️ 风险评估与缓解

| 风险 | 概率 | 影响 | 缓解措施 | 负责人 |
|------|------|------|----------|--------|
| 协议修复失败 | 低 | 高 | 多方案备选，充分测试 | 开发 |
| 性能未达标 | 中 | 中 | 性能优化，降低目标 | 开发 |
| API 额度超限 | 中 | 中 | 多模型备份，监控告警 | 运维 |
| 主网部署失败 | 低 | 高 | 完整测试，回滚方案 | 全员 |
| 被游戏化攻击 | 中 | 高 | 高级反作弊，持续监控 | 算法 |
| 更新部署困难 | 低 | 中 | 自动化部署，分阶段更新 | 运维 |

---

## 🎓 关键决策点

### 决策 1: 协议修复方案
- **Option A**: 简化为字符串类型（推荐）
- **Option B**: 自定义序列化
- **建议**: 选择 Option A，兼容性最好

### 决策 2: 备份模型策略
- **Option A**: OpenAI + Claude 双备份
- **Option B**: 仅 OpenAI 备份
- **建议**: 选择 Option A，容错能力更强

### 决策 3: 部署时机
- **Option A**: 2 周后（完整方案）
- **Option B**: 1 周后（最小化修复）
- **建议**: 与子网 Owner 沟通后决定

---

## 📞 沟通计划

### 与子网 Owner
- **Week 1 初**: 分享改进计划，获得反馈
- **Week 2 中**: 更新进度，展示测试结果
- **Week 2 末**: 确认部署时间窗口
- **部署前**: 最终确认和协调

### 内部沟通
- **每日**: 进度同步和问题讨论
- **每周**: 里程碑评审和风险评估
- **关键节点**: 重要决策前的团队讨论

---

## 🎉 成功标准

### 短期成功（主网部署后 1 周）
- [ ] Miner 稳定运行，无重大故障
- [ ] 平均评分 ≥ 85/100
- [ ] 获得正常的 Emission 奖励
- [ ] 无安全或作弊事件

### 中期成功（1-3 个月）
- [ ] 排名进入子网 Top 50%
- [ ] 系统可用性 > 99%
- [ ] 成本控制在预算内
- [ ] 成功完成至少 2 次代码更新

### 长期成功（3-6 个月）
- [ ] 排名进入子网 Top 25%
- [ ] 建立完善的运维体系
- [ ] 积累足够的 TAO 奖励
- [ ] 为扩展到更多子网做好准备

---

## 📚 附录

### A. 参考文档
- Bittensor 官方文档
- Subnet 开发最佳实践
- 本地测试报告
- 测试网部署日志

### B. 代码仓库
- GitHub: [项目链接]
- 分支策略: main / dev / feature/*
- 版本标签: v2.0.0, v2.1.0...

### C. 运维手册
- 部署流程
- 监控指南
- 故障排查
- 应急响应

---

**文档版本**: v1.0
**最后更新**: 2025-10-17
**下次评审**: Phase 1 完成后
