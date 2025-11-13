# StoryFi矿工架构修复方案

**创建日期**: 2025-10-28
**紧急程度**: 🔴 HIGH（明天上线）
**状态**: 设计阶段

---

## 🎯 目标

修复矿工强制绑定特定API的问题，实现真正的去中心化架构。

---

## 🔍 问题分析

### 当前问题
```python
# neurons/miner.py (当前代码)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")  # ❌ 强制要求
```

### 为什么这违反Bittensor理念
```
Bittensor核心原则：
- 矿工完全自由选择实现方式
- 验证者只评判输出质量
- 不限制使用的工具或方法

违反表现：
- ❌ 强制要求OpenAI API Key
- ❌ 所有矿工用相同模型 → 响应同质化
- ❌ 依赖单一服务商 → 单点故障
- ❌ 矿工需要付费给第三方
```

---

## 🏗 参考架构（基于Bittensor最佳实践）

### 成功案例分析

**BiAgent Subnet (SN20)**:
- ✅ 矿工上传自己的模型到HuggingFace
- ✅ 验证者用vLLM加载矿工的模型评估
- ✅ 完全自由的模型选择

**ByteAI Miner**:
- ✅ 使用vLLM进行本地推理
- ✅ 支持CUDA_VISIBLE_DEVICES指定GPU
- ✅ 异步处理，高效利用资源

**SoulX (SN115)**:
- ✅ 使用Qwen3-32B本地模型
- ✅ HuggingFace Transformers加载
- ✅ GPU加速推理

---

## 💡 推荐方案（3层架构）

### 架构设计
```python
矿工生成器 = ModelLoader → Generator → Responder

ModelLoader支持3种模式：
1. LOCAL (本地GPU) - 推荐，奖励倍数 1.5x
2. API (云端API) - 允许，奖励倍数 0.5x
3. CUSTOM (自定义) - 允许，奖励倍数 1.0x
```

### Layer 1: 配置抽象层
```python
# config/model_config.yaml (新建)
model:
  mode: "local"  # local | api | custom

  # LOCAL模式配置
  local:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    device: "cuda"
    max_memory: "16GB"
    quantization: "4bit"  # 节省显存

  # API模式配置（fallback）
  api:
    provider: "openai"  # openai | anthropic | gemini | custom
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-4o-mini"

  # CUSTOM模式配置
  custom:
    script_path: "./custom/generate.py"
    endpoint: null
```

### Layer 2: 生成器抽象类
```python
# generators/base.py (新建)
from abc import ABC, abstractmethod
from typing import Dict, Optional

class StoryGenerator(ABC):
    """故事生成器基类"""

    @abstractmethod
    async def generate(self, input_data: Dict) -> Dict:
        """
        生成故事内容

        Args:
            input_data: 包含user_input, blueprint等字段

        Returns:
            Dict: 包含generated_content的结果
        """
        pass

    @abstractmethod
    def get_mode(self) -> str:
        """返回生成器模式: local/api/custom"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass
```

### Layer 3: 具体实现

#### A. 本地模型生成器（推荐）
```python
# generators/local_generator.py (新建)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict
from .base import StoryGenerator

class LocalModelGenerator(StoryGenerator):
    """使用本地GPU模型生成（Transformers/vLLM）"""

    def __init__(self, config: Dict):
        self.model_name = config.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_memory = config.get("max_memory", "16GB")
        self.quantization = config.get("quantization", "4bit")

        print(f"🔄 Loading local model: {self.model_name}")
        self._load_model()

    def _load_model(self):
        """加载模型（支持量化）"""
        from transformers import BitsAndBytesConfig

        # 4bit量化配置（节省显存）
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        print(f"✅ Model loaded on {self.device}")

    async def generate(self, input_data: Dict) -> Dict:
        """使用本地模型生成"""
        try:
            # 构建prompt
            prompt = self._build_prompt(input_data)

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True
                )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取生成内容（去掉prompt部分）
            content = generated_text[len(prompt):].strip()

            return {
                "generated_content": content,
                "model": self.model_name,
                "mode": "local"
            }

        except Exception as e:
            print(f"❌ Local generation failed: {e}")
            raise

    def _build_prompt(self, input_data: Dict) -> str:
        """构建生成prompt"""
        user_input = input_data.get("user_input", "")
        blueprint = input_data.get("blueprint", {})

        prompt = f"""You are a creative story writer for an interactive story game.

User Input: {user_input}

Story Blueprint: {blueprint}

Generate engaging story content based on the above. Be creative, maintain consistency, and create compelling narratives.

Story Content:"""

        return prompt

    def get_mode(self) -> str:
        return "local"

    def health_check(self) -> bool:
        """检查模型是否正常"""
        try:
            test_input = {"user_input": "test"}
            result = self.generate(test_input)
            return len(result.get("generated_content", "")) > 0
        except:
            return False
```

#### B. API生成器（Fallback）
```python
# generators/api_generator.py (新建)
import os
from openai import AsyncOpenAI
from typing import Dict
from .base import StoryGenerator

class APIGenerator(StoryGenerator):
    """使用云端API生成（OpenAI/Gemini等）"""

    def __init__(self, config: Dict):
        self.provider = config.get("provider", "openai")
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.model = config.get("model", "gpt-4o-mini")

        # 读取API Key（如果没有就跳过，不强制）
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            print(f"⚠️ Warning: {api_key_env} not found. API mode will not work.")
            self.available = False
        else:
            self.available = True
            self.client = AsyncOpenAI(api_key=self.api_key)
            print(f"✅ API Generator initialized: {self.provider}/{self.model}")

    async def generate(self, input_data: Dict) -> Dict:
        """使用API生成"""
        if not self.available:
            raise RuntimeError("API Generator not available (no API key)")

        try:
            # 构建messages
            messages = self._build_messages(input_data)

            # 调用API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=2048
            )

            content = response.choices[0].message.content

            return {
                "generated_content": content,
                "model": self.model,
                "mode": "api",
                "provider": self.provider
            }

        except Exception as e:
            print(f"❌ API generation failed: {e}")
            raise

    def _build_messages(self, input_data: Dict) -> list:
        """构建OpenAI格式的messages"""
        user_input = input_data.get("user_input", "")
        blueprint = input_data.get("blueprint", {})

        return [
            {
                "role": "system",
                "content": "You are a creative story writer for an interactive story game."
            },
            {
                "role": "user",
                "content": f"User Input: {user_input}\n\nStory Blueprint: {blueprint}\n\nGenerate engaging story content."
            }
        ]

    def get_mode(self) -> str:
        return "api"

    def health_check(self) -> bool:
        return self.available
```

#### C. 自定义生成器
```python
# generators/custom_generator.py (新建)
import subprocess
import json
from typing import Dict
from .base import StoryGenerator

class CustomGenerator(StoryGenerator):
    """使用用户自定义脚本生成"""

    def __init__(self, config: Dict):
        self.script_path = config.get("script_path", "./custom/generate.py")
        print(f"✅ Custom Generator: {self.script_path}")

    async def generate(self, input_data: Dict) -> Dict:
        """调用自定义脚本"""
        try:
            # 将input_data传给脚本
            input_json = json.dumps(input_data)

            # 运行脚本
            result = subprocess.run(
                ["python", self.script_path],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=60
            )

            # 解析输出
            output = json.loads(result.stdout)

            return {
                "generated_content": output.get("content", ""),
                "model": "custom",
                "mode": "custom"
            }

        except Exception as e:
            print(f"❌ Custom generation failed: {e}")
            raise

    def get_mode(self) -> str:
        return "custom"

    def health_check(self) -> bool:
        return True
```

### Layer 4: 智能Generator加载器
```python
# generators/loader.py (新建)
import yaml
from typing import Dict, Optional
from .base import StoryGenerator
from .local_generator import LocalModelGenerator
from .api_generator import APIGenerator
from .custom_generator import CustomGenerator

class GeneratorLoader:
    """智能加载器：自动选择可用的生成器"""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.mode = self.config['model']['mode']
        self.generator: Optional[StoryGenerator] = None

        self._load_generator()

    def _load_generator(self):
        """按优先级加载生成器"""
        mode = self.mode

        # 尝试加载指定模式
        if mode == "local":
            try:
                config = self.config['model']['local']
                self.generator = LocalModelGenerator(config)
                print(f"✅ Loaded LOCAL generator")
                return
            except Exception as e:
                print(f"⚠️ Failed to load local generator: {e}")
                print(f"🔄 Falling back to API...")
                mode = "api"

        if mode == "api":
            try:
                config = self.config['model']['api']
                gen = APIGenerator(config)
                if gen.health_check():
                    self.generator = gen
                    print(f"✅ Loaded API generator")
                    return
                else:
                    print(f"⚠️ API not available (no key)")
            except Exception as e:
                print(f"⚠️ Failed to load API generator: {e}")

        if mode == "custom":
            try:
                config = self.config['model']['custom']
                self.generator = CustomGenerator(config)
                print(f"✅ Loaded CUSTOM generator")
                return
            except Exception as e:
                print(f"⚠️ Failed to load custom generator: {e}")

        # 如果都失败了
        raise RuntimeError("❌ No generator available! Please configure at least one mode.")

    async def generate(self, input_data: Dict) -> Dict:
        """生成内容"""
        if not self.generator:
            raise RuntimeError("No generator loaded")

        return await self.generator.generate(input_data)

    def get_mode(self) -> str:
        """获取当前模式"""
        return self.generator.get_mode() if self.generator else "none"
```

---

## 🔧 修改neurons/miner.py

### 最小改动方案
```python
# neurons/miner.py (修改)
from generators.loader import GeneratorLoader

class StoryFiMiner:
    def __init__(self, config=None):
        # ... 其他初始化代码 ...

        # ✅ 新代码：使用GeneratorLoader替代硬编码OpenAI
        self.generator = GeneratorLoader()  # 自动选择最佳生成器

        print(f"🚀 Miner started with {self.generator.get_mode()} mode")

    async def forward(self, synapse: StoryGenerationSynapse) -> StoryGenerationSynapse:
        """处理验证者请求"""
        try:
            # 构建input_data
            input_data = {
                "user_input": synapse.user_input,
                "blueprint": synapse.blueprint,
                "characters": synapse.characters,
                "story_arc": synapse.story_arc,
                "chapter_ids": synapse.chapter_ids
            }

            # ✅ 使用通用生成器
            result = await self.generator.generate(input_data)

            # 填充响应（Protocol v3.1.0格式）
            synapse.output_data = result
            synapse.generation_time = t.elapsed
            synapse.miner_version = "1.1.0"  # 更新版本号

            return synapse

        except Exception as e:
            bt.logging.error(f"Generation failed: {e}")
            synapse.output_data = {"error": str(e)}
            return synapse
```

---

## 📊 验证者改动（可选）

### 添加模式识别奖励
```python
# neurons/validator.py (可选改动)
def calculate_composite_score(self, uid: int, quality_score: float) -> float:
    """计算综合分数（质量+模式）"""

    # 获取矿工模式
    mode = self.get_miner_mode(uid)  # 从output_data读取

    # 模式奖励倍数
    mode_multipliers = {
        "local": 1.5,    # 本地模型：奖励最高
        "custom": 1.0,   # 自定义：标准奖励
        "api": 0.5       # API：奖励减半
    }

    multiplier = mode_multipliers.get(mode, 1.0)

    # 综合分数 = 质量分数 × 模式倍数
    return quality_score * multiplier
```

---

## 🚀 实施计划

### Option 1: 紧急方案（明天上线用）
**时间**: 2-3小时

1. ✅ **只实现API Generator（0.5小时）**
   - 改成可选的API（有key就用，没有就报错）
   - 至少不是"强制"了

2. ✅ **修改miner.py使用Generator（0.5小时）**
   - 用GeneratorLoader替换硬编码OpenAI
   - 保持Protocol兼容

3. ✅ **测试（1小时）**
   - 测试有API key的情况
   - 测试没有API key的报错是否清晰

4. ✅ **文档更新（1小时）**
   - 更新DEPLOYMENT_GUIDE.md
   - 说明API key是"可选"的

**结果**:
- ✅ 不再"强制"API
- ✅ 明确告知矿工可以用其他方式
- ⚠️ 但实际上还没有本地模型实现

---

### Option 2: 正确方案（2周后v1.1）
**时间**: 1-2周

**Week 1: 核心实现**
- Day 1-2: 实现LocalModelGenerator（支持Llama-3.1-8B）
- Day 3-4: 实现GeneratorLoader和配置系统
- Day 5: 测试和优化

**Week 2: 集成和部署**
- Day 1-2: 修改miner.py集成
- Day 3: 更新validator.py添加模式奖励
- Day 4-5: 测试网部署和测试
- Day 6-7: 文档和示例

---

## 📝 配置示例

### 本地模型配置
```yaml
# config/model_config.yaml
model:
  mode: "local"

  local:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    device: "cuda"
    max_memory: "16GB"
    quantization: "4bit"
```

### API Fallback配置
```yaml
# config/model_config.yaml
model:
  mode: "api"

  api:
    provider: "openai"
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-4o-mini"
```

---

## ✅ 成功标准

### 技术指标
- [ ] 矿工可以用本地GPU模型
- [ ] 矿工可以用任何API（可选）
- [ ] 矿工可以用自定义脚本
- [ ] 验证者能识别矿工模式
- [ ] 模式奖励正确计算

### 去中心化指标
- [ ] 不强制任何特定API
- [ ] 不限制模型选择
- [ ] 配置清晰易懂
- [ ] 有完整的文档和示例

---

## 🎯 推荐决策

### 明天上线
**使用Option 1（紧急方案）**:
- 改成"建议使用API"而不是"强制"
- 在文档中声明将在v1.1支持本地模型
- 至少理念上不违反去中心化

### 2周后v1.1
**实施Option 2（正确方案）**:
- 完整实现3层生成器架构
- 真正支持本地模型
- 实现模式奖励机制

---

## 📞 需要的资源

### 硬件（本地模型）
- GPU: 至少16GB VRAM (RTX 4090 / A100)
- 存储: 20GB（模型文件）

### 软件依赖
```bash
pip install transformers accelerate bitsandbytes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

**创建人**: Claude Code
**状态**: 等待决策
**建议**: Option 1明天上线，Option 2尽快实施
