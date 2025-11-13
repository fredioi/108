# StoryFi Miner Architecture Redesign - 正确的去中心化实现

## 🚨 当前问题

### 错误的设计 (neurons/miner_gemini.py)
```python
# ❌ 所有矿工强制使用 Gemini API
self.model = genai.GenerativeModel("gemini-2.5-flash")

# 问题:
1. 不是真正的去中心化（依赖Google单点）
2. 所有矿工用同一个模型→响应高度相似
3. 矿工必须付费给Google
4. Google故障→所有矿工失败
5. 不符合Bittensor设计理念
```

---

## ✅ 正确的设计 - 参考Top 5子网

### 研究成果总结

| 子网 | 矿工方式 | 硬件要求 | 去中心化程度 |
|------|---------|---------|-------------|
| SN1 (Text Prompting) | ❌ 使用OpenAI API | 1 vCPU | ⭐ 低（依赖OpenAI） |
| SN19 (Vision) | ✅ 本地GPU运行开源模型 | A100 80GB | ⭐⭐⭐⭐⭐ 高 |
| SN27 (Compute) | ✅ 提供GPU算力 | H100/A100 | ⭐⭐⭐⭐⭐ 高 |
| SN64 (Chutes) | ✅ 本地推理 | A100+ | ⭐⭐⭐⭐⭐ 高 |
| SN34 (BitMind) | ✅ 本地训练/推理 | Consumer GPU+ | ⭐⭐⭐⭐ 高 |

**结论**: 成功的子网都使用**本地GPU运行开源模型**，而非依赖API。

---

## 🎯 新架构设计

### Phase 2.0: 混合架构（本地优先 + API备份）

```
┌─────────────────────────────────────────┐
│         StoryFi Miner v2.0              │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   Model Loader (智能选择)        │  │
│  └───────────┬──────────────────────┘  │
│              │                          │
│      ┌───────┴───────┐                 │
│      │               │                 │
│  ┌───▼────┐    ┌────▼─────┐           │
│  │ Local  │    │   API    │           │
│  │ GPU    │    │ Fallback │           │
│  │ Models │    │ (Gemini) │           │
│  └────────┘    └──────────┘           │
│  优先级1        优先级2                │
│  奖励 1.0x      奖励 0.5x              │
└─────────────────────────────────────────┘
```

### 核心原则

1. **本地优先**: 优先使用矿工自己的GPU和模型
2. **模型多样性**: 支持多种开源模型（Llama, Mixtral, Qwen等）
3. **API备份**: 没有GPU的矿工可以降级使用API（但奖励减半）
4. **公平奖励**: 根据硬件投入和去中心化贡献调整奖励

---

## 📦 支持的模型列表

### Tier 1: 高性能模型（80GB+ VRAM）

```yaml
llama-3-70b-instruct:
  repo: "meta-llama/Meta-Llama-3-70B-Instruct"
  vram: 70GB
  reward_multiplier: 1.5x
  quality: 最高

mixtral-8x7b-instruct:
  repo: "mistralai/Mixtral-8x7B-Instruct-v0.1"
  vram: 45GB
  reward_multiplier: 1.3x
  quality: 很高

qwen-72b-chat:
  repo: "Qwen/Qwen-72B-Chat"
  vram: 72GB
  reward_multiplier: 1.4x
  quality: 很高
```

### Tier 2: 中等模型（24-40GB VRAM）

```yaml
llama-3-8b-instruct:
  repo: "meta-llama/Meta-Llama-3-8B-Instruct"
  vram: 16GB
  reward_multiplier: 1.0x
  quality: 良好

mistral-7b-instruct:
  repo: "mistralai/Mistral-7B-Instruct-v0.2"
  vram: 14GB
  reward_multiplier: 1.0x
  quality: 良好

yi-34b-chat:
  repo: "01-ai/Yi-34B-Chat"
  vram: 34GB
  reward_multiplier: 1.2x
  quality: 很好
```

### Tier 3: 轻量模型（<16GB VRAM）

```yaml
phi-3-medium:
  repo: "microsoft/Phi-3-medium-4k-instruct"
  vram: 8GB
  reward_multiplier: 0.8x
  quality: 中等

gemma-7b-it:
  repo: "google/gemma-7b-it"
  vram: 14GB
  reward_multiplier: 0.9x
  quality: 良好
```

### Tier 4: API备份（无GPU）

```yaml
gemini-2.5-flash:
  provider: Google
  cost: 按调用付费
  reward_multiplier: 0.5x
  quality: 很高
  note: 仅作备份，不鼓励使用

gpt-4-turbo:
  provider: OpenAI
  cost: 按调用付费
  reward_multiplier: 0.5x
  quality: 最高
  note: 仅作备份，不鼓励使用
```

---

## 💻 新Miner实现

### neurons/miner_v2.py

```python
"""
StoryFi Miner v2.0 - 本地GPU优先架构
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import bittensor as bt
import os

class ModelConfig:
    """模型配置"""
    SUPPORTED_MODELS = {
        # Tier 1: 高性能模型
        "llama-3-70b": {
            "repo": "meta-llama/Meta-Llama-3-70B-Instruct",
            "vram_gb": 70,
            "reward_multiplier": 1.5,
            "load_in_8bit": False,
            "load_in_4bit": True  # 使用4bit量化降低VRAM需求
        },
        "mixtral-8x7b": {
            "repo": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "vram_gb": 45,
            "reward_multiplier": 1.3,
            "load_in_8bit": True,
            "load_in_4bit": False
        },

        # Tier 2: 中等模型
        "llama-3-8b": {
            "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
            "vram_gb": 16,
            "reward_multiplier": 1.0,
            "load_in_8bit": True,
            "load_in_4bit": False
        },
        "mistral-7b": {
            "repo": "mistralai/Mistral-7B-Instruct-v0.2",
            "vram_gb": 14,
            "reward_multiplier": 1.0,
            "load_in_8bit": True,
            "load_in_4bit": False
        },

        # Tier 3: 轻量模型
        "phi-3-medium": {
            "repo": "microsoft/Phi-3-medium-4k-instruct",
            "vram_gb": 8,
            "reward_multiplier": 0.8,
            "load_in_8bit": True,
            "load_in_4bit": False
        }
    }


class LocalModelLoader:
    """本地模型加载器"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = ModelConfig.SUPPORTED_MODELS.get(model_name)

        if not self.config:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """加载模型到GPU"""
        bt.logging.info(f"Loading model: {self.config['repo']}")
        bt.logging.info(f"Target device: {self.device}")
        bt.logging.info(f"Required VRAM: {self.config['vram_gb']}GB")

        # 检查GPU内存
        if self.device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            bt.logging.info(f"Available GPU memory: {gpu_memory:.1f}GB")

            if gpu_memory < self.config['vram_gb'] * 0.8:
                bt.logging.warning(
                    f"GPU memory may be insufficient. "
                    f"Required: {self.config['vram_gb']}GB, "
                    f"Available: {gpu_memory:.1f}GB"
                )

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['repo'],
            trust_remote_code=True
        )

        # 加载模型（根据配置选择量化方式）
        load_kwargs = {
            "pretrained_model_name_or_path": self.config['repo'],
            "device_map": "auto",  # 自动分配到多GPU
            "torch_dtype": torch.float16,  # 使用半精度
            "trust_remote_code": True
        }

        # 量化选项
        if self.config.get('load_in_4bit'):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            bt.logging.info("Using 4-bit quantization")

        elif self.config.get('load_in_8bit'):
            load_kwargs["load_in_8bit"] = True
            bt.logging.info("Using 8-bit quantization")

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        bt.logging.success(f"✅ Model loaded successfully")
        return self

    def generate(self, prompt: str, max_new_tokens: int = 2000) -> str:
        """生成文本"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # 移除原始prompt
        response = generated_text[len(prompt):].strip()
        return response


class APIFallbackLoader:
    """API备份加载器（用于没有GPU的矿工）"""

    def __init__(self, provider: str = "gemini"):
        self.provider = provider

        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-2.5-flash")

        elif provider == "openai":
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = openai.ChatCompletion

        else:
            raise ValueError(f"Unsupported API provider: {provider}")

        bt.logging.warning("⚠️  Using API fallback mode (0.5x rewards)")

    def generate(self, prompt: str) -> str:
        """通过API生成文本"""
        if self.provider == "gemini":
            response = self.model.generate_content(prompt)
            return response.text

        elif self.provider == "openai":
            response = self.model.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content


class StoryFiMinerV2:
    """StoryFi Miner v2.0 - 智能模型选择"""

    def __init__(self, config=None):
        self.config = config or get_config()

        # 初始化Bittensor组件
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

        # 智能选择模型
        self.model_loader = self.initialize_model()
        self.mode = self.model_loader.__class__.__name__

        # 设置Axon
        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        bt.logging.info(f"✅ Miner initialized in {self.mode} mode")

    def initialize_model(self):
        """智能选择最佳模型"""

        # 1. 检查是否有GPU
        if not torch.cuda.is_available():
            bt.logging.warning("No CUDA GPU detected. Falling back to API mode.")
            return APIFallbackLoader(provider="gemini")

        # 2. 检查GPU内存
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        bt.logging.info(f"Detected GPU memory: {gpu_memory_gb:.1f}GB")

        # 3. 从配置或环境变量读取模型选择
        preferred_model = self.config.model_name or os.getenv("MODEL_NAME", "auto")

        if preferred_model == "auto":
            # 自动选择最适合的模型
            if gpu_memory_gb >= 70:
                selected_model = "llama-3-70b"
            elif gpu_memory_gb >= 45:
                selected_model = "mixtral-8x7b"
            elif gpu_memory_gb >= 16:
                selected_model = "llama-3-8b"
            elif gpu_memory_gb >= 8:
                selected_model = "phi-3-medium"
            else:
                bt.logging.warning(
                    f"GPU memory ({gpu_memory_gb:.1f}GB) insufficient for any local model. "
                    "Falling back to API mode."
                )
                return APIFallbackLoader(provider="gemini")
        else:
            selected_model = preferred_model

        # 4. 加载本地模型
        try:
            bt.logging.info(f"Attempting to load local model: {selected_model}")
            loader = LocalModelLoader(selected_model)
            loader.load()

            # 报告奖励倍数
            reward_multiplier = loader.config['reward_multiplier']
            bt.logging.success(
                f"✅ Local model loaded successfully. "
                f"Reward multiplier: {reward_multiplier}x"
            )

            return loader

        except Exception as e:
            bt.logging.error(f"Failed to load local model: {e}")
            bt.logging.warning("Falling back to API mode")
            return APIFallbackLoader(provider="gemini")

    async def generate_blueprint(self, synapse):
        """生成故事蓝图"""
        start_time = time.time()

        prompt = self.build_blueprint_prompt(synapse.user_input)

        # 使用加载的模型生成
        response_text = self.model_loader.generate(prompt)

        # 解析JSON
        try:
            output_data = json.loads(response_text)
        except json.JSONDecodeError:
            # 如果不是纯JSON，尝试提取
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                output_data = json.loads(json_match.group())
            else:
                bt.logging.error("Failed to parse JSON from model output")
                output_data = {"error": "Invalid JSON output"}

        synapse.output_data = output_data
        synapse.generation_time = time.time() - start_time
        synapse.miner_version = "2.0.0"
        synapse.miner_mode = self.mode  # 添加模式信息

        return synapse

    def build_blueprint_prompt(self, user_input: str) -> str:
        """构建Blueprint生成prompt"""
        return f"""你是一个专业的故事创作者。基于用户输入，生成一个详细的故事蓝图。

用户输入: {user_input}

请生成包含以下字段的JSON格式响应:
{{
  "title": "吸引人的故事标题",
  "genre": "故事类型（科幻/奇幻/悬疑等）",
  "setting": "详细的背景设定和世界观（至少300字）",
  "core_conflict": "核心冲突和主要矛盾（至少150字）",
  "themes": ["主题1", "主题2", "主题3"],
  "tone": "叙事基调（严肃/轻松/黑暗等）",
  "target_audience": "目标受众"
}}

只返回JSON，不要其他内容。"""

    # ... 其他生成方法类似 ...


# 配置解析
def get_config():
    parser = argparse.ArgumentParser()

    # Bittensor参数
    parser.add_argument("--netuid", type=int, default=108)
    parser.add_argument("--subtensor.network", type=str, default="test")
    parser.add_argument("--wallet.name", type=str, default="miner")
    parser.add_argument("--wallet.hotkey", type=str, default="default")
    parser.add_argument("--axon.port", type=int, default=8091)

    # 模型选择参数
    parser.add_argument(
        "--model.name",
        type=str,
        default="auto",
        choices=["auto", "llama-3-70b", "llama-3-8b", "mixtral-8x7b",
                 "mistral-7b", "phi-3-medium"],
        help="Model to use. 'auto' will select based on available GPU memory."
    )

    parser.add_argument(
        "--model.fallback_api",
        type=str,
        default="gemini",
        choices=["gemini", "openai"],
        help="API provider to use if local model loading fails"
    )

    return bt.config(parser)


if __name__ == "__main__":
    miner = StoryFiMinerV2()
    asyncio.run(miner.run())
```

---

## 🎯 Validator调整 - 奖励机制

### neurons/validator_v2.py 片段

```python
def calculate_reward_multiplier(self, miner_info: Dict) -> float:
    """
    根据矿工模式调整奖励

    Args:
        miner_info: {
            "mode": "LocalModelLoader" | "APIFallbackLoader",
            "model_name": "llama-3-70b" | "gemini" | ...,
            "gpu_memory_gb": 80 | None
        }

    Returns:
        奖励倍数 (0.5 - 1.5x)
    """

    # 1. API模式：低奖励
    if miner_info["mode"] == "APIFallbackLoader":
        return 0.5

    # 2. 本地模型：根据模型大小
    model_name = miner_info.get("model_name", "unknown")

    # 从ModelConfig获取基础倍数
    base_multiplier = ModelConfig.SUPPORTED_MODELS.get(
        model_name, {}
    ).get("reward_multiplier", 1.0)

    # 3. GPU性能加成（可选）
    gpu_memory = miner_info.get("gpu_memory_gb", 0)
    if gpu_memory >= 80:
        base_multiplier *= 1.1  # H100/A100 10%加成

    return base_multiplier


def score_response_with_multiplier(self, response, task_type: str) -> float:
    """
    评分 + 奖励倍数
    """
    # 1. 基础评分 (0-100)
    base_score = self.score_response(response, task_type)

    # 2. 提取矿工信息
    miner_info = {
        "mode": response.miner_mode,
        "model_name": getattr(response, "model_name", None),
        "gpu_memory_gb": getattr(response, "gpu_memory_gb", None)
    }

    # 3. 计算奖励倍数
    multiplier = self.calculate_reward_multiplier(miner_info)

    # 4. 最终分数
    final_score = base_score * multiplier

    bt.logging.info(
        f"Miner {response.miner_hotkey[:8]}: "
        f"Base={base_score:.1f}, "
        f"Multiplier={multiplier:.2f}x, "
        f"Final={final_score:.1f}"
    )

    return final_score
```

---

## 📋 部署指南

### 对于有GPU的矿工

```bash
# 1. 安装依赖
pip install transformers accelerate bitsandbytes torch

# 2. 设置环境变量（可选，用于手动选择模型）
export MODEL_NAME="llama-3-8b"  # 或 "auto"

# 3. 运行矿工
python neurons/miner_v2.py \
    --netuid 108 \
    --subtensor.network test \
    --wallet.name my_miner \
    --wallet.hotkey default \
    --model.name auto \
    --axon.port 8091
```

### 对于无GPU的矿工（API模式）

```bash
# 1. 设置API Key
export GEMINI_API_KEY=your_key_here

# 2. 运行矿工（会自动检测无GPU并切换到API模式）
python neurons/miner_v2.py \
    --netuid 108 \
    --subtensor.network test \
    --wallet.name my_miner \
    --wallet.hotkey default \
    --model.fallback_api gemini \
    --axon.port 8091

# 注意：API模式奖励仅为本地模式的50%
```

---

## 🔄 迁移计划

### Phase 2.1: 实现新Miner (1周)
- [x] 研究Top 5子网实现
- [ ] 创建 `neurons/miner_v2.py`
- [ ] 实现 `LocalModelLoader`
- [ ] 实现 `APIFallbackLoader`
- [ ] 添加自动模型选择逻辑

### Phase 2.2: 更新Validator (3天)
- [ ] 修改 `neurons/validator.py`
- [ ] 添加奖励倍数计算
- [ ] 更新评分系统

### Phase 2.3: 测试 (1周)
- [ ] 本地测试（不同GPU配置）
- [ ] 测试网部署
- [ ] 24小时稳定性测试
- [ ] 对比不同模型的质量和奖励

### Phase 2.4: 文档和部署 (3天)
- [ ] 更新部署文档
- [ ] 创建模型选择指南
- [ ] 主网部署

---

## 🎓 学到的经验

1. **不要闭门造车**: 研究成功案例比自己摸索更高效
2. **去中心化是核心**: Bittensor的本质是去中心化，依赖单一API违背设计理念
3. **公平激励**: 奖励应该与矿工的实际硬件投入成正比
4. **模型多样性**: 允许矿工选择不同模型→响应多样化→更难作弊
5. **渐进式设计**: 先支持本地+API混合，未来可以完全移除API

---

## 📚 参考资料

- **SN19 Vision**: https://github.com/rayonlabs/vision-workers
- **SN27 Compute**: https://github.com/neuralinternet/compute-subnet
- **SN1 Text Prompting**: https://github.com/opentensor/text-prompting
- **Transformers库**: https://huggingface.co/docs/transformers
- **BitsAndBytes量化**: https://github.com/TimDettmers/bitsandbytes

---

**版本**: 2.0.0
**作者**: StoryFi Team
**日期**: 2025-10-23
**状态**: 设计完成，待实现
