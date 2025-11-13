# SoulX (Subnet 115) 架构分析

**分析日期**: 2025-10-28
**仓库**: https://github.com/SentiVerse-AI/soulx
**任务类型**: NPC对话生成（与StoryFi类似：AI生成文本内容）

---

## 🏗 核心架构

### 两层分离设计

```
┌─────────────────────────────────────────────────────┐
│              Bittensor Miner (Port 8091)            │
│  (soulx/miner/server.py + soulx/miner/miner_server.py) │
│                                                       │
│  - 接收Validator请求                                  │
│  - 转发给Multimodal Server                           │
│  - 返回生成结果                                       │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP Request
                   ↓
┌─────────────────────────────────────────────────────┐
│         Multimodal Server (Port 6919/8000)          │
│         (multimodal_server/main.py)                 │
│                                                       │
│  - 加载本地模型: Qwen/Qwen3-32B                      │
│  - vLLM高效推理                                      │
│  - 模型管理: model_manager.py                        │
│  - 推理逻辑: inference.py                            │
└─────────────────────────────────────────────────────┘
```

### 为什么分离？

**优点**:
1. ✅ **职责清晰**: Bittensor通信 vs 模型推理
2. ✅ **可替换性**: 矿工可以替换Multimodal Server（用不同模型）
3. ✅ **资源隔离**: 模型服务器可以独立扩展
4. ✅ **开发灵活**: 可以单独测试模型服务

**关键点**:
- Bittensor Miner只是"代理"，负责协议通信
- Multimodal Server是"实际工作者"，负责AI生成
- 矿工可以自由选择不同的Multimodal Server实现

---

## 📁 文件结构

```
soulx/
├── soulx/
│   ├── miner/                      # Bittensor矿工层
│   │   ├── server.py               # FastAPI服务（路由）
│   │   ├── miner_server.py         # 启动脚本
│   │   ├── config.py               # 配置管理
│   │   ├── task_config.py          # 任务配置
│   │   └── storage.py              # Redis存储
│   │
│   ├── validator/                  # 验证者（未分析）
│   │
│   └── core/                       # 核心逻辑（未分析）
│
├── multimodal_server/              # 模型服务层
│   ├── main.py                     # FastAPI服务
│   ├── inference.py                # 推理逻辑
│   ├── model_manager.py            # 模型加载管理
│   ├── service_manager.py          # 服务管理
│   ├── start_server.py             # 启动脚本
│   ├── vllm/                       # vLLM集成
│   └── utils/                      # 工具函数
│
├── .env.miner.example              # 矿工配置示例
├── .multimodal_server.example      # 模型服务配置示例
└── requirements.txt
```

---

## 🔧 配置系统

### Miner配置 (.env.miner.example)

```bash
# Bittensor网络配置
SUBTENSOR_NETWORK=finney          # 主网
NETUID=115                         # 子网ID
SUBTENSOR_CHAIN_ENDPOINT=wss://...

# 钱包配置
WALLET_NAME=miner
WALLET_HOTKEY=default

# 矿工类型
MINER_TYPE=TEXT                    # 文本生成任务

# Multimodal Server配置
MULTIMODAL_SERVER_URL=http://localhost:6919

# Redis配置（缓存）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_TTL=3600
```

### Multimodal Server配置 (.multimodal_server.example)

```bash
# 模型配置
MODEL=Qwen/Qwen3-32B               # ⭐ 本地HuggingFace模型

# 服务配置
HOST=localhost
PORT=8000

# vLLM推理配置
MAX_BATCHED_TOKENS=2048
MAX_NUM_SEQS=16
MAX_MODEL_LEN=4096
```

---

## 🔑 关键设计决策

### 1. 使用本地模型（Qwen3-32B）
```python
# multimodal_server/model_manager.py (推测)
from vllm import LLM

class ModelManager:
    def __init__(self, model_name: str):
        self.llm = LLM(
            model=model_name,           # "Qwen/Qwen3-32B"
            max_model_len=4096,
            max_num_seqs=16
        )

    def generate(self, prompt: str):
        outputs = self.llm.generate(prompt)
        return outputs
```

**为什么不用API**:
- ✅ 完全去中心化（不依赖OpenAI/Google）
- ✅ 成本可控（只需GPU硬件成本）
- ✅ 隐私保护（数据不发送给第三方）
- ✅ 定制化（可以fine-tune自己的模型）

### 2. 使用vLLM进行推理
```python
# vLLM优势
- 高吞吐量（PagedAttention）
- 低延迟（CUDA优化）
- 批处理支持（MAX_NUM_SEQS=16）
- 支持量化（节省显存）
```

### 3. 两层架构分离
```python
# soulx/miner/server.py (简化版)
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.post("/generate")
async def generate(request: dict):
    # 转发给Multimodal Server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MULTIMODAL_SERVER_URL}/generate",
            json=request
        )

    return response.json()
```

**好处**:
- Miner可以轻松切换不同的模型服务
- 模型服务可以独立部署、更新
- 符合Bittensor的"矿工自由选择实现"理念

### 4. Redis缓存
```python
# soulx/miner/storage.py (推测)
import redis

class Storage:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379)

    def cache_response(self, key: str, value: str, ttl: int):
        self.redis.setex(key, ttl, value)

    def get_cached(self, key: str):
        return self.redis.get(key)
```

**用途**:
- 缓存常见请求的响应
- 减少重复计算
- 提高响应速度

---

## 🆚 SoulX vs 我们的当前实现

| 维度 | SoulX | StoryFi (当前) | StoryFi (应该) |
|------|-------|----------------|----------------|
| **架构** | 两层分离 | 单体 | 两层分离 ⭐ |
| **模型** | 本地Qwen3-32B | 强制OpenAI API ❌ | 可选多种 |
| **推理** | vLLM | API调用 | vLLM/Transformers |
| **配置** | .env分离 | 硬编码 ❌ | .env配置 |
| **缓存** | Redis | 无 | Redis可选 |
| **可替换性** | ✅ 高 | ❌ 无 | ✅ 高 |
| **去中心化** | ✅ 完全 | ❌ 依赖API | ✅ 完全 |

---

## 💡 对StoryFi的启发

### 应该学习的设计

#### 1. 两层架构
```
StoryFi Miner (neurons/miner.py)
    ↓ HTTP
Story Generator Service (独立服务)
    ├── Local Model (Llama-3.1-8B)
    ├── API Fallback (OpenAI/Gemini)
    └── Custom Script
```

#### 2. 配置文件分离
```bash
# .env.miner
GENERATOR_SERVICE_URL=http://localhost:8000
GENERATOR_MODE=local  # local | api | custom

# .env.generator
MODEL=meta-llama/Llama-3.1-8B-Instruct
DEVICE=cuda
MAX_MODEL_LEN=4096
```

#### 3. 抽象接口
```python
# 矿工只依赖接口，不关心实现
class GeneratorService:
    async def generate(self, input_data: dict) -> dict:
        pass

# 实现可以随意替换
class LocalGenerator(GeneratorService): ...
class APIGenerator(GeneratorService): ...
```

---

## 🚀 推荐实施方案

### Option A: 完全模仿SoulX（最符合Bittensor理念）

```
storyfi-subnet/
├── neurons/
│   └── miner.py                    # Bittensor通信层（轻量）
│
├── story_generator_service/        # 独立模型服务
│   ├── main.py                     # FastAPI服务
│   ├── inference.py                # 推理逻辑
│   ├── model_manager.py            # 模型加载
│   └── start_server.py             # 启动脚本
│
└── .env.generator                  # 生成器配置
```

**启动流程**:
```bash
# 1. 启动生成器服务
python story_generator_service/start_server.py

# 2. 启动矿工
python neurons/miner.py
```

### Option B: 单体但模块化（折中方案）

```
neurons/
├── miner.py                        # 主程序
└── generators/                     # 生成器模块
    ├── base.py                     # 抽象基类
    ├── local_generator.py          # 本地模型
    ├── api_generator.py            # API fallback
    └── loader.py                   # 智能加载器
```

**启动流程**:
```bash
# 单一启动（更简单）
python neurons/miner.py
```

---

## 🎯 建议

### 对于StoryFi，我推荐**Option B（单体但模块化）**

**原因**:
1. ✅ **部署简单**: 只需启动一个进程
2. ✅ **配置统一**: 所有配置在一个.env
3. ✅ **学习曲线低**: 不需要理解两层架构
4. ✅ **灵活性保留**: 通过抽象类实现可替换性

**保留Option A的核心理念**:
- ✅ 抽象接口（Generator基类）
- ✅ 配置化（.env + YAML）
- ✅ 本地模型优先
- ✅ API作为fallback

---

## 📊 实施步骤对比

### SoulX方式（两层）
1. 实现Multimodal Server（1周）
2. 修改Miner调用Server（2天）
3. 配置和测试（3天）

**总时间**: ~10-12天

### 我们的方式（单体模块化）
1. 实现Generator模块（4天）
2. 集成到Miner（1天）
3. 配置和测试（2天）

**总时间**: ~7天

---

## ✅ 结论

**SoulX的核心经验**:
1. ✅ **用本地模型**（Qwen3-32B + vLLM）
2. ✅ **架构分离**（Miner vs Model Service）
3. ✅ **配置化**（.env控制所有行为）
4. ✅ **可替换性**（符合Bittensor理念）

**StoryFi应该采纳**:
1. ✅ 本地模型优先（Llama-3.1-8B）
2. ✅ 配置化设计（.env + YAML）
3. ✅ 抽象接口（Generator基类）
4. ⚠️ 单体架构（部署更简单）

---

**分析人**: Claude Code
**建议**: 采用"单体但模块化"方案，保留两层架构的理念但简化部署
