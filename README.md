# StoryFi Bittensor Subnet

> **分布式AI故事生成子网** - 基于 Bittensor 的去中心化故事创作协议

## 📖 项目概述

StoryFi Subnet 是一个运行在 Bittensor 网络上的去中心化AI故事生成子网。Miners 运行 AI 模型生成高质量故事内容，Validators 评估质量并分配奖励。

### 核心特性

- ✅ **4阶段流水线**：Blueprint → Characters → Story Arc → Chapters
- ✅ **客观评分系统**：100分制（技术30% + 结构40% + 内容30%）
- ✅ **反作弊机制**：抄袭检测、黑名单、相似度分析
- ✅ **完整 Protocol**：基于 Pydantic 的类型安全通信协议

### 项目状态

当前版本：**v2.0.0-alpha**

- ✅ 激励机制设计完成
- ✅ Protocol 通信协议完成
- ✅ Miner 实现完成（支持灵活生成后端）
- ⏳ Validator 实现中

### 新特性 (v2.0.0)

- ✨ **灵活生成后端**：支持本地GPU模型、API、自定义实现
- ✨ **奖励乘数系统**：本地模型1.5x、API 0.5x、自定义1.0x
- ✨ **智能回退链**：本地 → API → 错误
- ✨ **4bit量化支持**：降低75% VRAM使用
- ✨ **配置化设计**：无需修改代码即可切换生成方法

## 🏗️ 架构

```
┌─────────────────────────────────────────────────┐
│                  Bittensor Network              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────┐         Query          ┌────────┐│
│  │Validator │ ────────────────────────→ Miner  ││
│  │          │                         │        ││
│  │ - 生成任务  │ ←────────────────────── │ - AI生成││
│  │ - 评分     │       Response         │ - OpenAI││
│  │ - 分配权重  │                         │ - Claude││
│  └──────────┘                         └────────┘│
│       ↓                                    ↑     │
│   set_weights()                       Emission  │
│       ↓                                    ↑     │
│  ┌────────────────────────────────────────────┐ │
│  │      Subtensor (Blockchain)               │ │
│  │      - Weight storage                     │ │
│  │      - TAO emission                       │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/your-org/storyfi-subnet.git
cd storyfi-subnet

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Bittensor
pip install bittensor
```

### 2. 配置生成器

StoryFi v2.0.0 支持三种生成方式：

#### 选项A：本地GPU模型（推荐，1.5x奖励）

```bash
# 配置生成器
cp config/generator_config.yaml.example config/generator_config.yaml

# 编辑配置，设置 mode: "local"
nano config/generator_config.yaml

# 安装本地模型依赖
pip install transformers accelerate bitsandbytes
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**要求**：6GB+ VRAM GPU

#### 选项B：API模式（快速开始，0.5x奖励）

```bash
# 配置生成器
cp config/generator_config.yaml.example config/generator_config.yaml

# 编辑配置，设置 mode: "api"
nano config/generator_config.yaml

# 设置API密钥
export OPENAI_API_KEY=sk-...
# 或
export GEMINI_API_KEY=your-key-here
```

#### 选项C：自定义实现（1.0x奖励）

查看 [Generator System 文档](./docs/GENERATOR_SYSTEM.md) 了解详情。

### 3. 配置环境变量

```bash
# 创建 .env 文件
cp .env.example .env

# 编辑配置
nano .env
```

### 4. 运行 Miner

```bash
python neurons/miner.py \
    --netuid 42 \
    --wallet.name my_miner \
    --wallet.hotkey default \
    --logging.info
```

矿工启动时会显示：
```
✅ Generator Mode: local
✅ Model: meta-llama/Llama-3.1-8B-Instruct
```

### 5. 运行 Validator

```bash
python neurons/validator.py \
    --netuid 42 \
    --wallet.name my_validator \
    --wallet.hotkey default \
    --logging.info
```

## 📦 项目结构

```
storyfi-subnet/
├── neurons/
│   ├── miner.py              # Miner 主程序
│   └── validator.py          # Validator 主程序
├── generators/               # 🆕 生成器模块
│   ├── __init__.py
│   ├── base.py               # 抽象基类
│   ├── local_generator.py    # 本地GPU生成器
│   ├── api_generator.py      # API生成器
│   └── loader.py             # 智能加载器
├── config/                   # 🆕 配置目录
│   ├── generator_config.yaml # 生成器配置
│   └── generator_config.yaml.example
├── template/
│   ├── __init__.py
│   ├── protocol.py           # Synapse 通信协议
│   ├── utils.py              # 工具函数
│   └── reward.py             # 评分算法（待实现）
├── scoring/
│   ├── technical.py          # Technical Score（待实现）
│   ├── structure.py          # Structure Score（待实现）
│   └── content.py            # Content Score（待实现）
├── tests/
│   ├── test_protocol.py
│   ├── test_scoring.py
│   └── test_e2e.py
├── docs/
│   ├── GENERATOR_SYSTEM.md       # 🆕 生成器系统文档
│   ├── INCENTIVE_MECHANISM.md    # 激励机制设计文档
│   ├── PROTOCOL_EXAMPLE.md       # Protocol 使用示例
│   └── PROJECT_PLAN.md           # 完整项目计划
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## 📚 文档

完整文档请查看 `docs/` 目录：

- [**生成器系统** (NEW!)](./docs/GENERATOR_SYSTEM.md)：灵活生成后端、本地模型、API配置
- [**激励机制设计**](./STORYFI_INCENTIVE_MECHANISM_DESIGN.md)：详细的任务定义、评分系统、反作弊机制
- [**Protocol 使用示例**](./PROTOCOL_EXAMPLE.md)：如何使用通信协议
- [**项目计划**](./STORYFI_BITTENSOR_SUBNET_PROJECT_PLAN.md)：完整的开发路线图
- [**技术难度评估**](./TECHNICAL_DIFFICULTY_ASSESSMENT.md)：实现难度分析

## 🎁 奖励乘数系统

验证者会根据生成方式应用不同的奖励乘数：

| 生成方式 | 奖励乘数 | 推荐度 | 原因 |
|---------|---------|-------|------|
| 本地GPU模型 | **1.5x** | ⭐⭐⭐ | 促进去中心化 |
| API (OpenAI/Gemini) | **0.5x** | ⭐ | 降低中心化依赖 |
| 自定义实现 | **1.0x** | ⭐⭐ | 中性 |

**示例**：基础分数 80/100
- 本地模型：80 × 1.5 = 120（上限100）→ 满分！
- API：80 × 0.5 = 40 → 低奖励
- 自定义：80 × 1.0 = 80 → 标准奖励

## 🧪 测试

### 运行单元测试

```bash
pytest tests/ -v
```

### 测试 Protocol

```bash
python -m pytest tests/test_protocol.py -v
```

### 端到端测试

```bash
python -m pytest tests/test_e2e.py -v
```

## 🔧 配置参数

关键参数及其默认值：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QUERY_INTERVAL` | 12秒 | Validator 查询间隔 |
| `TIMEOUT` | 60秒 | Miner 响应超时 |
| `EMA_ALPHA` | 0.1 | EMA 平滑系数 |
| `TEMPERATURE` | 2.0 | Softmax 温度 |
| `PLAGIARISM_THRESHOLD` | 0.90 | 抄袭检测阈值 |

详细配置请查看 [INCENTIVE_MECHANISM.md](./STORYFI_INCENTIVE_MECHANISM_DESIGN.md#7-参数配置总结)

## 📊 监控

### 查看 Miner 状态

```bash
btcli subnet list --netuid 42
btcli wallet overview --wallet.name my_miner
```

### 查看权重

```bash
btcli weights --netuid 42
```

### 查看收入

```bash
btcli wallet balance --wallet.name my_miner
```

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建 feature 分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

### 代码规范

```bash
# 格式化代码
black .

# 类型检查
mypy neurons/ template/

# Lint
flake8 neurons/ template/
```

## 📄 许可证

MIT License - 详见 [LICENSE](./LICENSE) 文件

## 🔗 相关链接

- [Bittensor 官网](https://bittensor.com)
- [Bittensor 文档](https://docs.bittensor.com)
- [StoryFi 主项目](https://github.com/your-org/storyfi)

## 📞 联系方式

- Discord: [加入我们的 Discord](https://discord.gg/storyfi)
- Twitter: [@StoryFiAI](https://twitter.com/storyfi)
- Email: team@storyfi.ai

## 🎯 路线图

### Phase 1（当前）- MVP 开发
- [x] 激励机制设计
- [x] Protocol 实现
- [ ] Miner 实现
- [ ] Validator 实现
- [ ] 本地测试

### Phase 2 - 测试网部署
- [ ] 部署到 Bittensor 测试网
- [ ] 招募测试 Miners
- [ ] 优化评分系统
- [ ] 性能调优

### Phase 3 - 主网上线
- [ ] 审计安全性
- [ ] 部署到主网
- [ ] 启动 Miner 招募计划
- [ ] 监控和运维

### Phase 4 - 功能扩展
- [ ] 多语言支持
- [ ] 更多故事类型
- [ ] 机器学习优化评分
- [ ] 多 Validator 共识

---

**Built with ❤️ by StoryFi Team**
