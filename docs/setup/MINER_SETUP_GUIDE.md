# StoryFi Miner 设置指南

## 前置要求

- Python 3.9+
- OpenAI API Key
- Bittensor wallet（已创建并注册）
- 至少 1 TAO（用于注册到子网）

## 快速开始（5分钟）

### 1. 安装依赖

```bash
cd storyfi-subnet
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置
nano .env
```

**必需配置**：
```bash
OPENAI_API_KEY=sk-your-key-here  # ← 你的 OpenAI API Key
NETUID=42                         # ← 你朋友的子网 UID
WALLET_NAME=my_miner              # ← 你的钱包名称
WALLET_HOTKEY=default             # ← 你的 hotkey
```

### 3. 创建 Bittensor 钱包（如果还没有）

```bash
# 创建新钱包
btcli wallet create --wallet.name my_miner

# 创建 hotkey
btcli wallet new_hotkey --wallet.name my_miner --wallet.hotkey default

# 查看地址
btcli wallet overview --wallet.name my_miner
```

### 4. 获取测试 TAO（测试网）

```bash
# 从测试网 faucet 获取
btcli wallet faucet --wallet.name my_miner --subtensor.network test
```

### 5. 注册到子网

```bash
# 注册到子网（需要 1 TAO）
btcli subnet register \
    --netuid 42 \
    --wallet.name my_miner \
    --wallet.hotkey default
```

### 6. 启动 Miner

```bash
python neurons/miner.py \
    --netuid 42 \
    --wallet.name my_miner \
    --wallet.hotkey default \
    --logging.info
```

**成功输出**：
```
2025-10-16 14:30:45 - INFO - Initializing StoryFi Miner...
2025-10-16 14:30:46 - INFO - ✅ Wallet: 5FxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxYz
2025-10-16 14:30:46 - INFO - ✅ Model: gpt-4-turbo-preview
2025-10-16 14:30:46 - INFO - ✅ Netuid: 42
2025-10-16 14:30:47 - INFO - Setting up axon...
2025-10-16 14:30:48 - INFO - ✅ Axon started on port 8091
2025-10-16 14:30:49 - INFO - ✅ Registered to subnet 42
2025-10-16 14:30:49 - INFO - 🚀 Starting miner...
```

## 详细配置

### OpenAI 模型选择

| 模型 | 成本 | 速度 | 质量 | 推荐场景 |
|------|------|------|------|----------|
| gpt-4-turbo-preview | $$ | 快 | 高 | **生产环境（推荐）** |
| gpt-4 | $$$ | 慢 | 最高 | 追求极致质量 |
| gpt-3.5-turbo | $ | 最快 | 中 | 测试和开发 |

**配置方法**：
```bash
# .env
OPENAI_MODEL=gpt-4-turbo-preview
MAX_TOKENS=3000
TEMPERATURE=0.7
```

### 端口配置

如果 8091 端口被占用：
```bash
python neurons/miner.py \
    --netuid 42 \
    --wallet.name my_miner \
    --wallet.hotkey default \
    --axon.port 8092 \
    --logging.info
```

### 日志级别

```bash
# Info 日志（推荐）
python neurons/miner.py --logging.info

# Debug 日志（详细调试）
python neurons/miner.py --logging.debug

# 同时启用
python neurons/miner.py --logging.info --logging.debug
```

## 测试 Miner

### 本地测试（无需连接子网）

创建 `test_miner_local.py`：
```python
import asyncio
import json
from template.protocol import create_blueprint_synapse

# 模拟测试
async def test_local():
    # 假设 miner 正在运行
    synapse = create_blueprint_synapse("一个关于太空探险的故事")
    print(f"Task: {synapse.task_type}")
    print(f"Input: {synapse.input_data}")

asyncio.run(test_local())
```

### 使用 Validator 测试

```bash
# 等待 Validator 实现后测试
python neurons/validator.py --test_mode
```

## 监控和维护

### 查看 Miner 状态

```bash
# 查看子网所有 Miners
btcli subnet list --netuid 42

# 查看你的 Miner 信息
btcli wallet overview --wallet.name my_miner

# 查看权重分配
btcli weights --netuid 42
```

### 查看实时日志

Miner 每 60 秒打印一次统计信息：
```
📊 Stats: Requests=150, AvgTime=2.34s, Errors=2
```

**指标说明**：
- `Requests`: 处理的总请求数
- `AvgTime`: 平均生成时间（秒）
- `Errors`: 错误数量

### 重启 Miner

```bash
# 找到进程
ps aux | grep miner.py

# 杀死进程
kill <PID>

# 重新启动
python neurons/miner.py --netuid 42 --wallet.name my_miner --logging.info
```

### 使用 PM2 管理（推荐生产环境）

```bash
# 安装 PM2
npm install -g pm2

# 启动 Miner
pm2 start neurons/miner.py --name storyfi-miner --interpreter python3 -- \
    --netuid 42 \
    --wallet.name my_miner \
    --wallet.hotkey default \
    --logging.info

# 查看日志
pm2 logs storyfi-miner

# 查看状态
pm2 status

# 重启
pm2 restart storyfi-miner

# 开机自启
pm2 startup
pm2 save
```

## 常见问题

### 1. `OPENAI_API_KEY not found`

**错误**：
```
ValueError: OPENAI_API_KEY not found in environment variables
```

**解决**：
```bash
# 检查 .env 文件是否存在
ls -la .env

# 检查内容
cat .env | grep OPENAI_API_KEY

# 确保格式正确（没有空格）
OPENAI_API_KEY=sk-your-key-here
```

### 2. 端口被占用

**错误**：
```
OSError: [Errno 48] Address already in use
```

**解决**：
```bash
# 查看占用端口的进程
lsof -i :8091

# 杀死进程或使用其他端口
python neurons/miner.py --axon.port 8092 ...
```

### 3. 注册失败

**错误**：
```
Failed to register to subnet
```

**检查**：
```bash
# 1. 检查钱包余额（需要 ≥1 TAO）
btcli wallet balance --wallet.name my_miner

# 2. 检查子网是否存在
btcli subnet list

# 3. 检查是否已经注册
btcli subnet list --netuid 42
```

### 4. OpenAI API 超时

**错误**：
```
openai.error.Timeout: Request timed out
```

**解决**：
```bash
# 增加超时时间
# 在 miner.py 中修改:
self.openai_client = AsyncOpenAI(
    api_key=api_key,
    timeout=120.0  # 增加到 120 秒
)
```

### 5. JSON 解析失败

**错误**：
```
Failed to parse JSON response
```

**原因**：OpenAI 返回了包含 Markdown 的内容

**解决**：
代码已包含自动处理逻辑，但可以优化 prompt：
```python
prompt = f"""你必须只返回纯JSON，不要有任何Markdown格式！

不要返回：
```json
{{...}}
```

只返回：
{{...}}

{原始prompt}
"""
```

## 收益计算

### 估算日收益

假设：
- 子网每日排放：1000 TAO
- Miners 分配：410 TAO（41%）
- 你的权重：2%
- TAO 价格：$500

```
日收益 = 410 TAO * 2% * $500 = $4,100
月收益 = $4,100 * 30 = $123,000
```

**实际收益取决于**：
1. 你的评分（质量）
2. 其他 Miners 的质量
3. Validator 的评分算法
4. 网络总排放量

### 提高收益策略

1. **提升质量**：使用 GPT-4 而不是 GPT-3.5
2. **优化响应时间**：减少 API 调用延迟
3. **避免作弊**：不要复制模板或其他 Miner
4. **保持在线**：24/7 运行，不错过请求
5. **多个 Miners**：运行多个 hotkeys（需要多个 TAO 注册费）

## 下一步

1. ✅ Miner 成功运行
2. ⏳ 等待 Validator 部署并开始评分
3. ⏳ 监控你的权重和收益
4. ⏳ 根据评分反馈优化生成质量

## 支持

遇到问题？

- 查看 [完整文档](./README.md)
- 查看 [激励机制设计](./STORYFI_INCENTIVE_MECHANISM_DESIGN.md)
- 加入 [Discord](https://discord.gg/storyfi)
- 发送邮件：support@storyfi.ai

---

**祝你挖矿顺利！💰**
