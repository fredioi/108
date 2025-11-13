# StoryFi Subnet 108 部署指南

**版本**: 1.0.0
**更新时间**: 2025-11-05
**目标网络**: Bittensor Testnet
**Netuid**: 108

---

## 📋 目录

1. [系统概览](#系统概览)
2. [前置要求](#前置要求)
3. [环境准备](#环境准备)
4. [Wallet配置](#wallet配置)
5. [代码部署](#代码部署)
6. [配置文件设置](#配置文件设置)
7. [注册到Subnet](#注册到subnet)
8. [启动服务](#启动服务)
9. [监控和维护](#监控和维护)
10. [故障排查](#故障排查)
11. [生产环境建议](#生产环境建议)

---

## 系统概览

### StoryFi Subnet架构

```
┌─────────────────┐         ┌─────────────────┐
│   Validator     │────────▶│     Miner       │
│  (验证器)        │  Query   │   (矿工)         │
│                 │◀────────│                 │
│ - 发送任务       │ Response │ - 接收任务       │
│ - 评分         │          │ - 生成故事       │
│ - 设置权重      │          │ - 返回结果       │
└─────────────────┘         └─────────────────┘
         │                           │
         │                           │
         ▼                           ▼
┌────────────────────────────────────────────┐
│        Bittensor Testnet (Netuid 108)      │
│            wss://test.finney.opentensor.ai │
└────────────────────────────────────────────┘
```

### 故事生成流程

```
User Request → Validator
                  │
                  ├─▶ Task 1: blueprint (故事蓝图)
                  │      └─▶ Miner generates world/theme
                  │
                  ├─▶ Task 2: characters (角色生成)
                  │      └─▶ Miner generates 5 characters
                  │
                  ├─▶ Task 3: story_arc (故事结构)
                  │      └─▶ Miner generates 12-chapter outline
                  │
                  └─▶ Task 4: chapters (章节内容)
                         └─▶ Miner generates detailed chapters
```

---

## 前置要求

### 硬件要求

#### Miner (API模式 - 推荐新手)
- **CPU**: 2核心+
- **内存**: 4GB+
- **存储**: 10GB+
- **网络**: 稳定的互联网连接
- **带宽**: 10Mbps+

#### Miner (本地模型模式 - 高级用户)
- **CPU**: 8核心+
- **内存**: 32GB+
- **GPU**: NVIDIA RTX 3090 / A100 (16GB+ VRAM)
- **存储**: 50GB+
- **带宽**: 50Mbps+

#### Validator
- **CPU**: 4核心+
- **内存**: 8GB+
- **存储**: 20GB+
- **网络**: 稳定的互联网连接
- **带宽**: 20Mbps+

### 软件要求

- **操作系统**: Ubuntu 20.04+ / macOS 12+
- **Python**: 3.8+ (推荐3.10)
- **Git**: 2.0+
- **SSH**: 用于远程服务器管理

### 成本估算

#### Testnet (测试网)
- **Miner注册**: 约1 TAO (测试币，可从水龙头获取)
- **Validator注册**: 约1 TAO (测试币)
- **服务器**: $10-50/月 (阿里云/AWS)
- **API费用**: $0-50/月 (取决于使用量)

#### Mainnet (主网，未来)
- **Miner注册**: 需要实际TAO
- **Validator注册**: 需要更多TAO
- **服务器**: $50-500/月
- **API费用**: $50-500/月

---

## 环境准备

### 1. 创建服务器

#### 阿里云轻量应用服务器 (推荐中国用户)

```bash
# 配置建议
地域: 国内任意地域
实例规格: 2核4GB (最低) / 4核8GB (推荐)
系统镜像: Ubuntu 20.04 / 22.04
带宽: 5Mbps+
```

#### AWS EC2 (推荐国际用户)

```bash
# 配置建议
Region: 任意
Instance Type: t3.medium (最低) / t3.large (推荐)
AMI: Ubuntu 20.04 LTS
Security Group: 开放8091端口 (Miner) / 8092端口 (Validator)
```

### 2. SSH密钥配置

#### 生成SSH密钥 (本地Mac/Linux)

```bash
# 生成新的SSH密钥
ssh-keygen -t rsa -b 4096 -C "storyfi_deployment" -f ~/.ssh/storyfi_server_key

# 添加到SSH agent
ssh-add ~/.ssh/storyfi_server_key

# 上传公钥到服务器
ssh-copy-id -i ~/.ssh/storyfi_server_key.pub root@YOUR_SERVER_IP
```

#### 测试连接

```bash
ssh -i ~/.ssh/storyfi_server_key root@YOUR_SERVER_IP
```

### 3. 服务器初始化

```bash
# 更新系统
apt update && apt upgrade -y

# 安装必要工具
apt install -y python3 python3-pip git curl wget vim htop

# 安装Python 3.10 (如果系统版本较低)
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.10 python3.10-venv python3.10-dev

# 设置Python 3.10为默认
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
```

---

## Wallet配置

### 1. 安装Bittensor CLI

```bash
# 安装bittensor
pip3 install bittensor

# 验证安装
python3 -m bittensor --version
```

### 2. 创建Miner Wallet

```bash
# 创建coldkey (冷钱包，存储TAO)
btcli wallet new_coldkey --wallet.name storyfi_miner

# 创建hotkey (热钱包，用于注册)
btcli wallet new_hotkey --wallet.name storyfi_miner --wallet.hotkey default

# 查看钱包地址
btcli wallet overview --wallet.name storyfi_miner
```

**重要**: 保存好助记词！丢失后无法恢复！

### 3. 创建Validator Wallet (可选)

```bash
# 创建coldkey
btcli wallet new_coldkey --wallet.name storyfi_validator

# 创建hotkey
btcli wallet new_hotkey --wallet.name storyfi_validator --wallet.hotkey default
```

### 4. 获取测试TAO

#### 方法1: Bittensor Discord水龙头

1. 加入Bittensor Discord: https://discord.gg/bittensor
2. 前往 #testnet-faucet 频道
3. 发送: `/faucet YOUR_COLDKEY_ADDRESS`
4. 等待确认

#### 方法2: 社区水龙头

```bash
# 访问社区水龙头网站
# https://faucet.bittensor.com/
```

### 5. 验证余额

```bash
# 查看余额
btcli wallet balance --wallet.name storyfi_miner --network test

# 输出示例:
# Wallet: storyfi_miner
# Coldkey: 5F9gsRBgHrQdkG2f3fWP6NRkQREfwQdk3hGdsif2tdvKczTH
# Balance: 10.000000000 τ (TAO)
```

---

## 代码部署

### 1. 克隆代码仓库

```bash
# SSH到服务器
ssh -i ~/.ssh/storyfi_server_key root@YOUR_SERVER_IP

# 克隆仓库
cd /root
git clone https://github.com/your-org/storyfi-subnet.git
cd storyfi-subnet

# 查看当前分支
git branch
```

### 2. 安装Python依赖

```bash
# 创建虚拟环境 (可选，推荐)
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip3 install -r requirements.txt

# 验证安装
pip3 list | grep bittensor
```

### 3. 项目结构说明

```
storyfi-subnet/
├── neurons/
│   ├── miner.py          # Miner主程序
│   └── validator.py      # Validator主程序
├── template/
│   └── protocol.py       # 通信协议定义
├── generators/
│   ├── api_generator.py  # API模式生成器
│   ├── local_generator.py # 本地模型生成器
│   └── base.py           # 生成器基类
├── config/
│   ├── generator_config.yaml  # 生成器配置
│   └── prompts/          # 提示词模板
├── .env                  # 环境变量 (需创建)
├── requirements.txt      # Python依赖
└── README.md             # 项目文档
```

---

## 配置文件设置

### 1. 创建.env文件

#### 方法1: 使用智谱AI (推荐中国用户)

```bash
# 创建.env文件
cat > .env << 'EOF'
# 智谱AI API密钥
ZHIPU_API_KEY=your_zhipu_api_key_here

# (可选) Google Gemini API密钥
GEMINI_API_KEY=your_gemini_api_key_here
EOF
```

#### 获取智谱AI API密钥

1. 访问: https://open.bigmodel.cn/
2. 注册/登录账号
3. 前往"个人中心" → "API Keys"
4. 创建新的API Key
5. 复制密钥到`.env`文件

#### 方法2: 使用Google Gemini (推荐国际用户)

```bash
# 创建.env文件
cat > .env << 'EOF'
# Google Gemini API密钥
GEMINI_API_KEY=your_gemini_api_key_here
EOF
```

#### 获取Gemini API密钥

1. 访问: https://makersuite.google.com/app/apikey
2. 登录Google账号
3. 点击"Create API Key"
4. 复制密钥到`.env`文件

### 2. 配置generator_config.yaml

#### 配置示例 (智谱AI)

```yaml
generator:
  mode: "api"  # 模式: api / local / vllm

  api:
    provider: "openai"  # OpenAI-compatible API
    api_key_env: "ZHIPU_API_KEY"  # 环境变量名
    model: "glm-4-flash"  # 模型名称
    endpoint: "https://open.bigmodel.cn/api/paas/v4"  # API端点
```

#### 配置示例 (Google Gemini)

```yaml
generator:
  mode: "api"

  api:
    provider: "gemini"
    api_key_env: "GEMINI_API_KEY"
    model: "gemini-2.0-flash-exp"
    endpoint: null  # Gemini不需要自定义endpoint
```

#### 配置示例 (本地模型 - 高级用户)

```yaml
generator:
  mode: "local"

  local:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    device: "cuda"  # 使用GPU
    quantization: "4bit"  # 4bit量化，节省VRAM
    max_memory: "16GB"
    use_flash_attention: true
```

### 3. 验证配置

```bash
# 测试配置文件
python3 -c "
import yaml
with open('config/generator_config.yaml') as f:
    config = yaml.safe_load(f)
    print('✅ Config loaded:', config['generator']['mode'])
"

# 测试环境变量
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('✅ API Key loaded:', os.getenv('ZHIPU_API_KEY')[:10] + '...')
"
```

---

## 注册到Subnet

### 1. 注册Miner

```bash
# 注册到testnet subnet 108
btcli subnet register \
  --netuid 108 \
  --subtensor.network test \
  --wallet.name storyfi_miner \
  --wallet.hotkey default

# 预期输出:
# ✅ Registered UID 6 to subnet 108
# Transaction: 0x1234...
```

### 2. 注册Validator (可选)

```bash
# 注册Validator
btcli subnet register \
  --netuid 108 \
  --subtensor.network test \
  --wallet.name storyfi_validator \
  --wallet.hotkey default
```

### 3. 验证注册状态

```bash
# 查看metagraph
btcli subnet metagraph --netuid 108 --network test

# 预期输出:
# UID | Stake | Trust | Rank | Incentive | Dividends | Emission | ...
#  0  | 0.0   | 0.0   | 0.0  | 0.0       | 0.0       | 0.0      | ...
#  6  | 0.0   | 0.0   | 0.0  | 0.0       | 0.0       | 0.0      | ... (你的节点)
```

### 4. 故障排查

#### 问题: "Insufficient balance"

```bash
# 解决方案: 获取更多测试TAO
btcli wallet balance --wallet.name storyfi_miner --network test

# 如果余额不足，返回"获取测试TAO"步骤
```

#### 问题: "UID already registered"

```bash
# 解决方案: 检查是否已注册
btcli subnet metagraph --netuid 108 --network test | grep YOUR_HOTKEY

# 如果已注册，跳过此步骤
```

---

## 启动服务

### 1. 启动Miner

#### 方法1: 前台运行 (测试用)

```bash
cd /root/storyfi-subnet

python3 neurons/miner.py \
  --netuid 108 \
  --subtensor.network test \
  --wallet.name storyfi_miner \
  --wallet.hotkey default \
  --axon.port 8091 \
  --logging.debug
```

#### 方法2: 后台运行 (生产用)

```bash
cd /root/storyfi-subnet

nohup python3 neurons/miner.py \
  --netuid 108 \
  --subtensor.network test \
  --wallet.name storyfi_miner \
  --wallet.hotkey default \
  --axon.port 8091 \
  --logging.debug \
  > miner.log 2>&1 &

# 保存进程ID
echo $! > miner.pid
```

#### 方法3: 使用systemd (推荐)

```bash
# 创建systemd服务文件
cat > /etc/systemd/system/storyfi-miner.service << 'EOF'
[Unit]
Description=StoryFi Bittensor Miner
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/storyfi-subnet
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 neurons/miner.py --netuid 108 --subtensor.network test --wallet.name storyfi_miner --wallet.hotkey default --axon.port 8091 --logging.debug
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
systemctl daemon-reload
systemctl enable storyfi-miner
systemctl start storyfi-miner

# 查看状态
systemctl status storyfi-miner
```

### 2. 启动Validator (可选)

#### 后台运行

```bash
cd /root/storyfi-subnet

nohup python3 neurons/validator.py \
  --netuid 108 \
  --subtensor.network test \
  --wallet.name storyfi_validator \
  --wallet.hotkey default \
  --logging.debug \
  > validator.log 2>&1 &

echo $! > validator.pid
```

#### 使用systemd

```bash
cat > /etc/systemd/system/storyfi-validator.service << 'EOF'
[Unit]
Description=StoryFi Bittensor Validator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/storyfi-subnet
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 neurons/validator.py --netuid 108 --subtensor.network test --wallet.name storyfi_validator --wallet.hotkey default --logging.debug
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable storyfi-validator
systemctl start storyfi-validator
systemctl status storyfi-validator
```

### 3. 验证服务运行

```bash
# 检查进程
ps aux | grep miner.py
ps aux | grep validator.py

# 检查日志
tail -f miner.log
tail -f validator.log

# 检查端口
netstat -tulpn | grep 8091  # Miner
netstat -tulpn | grep 8092  # Validator (如果运行)
```

### 4. 查看实时日志

```bash
# Miner日志
tail -f miner.log | grep -E '(INFO|SUCCESS|ERROR)'

# 预期输出:
# 📨 Received blueprint request
# ✅ Generated blueprint in 31.79s (output: 4712 chars)
# 📊 Stats: Requests=20, AvgTime=25.90s, Errors=0

# Validator日志
tail -f validator.log | grep -E '(INFO|SUCCESS|ERROR)'

# 预期输出:
# 🎯 Task type: blueprint
# 📡 Querying 9 miners: [0, 1, 2, 3, 4, 5, 7, 8, 6]
# ✅ Miner 6 score: 0.95
```

---

## 监控和维护

### 1. 性能指标监控

#### 创建监控脚本

```bash
cat > /root/monitor_subnet.sh << 'EOF'
#!/bin/bash

echo "================================================"
echo "StoryFi Subnet 108 监控面板"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

echo ""
echo "【进程状态】"
ps aux | grep -E '(miner|validator)' | grep -v grep | awk '{printf "%-20s PID: %-8s CPU: %-6s MEM: %-6s\n", $11, $2, $3"%", $4"%"}'

echo ""
echo "【Miner统计】"
tail -20 miner.log | grep "📊 Stats" | tail -1

echo ""
echo "【最近请求】"
tail -30 miner.log | grep -E '(Received|Generated)' | tail -5

echo ""
echo "【错误检查】"
tail -50 miner.log | grep ERROR | tail -3 || echo "✅ 无错误"

echo ""
echo "【网络连接】"
netstat -an | grep 8091 | wc -l | xargs echo "活跃连接数:"

echo ""
echo "【系统资源】"
free -h | grep Mem | awk '{printf "内存使用: %s / %s (%.1f%%)\n", $3, $2, ($3/$2)*100}'
df -h / | tail -1 | awk '{printf "磁盘使用: %s / %s (%s)\n", $3, $2, $5}'

echo "================================================"
EOF

chmod +x /root/monitor_subnet.sh

# 运行监控
/root/monitor_subnet.sh
```

#### 设置定时监控

```bash
# 添加cron任务，每5分钟记录一次
crontab -e

# 添加以下行:
*/5 * * * * /root/monitor_subnet.sh >> /var/log/subnet_monitor.log 2>&1

# 查看监控日志
tail -f /var/log/subnet_monitor.log
```

### 2. 自动重启脚本

```bash
cat > /root/check_and_restart.sh << 'EOF'
#!/bin/bash

# 检查Miner进程
if ! pgrep -f "neurons/miner.py" > /dev/null; then
    echo "[$(date)] ⚠️ Miner进程已停止，正在重启..." >> /var/log/auto_restart.log
    cd /root/storyfi-subnet
    nohup python3 neurons/miner.py \
      --netuid 108 \
      --subtensor.network test \
      --wallet.name storyfi_miner \
      --wallet.hotkey default \
      --axon.port 8091 \
      --logging.debug \
      > miner.log 2>&1 &
    echo "[$(date)] ✅ Miner已重启" >> /var/log/auto_restart.log
fi

# 检查Validator进程 (如果运行)
if pgrep -f "neurons/validator.py" > /dev/null; then
    echo "[$(date)] ✅ Validator运行正常" >> /var/log/auto_restart.log
elif [ -f /root/storyfi-subnet/validator.pid ]; then
    echo "[$(date)] ⚠️ Validator进程已停止，正在重启..." >> /var/log/auto_restart.log
    cd /root/storyfi-subnet
    nohup python3 neurons/validator.py \
      --netuid 108 \
      --subtensor.network test \
      --wallet.name storyfi_validator \
      --wallet.hotkey default \
      --logging.debug \
      > validator.log 2>&1 &
    echo "[$(date)] ✅ Validator已重启" >> /var/log/auto_restart.log
fi
EOF

chmod +x /root/check_and_restart.sh

# 添加到cron，每分钟检查一次
crontab -e
# 添加:
* * * * * /root/check_and_restart.sh
```

### 3. 日志轮转

```bash
# 创建logrotate配置
cat > /etc/logrotate.d/storyfi-subnet << 'EOF'
/root/storyfi-subnet/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0644 root root
}
EOF

# 测试配置
logrotate -d /etc/logrotate.d/storyfi-subnet
```

### 4. 备份钱包

```bash
# 创建备份脚本
cat > /root/backup_wallet.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/root/wallet_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# 备份钱包
cp -r ~/.bittensor/wallets $BACKUP_DIR/wallets_$TIMESTAMP

# 压缩
tar -czf $BACKUP_DIR/wallets_$TIMESTAMP.tar.gz $BACKUP_DIR/wallets_$TIMESTAMP
rm -rf $BACKUP_DIR/wallets_$TIMESTAMP

# 保留最近7天的备份
find $BACKUP_DIR -name "wallets_*.tar.gz" -mtime +7 -delete

echo "✅ 钱包已备份: $BACKUP_DIR/wallets_$TIMESTAMP.tar.gz"
EOF

chmod +x /root/backup_wallet.sh

# 手动备份
/root/backup_wallet.sh

# 自动每日备份
crontab -e
# 添加:
0 2 * * * /root/backup_wallet.sh >> /var/log/wallet_backup.log 2>&1
```

---

## 故障排查

### 常见问题1: "Requests=0" (没有收到请求)

#### 诊断步骤

```bash
# 1. 检查注册状态
btcli subnet metagraph --netuid 108 --network test | grep YOUR_UID

# 2. 检查端口是否开放
netstat -tulpn | grep 8091

# 3. 检查防火墙
ufw status

# 4. 检查axon是否已serve
grep "Axon started" miner.log

# 5. 检查网络连接
curl -I http://YOUR_SERVER_IP:8091
```

#### 解决方案

```bash
# 方案1: 重启Miner
pkill -f miner.py
sleep 2
# 重新启动 (见"启动服务"章节)

# 方案2: 检查配置
grep "axon.port" miner.log

# 方案3: 开放端口
ufw allow 8091/tcp
ufw reload
```

### 常见问题2: API错误 "Timeout" / "Connection refused"

#### 诊断步骤

```bash
# 1. 检查API密钥
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('ZHIPU_API_KEY'))"

# 2. 测试API连接
curl -X POST https://open.bigmodel.cn/api/paas/v4/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-4-flash","messages":[{"role":"user","content":"test"}]}'

# 3. 检查网络访问
ping -c 3 open.bigmodel.cn
```

#### 解决方案

```bash
# 方案1: 更换API提供商
# 如果在中国，使用智谱AI
# 如果在国外，使用Gemini

# 方案2: 检查API额度
# 登录API控制台查看使用量和余额

# 方案3: 增加超时时间
# 修改 generators/api_generator.py 中的timeout参数
```

### 常见问题3: "SynapseParsingError"

#### 诊断步骤

```bash
# 检查协议版本
grep "protocol_version" template/protocol.py

# 检查日志
grep "SynapseParsingError" miner.log
```

#### 解决方案

```bash
# 更新到最新版本
cd /root/storyfi-subnet
git pull origin main

# 重启服务
systemctl restart storyfi-miner
```

### 常见问题4: 内存不足

#### 诊断步骤

```bash
# 检查内存使用
free -h
top -o %MEM | head -20
```

#### 解决方案

```bash
# 方案1: 升级服务器配置

# 方案2: 使用API模式代替本地模型
# 修改 config/generator_config.yaml:
#   mode: "api"

# 方案3: 添加swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 常见问题5: 生成超时

#### 诊断步骤

```bash
# 检查生成时间
grep "generation_time" miner.log | tail -20
```

#### 解决方案

```bash
# 方案1: 使用更快的模型
# 智谱AI: glm-4-flash (推荐)
# Gemini: gemini-2.0-flash-exp

# 方案2: 调整timeout
# 修改 neurons/miner.py 中的timeout参数

# 方案3: 优化提示词
# 编辑 config/prompts/ 中的模板文件
```

---

## 生产环境建议

### 1. 安全加固

```bash
# 禁用root SSH登录
sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
systemctl restart sshd

# 配置防火墙
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 8091/tcp  # Miner
ufw enable

# 设置fail2ban
apt install -y fail2ban
systemctl enable fail2ban
systemctl start fail2ban
```

### 2. 监控告警

```bash
# 安装监控工具
apt install -y prometheus-node-exporter

# 配置Telegram Bot告警 (可选)
cat > /root/alert_telegram.sh << 'EOF'
#!/bin/bash

BOT_TOKEN="YOUR_BOT_TOKEN"
CHAT_ID="YOUR_CHAT_ID"
MESSAGE="$1"

curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
  -d chat_id="$CHAT_ID" \
  -d text="$MESSAGE"
EOF

chmod +x /root/alert_telegram.sh

# 测试
/root/alert_telegram.sh "✅ StoryFi Subnet 108 已启动"
```

### 3. 性能优化

```bash
# 调整系统参数
cat >> /etc/sysctl.conf << 'EOF'
# 增加文件描述符限制
fs.file-max = 65536

# 优化网络
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864
EOF

sysctl -p

# 调整ulimit
cat >> /etc/security/limits.conf << 'EOF'
* soft nofile 65536
* hard nofile 65536
EOF
```

### 4. 高可用部署 (可选)

```bash
# 配置多台Miner服务器
# Server 1: Miner A (主)
# Server 2: Miner B (备)

# 使用keepalived实现故障转移
apt install -y keepalived

# 配置keepalived.conf
# (具体配置根据实际情况调整)
```

### 5. 成本优化

```bash
# 1. 使用预留实例 (Reserved Instances)
#    - 阿里云: 节省30-60%
#    - AWS: 节省30-75%

# 2. 使用竞价实例 (Spot Instances) - 非生产环境
#    - AWS: 节省50-90%

# 3. 优化API使用
#    - 使用缓存减少重复请求
#    - 选择性价比高的模型 (glm-4-flash)

# 4. 定期清理日志
find /root/storyfi-subnet -name "*.log" -mtime +30 -delete
```

---

## 附录A: 快速命令参考

### Miner管理

```bash
# 启动
systemctl start storyfi-miner

# 停止
systemctl stop storyfi-miner

# 重启
systemctl restart storyfi-miner

# 查看状态
systemctl status storyfi-miner

# 查看日志
journalctl -u storyfi-miner -f

# 手动启动 (调试)
cd /root/storyfi-subnet && python3 neurons/miner.py --netuid 108 --subtensor.network test --wallet.name storyfi_miner --wallet.hotkey default --logging.debug
```

### Validator管理

```bash
# 启动
systemctl start storyfi-validator

# 停止
systemctl stop storyfi-validator

# 重启
systemctl restart storyfi-validator

# 查看状态
systemctl status storyfi-validator

# 查看日志
journalctl -u storyfi-validator -f
```

### Bittensor命令

```bash
# 查看余额
btcli wallet balance --wallet.name storyfi_miner --network test

# 查看metagraph
btcli subnet metagraph --netuid 108 --network test

# 查看钱包信息
btcli wallet overview --wallet.name storyfi_miner

# 转账 (testnet)
btcli wallet transfer --wallet.name storyfi_miner --dest COLDKEY_ADDRESS --amount 1.0 --network test
```

### 监控命令

```bash
# 实时监控
watch -n 5 '/root/monitor_subnet.sh'

# 查看最近统计
tail -100 miner.log | grep "📊 Stats"

# 查看错误
tail -200 miner.log | grep ERROR

# 查看网络连接
netstat -an | grep 8091

# 查看进程
ps aux | grep -E '(miner|validator)'
```

---

## 附录B: 配置文件模板

### .env模板

```bash
# StoryFi Subnet环境变量

# 智谱AI (推荐中国用户)
ZHIPU_API_KEY=1cb142e535834ee1adea6cd7ea099512.sfS5SvWMyScddCQ6

# Google Gemini (推荐国际用户)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI (可选)
OPENAI_API_KEY=your_openai_api_key_here

# 自定义配置
MAX_RETRIES=3
TIMEOUT_SECONDS=60
```

### generator_config.yaml模板 (智谱AI)

```yaml
generator:
  mode: "api"
  use_templates: true
  template_dir: "./config/prompts"

  api:
    provider: "openai"
    api_key_env: "ZHIPU_API_KEY"
    model: "glm-4-flash"
    endpoint: "https://open.bigmodel.cn/api/paas/v4"
```

### generator_config.yaml模板 (Gemini)

```yaml
generator:
  mode: "api"
  use_templates: true
  template_dir: "./config/prompts"

  api:
    provider: "gemini"
    api_key_env: "GEMINI_API_KEY"
    model: "gemini-2.0-flash-exp"
    endpoint: null
```

---

## 附录C: 问题反馈和支持

### 提交Issue

如果遇到问题，请按以下格式提交Issue:

```markdown
**问题描述**
简要描述遇到的问题

**环境信息**
- 操作系统: Ubuntu 20.04
- Python版本: 3.10
- Bittensor版本: 9.12.2
- Subnet: 108
- 网络: testnet

**复现步骤**
1. 执行命令X
2. 观察到现象Y
3. 期望结果Z

**日志信息**
```
粘贴相关日志
```

**已尝试的解决方案**
列出已经尝试过的方法
```

### 获取帮助

- **Discord**: https://discord.gg/bittensor (官方)
- **GitHub Issues**: https://github.com/your-org/storyfi-subnet/issues
- **Telegram**: @storyfi_support (如有)
- **Email**: support@storyfi.ai (如有)

---

## 更新日志

### v1.0.0 (2025-11-05)
- ✅ 首次发布完整部署指南
- ✅ 支持API模式 (智谱AI / Gemini)
- ✅ 支持本地模型模式
- ✅ 完整的监控和维护方案
- ✅ 详细的故障排查指南

---

**文档维护**: StoryFi Team
**最后更新**: 2025-11-05
**联系方式**: GitHub Issues
