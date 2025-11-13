# StoryFi子网文档索引

**最后更新**: 2025-10-28
**状态**: 测试网已部署 (Subnet 108)

---

## 📂 文档结构

### 根目录
- `README.md` - 项目主要说明文档
- `DOCS_INDEX.md` - 本文档（文档导航）

### docs/setup/ - 设置指南
1. **DEPLOYMENT_GUIDE.md** - 部署指南（主网/测试网）
2. **MINER_SETUP_GUIDE.md** - 矿工设置指南
3. **LOCAL_TEST_GUIDE.md** - 本地测试指南
4. **GEMINI_SETUP_GUIDE.md** - Gemini API设置
5. **AI_MODEL_OPTIONS.md** - AI模型选择建议

### docs/reports/ - 进度报告
1. **FIXES_SUMMARY.md** - Bug修复总结（2025-10-28）
2. **IMPLEMENTATION_COMPLETE.md** - 实现完成报告
3. **PHASE1_FINAL_REPORT.md** - Phase 1最终报告
4. **PHASE1_PROGRESS_REPORT.md** - Phase 1进度报告

### docs/analysis/ - 技术分析
1. **MINER_ARCHITECTURE_REDESIGN.md** - 矿工架构重设计（⚠️ 设计完成，待实现）
2. **PROTOCOL_V3_ANALYSIS.md** - Protocol v3.1.0分析
3. **BITTENSOR_RESEARCH_REPORT.md** - Bittensor研究报告
4. **ALGORITHM_REFERENCE.md** - 算法参考
5. **CRITICAL_ANALYSIS.md** - 关键问题分析
6. **SCHEMA_MISMATCH_FINDINGS.md** - Schema不匹配发现

### docs/archive/ - 过时文档
- COMPREHENSIVE_IMPROVEMENT_PLAN.md
- PROMPT_IMPROVEMENTS_SUMMARY.md
- DEPLOYMENT_CHECKLIST.md
- QUESTIONS_FOR_SUBNET_OWNER.md
- PROTOCOL_EXAMPLE.md
- BLOCKCHAIN_EXPLORER_LINKS.md

### scripts/ - 运行脚本
- `check_miner_status.sh` - 检查矿工状态
- （其他启动脚本在项目根目录）

### tools/ - 工具脚本
- `create_wallet.py` - 创建钱包
- `import_wallet.py` - 导入钱包
- `check_deployment_readiness.py` - 检查部署准备情况
- `fix_ssl.py` - 修复SSL问题
- `list_gemini_models.py` - 列出Gemini模型

---

## 🚨 当前关键问题

### ⚠️ 架构问题（未解决）

**问题**: 矿工绑定特定API（OpenAI/Gemini），违反Bittensor去中心化哲学

**状态**:
- ✅ 架构设计完成（见 `docs/analysis/MINER_ARCHITECTURE_REDESIGN.md`）
- ❌ 实现未完成
- ❌ 当前代码仍使用硬编码API

**影响**:
- 明天（2025-10-29）上线时将使用不符合理念的架构
- 需要决定：延迟上线 vs 上线后修复

**正确架构**:
```
矿工应该自由选择任何实现方式:
- 本地GPU模型（Llama, Mixtral等）
- 任何API（OpenAI, Gemini, Claude等）
- 人工写作
- 其他任何方法

验证者只评判输出质量，不限制生成方法
```

---

## 📋 快速开始

### 测试网部署
```bash
# 1. 创建钱包
python tools/create_wallet.py

# 2. 获取测试TAO
# 访问 https://discord.gg/bittensor 获取测试币

# 3. 注册到子网108
btcli subnet register --netuid 108 --wallet.name your_wallet --subtensor.network test

# 4. 启动矿工
./start_testnet_miner.sh

# 5. 启动验证者
./start_testnet_validator.sh

# 6. 检查状态
./scripts/check_miner_status.sh
```

### 查看文档
```bash
# 部署指南
cat docs/setup/DEPLOYMENT_GUIDE.md

# Bug修复总结
cat docs/reports/FIXES_SUMMARY.md

# 架构问题分析
cat docs/analysis/MINER_ARCHITECTURE_REDESIGN.md
```

---

## 🔧 技术栈

- **Python 3.8+**
- **Bittensor 7.3+**
- **OpenAI API / Gemini API** （⚠️ 当前强制要求，需修复）
- **Protocol**: v3.1.0

---

## 📞 相关资源

### Bittensor
- 主网: https://taostats.io
- 测试网: https://test.taostats.io
- 文档: https://docs.bittensor.com

### StoryFi子网
- 子网ID: 108 (测试网)
- 状态: 运行中

---

## ⚠️ 已知限制

1. **矿工架构** - 当前强制使用特定API（违反设计理念）
2. **未实现功能** - Handshake机制、Redis持久化、Prometheus监控

---

## 📝 贡献

查看 `docs/analysis/` 了解技术细节
查看 `docs/reports/` 了解最新进展

---

**文档维护**: Claude Code
**项目状态**: 测试网部署，等待架构修复
