"""
检查 StoryFi Subnet 部署准备情况
"""
import os
import sys
import subprocess

print("=" * 80)
print("StoryFi Subnet 部署准备检查")
print("=" * 80)
print()

# 1. 检查 Bittensor 版本
print("1. 检查 Bittensor 版本...")
try:
    import bittensor as bt
    print(f"   ✅ Bittensor v{bt.__version__} 已安装")
except ImportError:
    print(f"   ❌ Bittensor 未安装")
    print(f"   安装命令: pip3 install bittensor")
    sys.exit(1)

print()

# 2. 检查 Google Gemini API
print("2. 检查 Google Gemini API...")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"   ✅ GEMINI_API_KEY 已配置")
    print(f"   Key: {api_key[:20]}...")
else:
    print(f"   ❌ GEMINI_API_KEY 未配置")
    sys.exit(1)

print()

# 3. 检查钱包
print("3. 检查 Bittensor 钱包...")
wallet_name = os.getenv("WALLET_NAME", "test_miner")
print(f"   钱包名称: {wallet_name}")

wallet_path = os.path.expanduser(f"~/.bittensor/wallets/{wallet_name}")
if os.path.exists(wallet_path):
    print(f"   ✅ 钱包目录存在: {wallet_path}")

    # 检查 coldkey
    coldkey_path = os.path.join(wallet_path, "coldkey")
    hotkey_path = os.path.join(wallet_path, "hotkeys", "default")

    if os.path.exists(coldkey_path):
        print(f"   ✅ Coldkey 存在")
    else:
        print(f"   ⚠️ Coldkey 不存在")
        print(f"   创建命令: btcli wallet new_coldkey --wallet.name {wallet_name}")

    if os.path.exists(hotkey_path):
        print(f"   ✅ Hotkey (default) 存在")
    else:
        print(f"   ⚠️ Hotkey 不存在")
        print(f"   创建命令: btcli wallet new_hotkey --wallet.name {wallet_name} --wallet.hotkey default")
else:
    print(f"   ❌ 钱包不存在: {wallet_path}")
    print(f"   创建钱包命令:")
    print(f"   btcli wallet new_coldkey --wallet.name {wallet_name}")
    print(f"   btcli wallet new_hotkey --wallet.name {wallet_name} --wallet.hotkey default")

print()

# 4. 检查子网信息
print("4. 检查子网配置...")
netuid = os.getenv("NETUID", "108")
print(f"   子网 ID (netuid): {netuid}")
print(f"   目标网络: testnet (测试网)")
print()

# 5. 总结
print("=" * 80)
print("部署准备总结")
print("=" * 80)
print()
print("需要完成的步骤:")
print()
print("[ ] 1. 创建 Bittensor 钱包（如果还没有）")
print("       btcli wallet new_coldkey --wallet.name test_miner")
print("       btcli wallet new_hotkey --wallet.name test_miner --wallet.hotkey default")
print()
print("[ ] 2. 获取测试网 TAO（用于注册）")
print("       方法1: 访问 Discord #testnet-faucet 频道")
print("       方法2: 联系子网 owner")
print()
print("[ ] 3. 注册到测试网（需要约 0.1 testnet TAO）")
print(f"       btcli subnet register --netuid {netuid} --subtensor.network test --wallet.name test_miner --wallet.hotkey default")
print()
print("[ ] 4. 启动 Miner")
print(f"       python3 neurons/miner_gemini.py --netuid {netuid} --subtensor.network test --wallet.name test_miner --logging.info")
print()
print("[ ] 5. (可选) 启动 Validator")
print(f"       python3 neurons/validator.py --netuid {netuid} --subtensor.network test --wallet.name test_validator --logging.info")
print()
print("=" * 80)
print()
print("⚠️ 注意事项:")
print("- 测试网部署用于验证功能，不会获得真实 TAO")
print("- 确认一切正常后，再部署到主网")
print("- 主网注册需要真实的 TAO (约 0.1-1 TAO)")
print()
