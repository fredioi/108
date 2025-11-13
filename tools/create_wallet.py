"""
使用 Python API 创建 Bittensor 钱包
"""
import bittensor as bt

print("=" * 80)
print("创建 Bittensor 测试网钱包")
print("=" * 80)
print()

# 配置
wallet_name = "storyfi_miner"
hotkey_name = "default"

print(f"钱包名称: {wallet_name}")
print(f"Hotkey: {hotkey_name}")
print()

# 创建钱包
print("步骤 1/3: 创建 Wallet 对象...")
wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
print("✅ Wallet 对象创建成功")
print()

# 创建新的 coldkey 和 hotkey
print("步骤 2/3: 生成密钥...")
try:
    # 创建新的 coldkey（如果不存在）
    if not wallet.coldkey_file.exists_on_device():
        wallet.create_new_coldkey(use_password=False, overwrite=False)
        print("✅ Coldkey 创建成功")
    else:
        print("⚠️ Coldkey 已存在，跳过创建")

    # 创建新的 hotkey（如果不存在）
    if not wallet.hotkey_file.exists_on_device():
        wallet.create_new_hotkey(use_password=False, overwrite=False)
        print("✅ Hotkey 创建成功")
    else:
        print("⚠️ Hotkey 已存在，跳过创建")

except Exception as e:
    print(f"❌ 密钥生成失败: {e}")
    exit(1)

print()

# 显示钱包信息
print("步骤 3/3: 获取钱包地址...")
print()
print("=" * 80)
print("钱包信息")
print("=" * 80)
print()
print(f"Coldkey 地址: {wallet.coldkey.ss58_address}")
print(f"Hotkey 地址: {wallet.hotkey.ss58_address}")
print()
print("=" * 80)
print("✅ 钱包创建成功！")
print("=" * 80)
print()

# 保存地址到文件
with open("wallet_addresses.txt", "w") as f:
    f.write(f"Wallet Name: {wallet_name}\n")
    f.write(f"Hotkey Name: {hotkey_name}\n")
    f.write(f"Coldkey Address: {wallet.coldkey.ss58_address}\n")
    f.write(f"Hotkey Address: {wallet.hotkey.ss58_address}\n")

print("📝 地址已保存到: wallet_addresses.txt")
print()

# 下一步说明
print("=" * 80)
print("接下来的步骤")
print("=" * 80)
print()
print("1️⃣ 获取测试网 TAO (选择以下方式之一):")
print()
print("   方式 A - 使用 faucet 命令 (推荐):")
print(f"   btcli wallet faucet --wallet.name {wallet_name} --subtensor.network test")
print()
print("   方式 B - Discord faucet:")
print("   1. 加入 Bittensor Discord")
print("   2. 访问 #testnet-faucet 频道")
print(f"   3. 发送: !faucet {wallet.coldkey.ss58_address}")
print()
print("   方式 C - 联系子网 owner 直接转账")
print()
print(f"2️⃣ 检查余额:")
print(f"   btcli wallet balance --wallet.name {wallet_name} --subtensor.network test")
print()
print(f"3️⃣ 注册到子网 108:")
print(f"   btcli subnet register --netuid 108 --subtensor.network test --wallet.name {wallet_name} --wallet.hotkey {hotkey_name}")
print()
print(f"4️⃣ 启动 Miner:")
print(f"   python3 neurons/miner_gemini.py --netuid 108 --subtensor.network test --wallet.name {wallet_name} --wallet.hotkey {hotkey_name} --logging.info")
print()
print("=" * 80)
