"""
导入钱包使用助记词
"""
import bittensor as bt

print("=" * 80)
print("导入 Bittensor 钱包")
print("=" * 80)
print()

# 助记词
mnemonic = "vacuum party final home zone quick talent yard click excuse carbon siege"

# 钱包配置
wallet_name = "storyfi_miner"
hotkey_name = "default"

print(f"钱包名称: {wallet_name}")
print(f"Hotkey: {hotkey_name}")
print()

# 创建钱包对象
wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)

try:
    # 从助记词恢复 coldkey
    print("步骤 1/2: 导入 Coldkey...")
    wallet.regenerate_coldkey(mnemonic=mnemonic, use_password=False, overwrite=True)
    print(f"✅ Coldkey 导入成功")
    print(f"   地址: {wallet.coldkey.ss58_address}")
    print()

    # 从助记词恢复 hotkey
    print("步骤 2/2: 导入 Hotkey...")
    wallet.regenerate_hotkey(mnemonic=mnemonic, use_password=False, overwrite=True)
    print(f"✅ Hotkey 导入成功")
    print(f"   地址: {wallet.hotkey.ss58_address}")
    print()

    print("=" * 80)
    print("✅ 钱包导入成功！")
    print("=" * 80)
    print()
    print("钱包信息:")
    print(f"  Coldkey: {wallet.coldkey.ss58_address}")
    print(f"  Hotkey: {wallet.hotkey.ss58_address}")
    print()

    # 验证地址
    expected_address = "5F9gsRBgHrQdkG2f3fWP6NRkQREfwQdk3hGdsif2tdvKczTH"
    if wallet.hotkey.ss58_address == expected_address:
        print(f"✅ 地址匹配！这是你的钱包")
    else:
        print(f"⚠️ 地址不匹配")
        print(f"   期望: {expected_address}")
        print(f"   实际: {wallet.hotkey.ss58_address}")

    print()
    print("下一步: 启动 Miner")

except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
