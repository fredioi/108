"""
本地 Miner-Validator 通信测试
直接测试本地 miner，不依赖测试网 metagraph
"""

import asyncio
import bittensor as bt
from template.protocol import create_blueprint_synapse

async def test_local_miner():
    """直接测试本地 miner"""

    print("=" * 60)
    print("本地 Miner-Validator 通信测试")
    print("=" * 60)

    # 创建 dendrite（validator 用来发送请求）
    wallet = bt.wallet(name="storyfi_miner", hotkey="default")
    dendrite = bt.dendrite(wallet=wallet)

    # 本地 miner 地址
    miner_axon = bt.AxonInfo(
        version=4,
        ip="127.0.0.1",  # 本地地址
        port=8091,
        ip_type=4,
        hotkey="5F9gsRBgHrQdkG2f3fWP6NRkQREfwQdk3hGdsif2tdvKczTH",
        coldkey="5F9gsRBgHrQdkG2f3fWP6NRkQREfwQdk3hGdsif2tdvKczTH"
    )

    print(f"\n🎯 测试目标: {miner_axon.ip}:{miner_axon.port}")

    # 直接创建 synapse 对象，手动设置所有字段
    from template.protocol import StoryGenerationSynapse
    synapse = StoryGenerationSynapse(
        task_type="blueprint",
        user_input="Create a mystery story about a detective",
        name="StoryGenerationSynapse"  # 必须设置 name 字段！
    )
    synapse.validator_hotkey = wallet.hotkey.ss58_address

    # 打印 synapse 以便调试
    print(f"\n🔍 Synapse fields:")
    print(f"   name: '{synapse.name}'")
    print(f"   task_type: '{synapse.task_type}'")
    print(f"   user_input: '{synapse.user_input}'")
    print(f"   validator_hotkey: '{synapse.validator_hotkey}'")

    print(f"\n📨 发送请求:")
    print(f"   Task: {synapse.task_type}")
    print(f"   Input: {synapse.user_input}")

    # 发送请求
    try:
        print("\n⏳ 等待响应...")
        # 正确用法: dendrite() 而不是 dendrite.forward()
        responses = await dendrite(
            axons=[miner_axon],
            synapse=synapse,
            timeout=120
        )
        response = responses[0]

        print("\n" + "=" * 60)
        print("✅ 测试结果")
        print("=" * 60)

        if response and hasattr(response, 'output_data') and response.output_data:
            print(f"\n✅ 成功收到响应!")
            print(f"   生成时间: {response.generation_time:.2f}s")
            print(f"   Miner 版本: {response.miner_version}")
            print(f"   输出数据长度: {len(str(response.output_data))} 字符")
            print(f"\n📄 生成内容:")
            print(f"   {str(response.output_data)[:500]}...")

            print("\n" + "=" * 60)
            print("🎉 本地测试通过！Miner-Validator 通信正常")
            print("=" * 60)
            return True
        else:
            print(f"\n❌ 收到空响应或无效响应")
            print(f"   Response: {response}")
            return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_local_miner())
    exit(0 if result else 1)
