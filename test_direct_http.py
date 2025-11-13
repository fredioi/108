"""
直接用HTTP POST测试miner（绕过Bittensor dendrite）
"""
import requests
import json

MINER_URL = "http://127.0.0.1:8091"

print("=" * 60)
print("直接HTTP POST测试")
print("=" * 60)

# 构造一个标准的Bittensor synapse HTTP请求
headers = {
    "Content-Type": "application/json",
    # Bittensor使用的headers
    "bt-synapse-name": "StoryGenerationSynapse",
}

# Synapse数据
data = {
    "name": "StoryGenerationSynapse",
    "task_type": "blueprint",
    "user_input": "Create a mystery story about a detective",
    "protocol_version": "3.1.0"
}

print(f"\n📨 发送POST请求到 {MINER_URL}")
print(f"   Headers: {headers}")
print(f"   Data: {json.dumps(data, indent=2)}")

try:
    response = requests.post(
        MINER_URL,
        headers=headers,
        json=data,
        timeout=120
    )

    print(f"\n✅ 响应状态码: {response.status_code}")
    print(f"   响应Headers: {dict(response.headers)}")
    print(f"\n📄 响应内容:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
