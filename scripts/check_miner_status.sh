#!/bin/bash

echo "========================================"
echo "StoryFi Miner 状态检查"
echo "========================================"
echo ""

echo "📍 1. Miner 钱包信息"
echo "----------------------------------------"
btcli wallet overview \
    --wallet.name storyfi_miner \
    --subtensor.network test \
    --netuid 108 2>/dev/null | head -n 20

echo ""
echo "📊 2. 子网 108 状态"
echo "----------------------------------------"
btcli subnet list --subtensor.network test 2>/dev/null | grep -A 1 "NETUID" | head -n 2
btcli subnet list --subtensor.network test 2>/dev/null | grep "108"

echo ""
echo "💰 3. 钱包余额"
echo "----------------------------------------"
btcli wallet balance \
    --wallet.name storyfi_miner \
    --subtensor.network test 2>/dev/null

echo ""
echo "⚡ 4. Miner 进程状态"
echo "----------------------------------------"
if pgrep -f "miner_gemini.py" > /dev/null; then
    echo "✅ Miner 正在运行"
    echo "PID: $(pgrep -f miner_gemini.py)"
else
    echo "❌ Miner 未运行"
fi

echo ""
echo "========================================"
echo "最后更新: $(date)"
echo "========================================"
