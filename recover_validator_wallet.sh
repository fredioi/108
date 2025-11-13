#!/bin/bash
# 恢复 validator 钱包的脚本

# 助记词
MNEMONIC="before fatal buyer come lazy chef slight unit shift solar ginger spend"

# 使用 expect 来自动化交互式输入
expect << 'EXPECT_SCRIPT'
set timeout -1

# 恢复 coldkey
spawn btcli wallet regen_coldkey --wallet.name storyfi_validator

expect "Enter the path for the wallets directory*"
send "\r"

expect "Enter mnemonic*"
send "before fatal buyer come lazy chef slight unit shift solar ginger spend\r"

expect "Specify password for key encryption*"
send "\r"

expect "Retype your password*"
send "\r"

expect eof
EXPECT_SCRIPT

echo "✅ Validator coldkey created"

# 创建 hotkey (使用相同助记词临时)
expect << 'EXPECT_SCRIPT'
set timeout -1

spawn btcli wallet regen_hotkey --wallet.name storyfi_validator --wallet.hotkey default

expect "Enter the path for the wallets directory*"
send "\r"

expect "Enter mnemonic*"
send "before fatal buyer come lazy chef slight unit shift solar ginger spend\r"

expect "Specify password for key encryption*"
send "\r"

expect "Retype your password*"
send "\r"

expect eof
EXPECT_SCRIPT

echo "✅ Validator hotkey created"
echo "✅ Validator wallet (storyfi_validator) created successfully!"
