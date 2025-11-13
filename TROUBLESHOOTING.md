# StoryFi Subnet Troubleshooting Guide

## 🔴 Common Issue: Validator Cannot Connect to Miners (0.0.0.0 Problem)

### Symptoms
- Validator logs show: `Cannot connect to host 0.0.0.0:0`
- Validator stats show: `Successful: 0`
- All miners marked as "Invalid response"

### Root Cause
Miners register to the blockchain with `0.0.0.0` IP address instead of their actual public IP.

This happens when:
- Running on cloud servers (AWS, GCP, Azure)
- Behind NAT/firewall
- In Docker containers
- Bittensor cannot auto-detect external IP

### Solution for Miners

#### Option 1: Specify External IP (Recommended)
```bash
# Get your public IP first
curl ifconfig.me

# Start miner with external IP
python neurons/miner.py \
    --netuid 108 \
    --wallet.name your_wallet \
    --wallet.hotkey default \
    --axon.external_ip YOUR_PUBLIC_IP \
    --logging.info
```

Example:
```bash
python neurons/miner.py \
    --netuid 108 \
    --wallet.name miner1 \
    --wallet.hotkey default \
    --axon.external_ip 195.211.96.76 \
    --logging.info
```

#### Option 2: Auto-detect (Works in Most Cases)
If your server can auto-detect external IP:
```bash
python neurons/miner.py \
    --netuid 108 \
    --wallet.name your_wallet \
    --wallet.hotkey default \
    --logging.info
```

Check the startup logs for:
```
📡 Axon IP: YOUR_IP_HERE
📡 Axon Port: 8091
✅ Axon registered: YOUR_IP_HERE:8091
```

If you see `0.0.0.0`, use Option 1.

### Verification

**After starting your miner, verify it's registered correctly:**

```bash
# Install btcli if not already installed
pip install bittensor

# Check metagraph
btcli subnet metagraph --netuid 108

# Look for your hotkey and verify:
# - IP is not 0.0.0.0
# - Port is correct (default: 8091)
```

**Test direct connection:**
```bash
# Replace with your actual IP and port
curl http://YOUR_IP:8091/health

# Should return some response (not connection refused)
```

### For Validators

The validator now automatically filters out invalid miners (0.0.0.0 addresses).

If you're running a validator and still seeing connection issues:

1. **Enable debug logging:**
```bash
python neurons/validator.py \
    --netuid 108 \
    --wallet.name your_validator \
    --wallet.hotkey default \
    --logging.debug
```

2. **Check filtered miners:**
Look for logs like:
```
Filtered out N invalid axons (0.0.0.0 or invalid config)
```

3. **Verify you can connect to valid miners:**
```bash
# Get a valid miner's IP from metagraph
btcli subnet metagraph --netuid 108

# Test connection
curl http://MINER_IP:MINER_PORT/health
```

## 🔴 Issue: All Miners Returning Format Errors

### Symptoms
- Miner logs show: `Format mismatch: task_type must return JSON object, not array`
- Validator gives 0 scores for all responses

### Solution
1. Update to latest code:
```bash
cd /path/to/storyfi-subnet
git pull
```

2. Restart all miner processes (important!):
```bash
# Kill old processes
pkill -f "python.*miner.py"

# Start fresh
python neurons/miner.py [your args]
```

3. Verify prompt templates are updated:
```bash
grep "Output as a single JSON object" generators/prompt_templates.py
```

Should show the updated prompt format.

## 🔴 Issue: Stake Not Improving Scores

### Understanding the Scoring System

Validator uses a composite scoring formula:
```
Final Score = 15% Stake + 75% Quality + 10% History
```

**Important:**
- Stake alone won't help if quality score is 0
- You need valid responses first, then stake will boost your score
- Quality score comes from: Technical (30) + Structure (40) + Content (30)

### Debugging Low Scores

1. **Check if your responses are valid:**
```bash
# Look for score breakdowns in validator logs
grep "📊 Miner" validator.log | grep "UID_YOUR_MINER"
```

2. **Common reasons for 0 score:**
   - Format errors (List instead of Dict)
   - Connection failures
   - Plagiarism detection (>95% similarity)
   - Model quality penalty (no model_info)

3. **Improve quality score:**
   - Use recommended models (local/vllm: 1.5x, custom: 1.0x, api: 0.5x)
   - Ensure proper JSON format
   - Avoid template reuse
   - Include model_info in responses

## Need Help?

1. Check GitHub Issues: https://github.com/vya42/storyfi-subnet/issues
2. Join Discord: [Link]
3. Review logs carefully - they usually contain the answer!
