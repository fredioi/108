# StoryFi Bittensor Subnet - Setup Guide

## Quick Start

### For Miners

**Step 1: Choose Your Generation Method**

We support 3 generation methods:

| Method | Rewards | Setup Difficulty | Requirements |
|--------|---------|------------------|--------------|
| **API** (Default) | 0.5x | ⭐ Easy | API key only |
| **Local GPU** | 1.5x | ⭐⭐ Medium | GPU + 16GB VRAM |
| **vLLM** | 1.5x | ⭐⭐⭐ Advanced | GPU + Advanced setup |

**Step 2: Configure Generator**

```bash
# Copy example config
cp config/generator_config.yaml.example config/generator_config.yaml

# Edit the config file
nano config/generator_config.yaml
```

**Option A: API Mode (Easiest - 0.5x rewards)**

```yaml
generator:
  mode: "api"

  api:
    provider: "gemini"  # or "openai"
    api_key_env: "GEMINI_API_KEY"  # or "OPENAI_API_KEY"
    model: "gemini-2.0-flash-exp"  # or "gpt-4o"
```

Then set your API key:
```bash
export GEMINI_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

**Option B: Local GPU Mode (Best rewards - 1.5x)**

```yaml
generator:
  mode: "local"

  local:
    model_name: "Qwen/Qwen2.5-7B-Instruct"  # FREE, no HuggingFace login needed
    device: "cuda"
    quantization: "4bit"
    max_memory: "16GB"
```

**Important**: If you want to use Llama models (meta-llama/Llama-3.1-8B-Instruct):
1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Request access" and accept license
3. Create a HuggingFace token: https://huggingface.co/settings/tokens
4. Login: `huggingface-cli login` or `export HF_TOKEN="your-token"`

**Recommended FREE models (no login required):**
- `Qwen/Qwen2.5-7B-Instruct` (7B, excellent quality)
- `Qwen/Qwen2.5-14B-Instruct` (14B, better quality, needs 28GB VRAM)
- `mistralai/Mistral-7B-Instruct-v0.2` (7B, good quality)

**Step 3: Install Dependencies**

For API mode:
```bash
pip install -r requirements.txt
```

For Local mode:
```bash
pip install -r requirements.txt
pip install transformers accelerate bitsandbytes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 4: Run Miner**

For **MAINNET** (production):
```bash
python neurons/miner.py \
    --netuid 108 \
    --subtensor.network finney \
    --wallet.name your_miner \
    --wallet.hotkey default \
    --logging.info
```

For **TESTNET** (testing):
```bash
python neurons/miner.py \
    --netuid 108 \
    --subtensor.network test \
    --wallet.name your_miner \
    --wallet.hotkey default \
    --logging.info
```

### For Validators

**Step 1: Run Validator**

For **MAINNET**:
```bash
python neurons/validator.py \
    --netuid 108 \
    --subtensor.network finney \
    --wallet.name your_validator \
    --wallet.hotkey default \
    --logging.info
```

For **TESTNET**:
```bash
python neurons/validator.py \
    --netuid 108 \
    --subtensor.network test \
    --wallet.name your_validator \
    --wallet.hotkey default \
    --logging.info
```

## FAQ

**Q: Why do I get "Access to model meta-llama/Llama-3.1-8B-Instruct is restricted"?**

A: Llama models are "gated" by Meta - you need to:
1. Create a HuggingFace account
2. Request access on the model page
3. Login with `huggingface-cli login`

**Solution**: Use Qwen models instead (no login required):
- Change `model_name` to `"Qwen/Qwen2.5-7B-Instruct"` in your config

**Q: What's the difference between testnet and mainnet?**

A:
- **Testnet** (`--subtensor.network test`): For testing, fake TAO rewards
- **Mainnet** (`--subtensor.network finney`): Production, real TAO rewards

**Q: Which generation method should I use?**

A:
- **Have GPU?** → Use local mode (1.5x rewards)
- **No GPU?** → Use API mode (0.5x rewards, but easier)
- **Want max performance?** → Use vLLM mode (1.5x rewards, advanced)

**Q: Can I switch between API and Local mode?**

A: Yes! Just edit `config/generator_config.yaml` and change the `mode` field.

**Q: Do validators need to configure a generator?**

A: No. Validators only evaluate miner outputs, they don't generate stories themselves.

## Network Configuration Summary

| Network | Flag | Endpoint | Purpose |
|---------|------|----------|---------|
| Mainnet | `--subtensor.network finney` | `wss://entrypoint-finney.opentensor.ai:443` | Production |
| Testnet | `--subtensor.network test` | `wss://test.finney.opentensor.ai:443` | Testing |

## Model Quality Policy

Our subnet implements a quality-based reward system:

| Mode | Multiplier | Description |
|------|------------|-------------|
| Local GPU | 1.5x | Full decentralization bonus |
| vLLM | 1.5x | High-performance local inference |
| Custom | 1.0x | Advanced implementations |
| API | 0.5x | Cloud fallback |

Additionally, using high-quality models gets extra bonuses:
- GPT-4: 1.3x
- Claude-3: 1.3x
- Llama-70B: 1.5x
- Qwen-72B: 1.4x

**Total reward example**:
- Local mode (1.5x) × Llama-70B (1.5x) = **2.25x base rewards**
- API mode (0.5x) × GPT-4 (1.3x) = **0.65x base rewards**

See `validators/config/model_policy.yaml` for full policy.

## Support

- GitHub Issues: https://github.com/vya42/storyfi-subnet/issues
- Documentation: https://github.com/vya42/storyfi-subnet

## Architecture

```
Miners:
- Listen for story generation requests
- Support multiple backends (Local GPU, API, Custom)
- Return structured story content

Validators:
- Send generation tasks to miners
- Score responses (Technical + Structure + Content + Model Quality)
- Set weights on-chain based on performance
```

For detailed architecture, see `README.md`.
