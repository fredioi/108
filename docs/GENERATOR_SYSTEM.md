# StoryFi Generator System

## Overview

The StoryFi miner uses a flexible generator system that supports multiple generation backends:
- **Local GPU models** (recommended, 1.5x reward multiplier)
- **Cloud APIs** (fallback, 0.5x reward multiplier)
- **Custom implementations** (1.0x reward multiplier)

This design follows Bittensor's core philosophy of **decentralization** - miners are free to choose any generation method that works best for their hardware and resources.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Miner (neurons/miner.py)               │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 GeneratorLoader                         │ │
│  │           (generators/loader.py)                        │ │
│  │                                                          │ │
│  │  Fallback Chain: Local → API → Error                   │ │
│  └─────────────┬────────────────────────────────────────┘  │
│                │                                             │
│         ┌──────┴──────┐                                     │
│         ▼             ▼                                     │
│  ┌──────────┐  ┌─────────┐  ┌────────────┐                │
│  │  Local   │  │   API   │  │   Custom   │                │
│  │Generator │  │Generator│  │ Generator  │                │
│  └──────────┘  └─────────┘  └────────────┘                │
│       │             │              │                        │
│       ▼             ▼              ▼                        │
│  ┌─────────┐  ┌─────────┐  ┌──────────────┐               │
│  │  Llama  │  │ OpenAI  │  │    Custom    │               │
│  │  3.1-8B │  │  Gemini │  │ Script/HTTP  │               │
│  └─────────┘  └─────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## Generator Types

### 1. Local GPU Generator (Recommended)

**Pros:**
- ✅ 1.5x reward multiplier from validators
- ✅ True decentralization (no API dependency)
- ✅ No API costs
- ✅ Full control over model and generation
- ✅ Privacy - data never leaves your machine

**Cons:**
- ❌ Requires GPU (6GB+ VRAM)
- ❌ Higher initial setup complexity
- ❌ Model download required (5-40GB)

**Supported Models:**
- `meta-llama/Llama-3.1-8B-Instruct` (8B, 6GB VRAM with 4bit)
- `meta-llama/Llama-3.1-70B-Instruct` (70B, 40GB VRAM with 4bit)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (56B, 30GB VRAM with 4bit)
- `Qwen/Qwen2.5-7B-Instruct` (7B, 5GB VRAM with 4bit)

**Configuration:**
```yaml
generator:
  mode: "local"
  local:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    device: "cuda"
    quantization: "4bit"
    max_memory: "16GB"
    use_flash_attention: true
```

### 2. API Generator (Fallback)

**Pros:**
- ✅ Easy setup (just add API key)
- ✅ No GPU required
- ✅ Immediate availability

**Cons:**
- ❌ 0.5x reward multiplier (half rewards!)
- ❌ API costs per request
- ❌ Centralization (depends on cloud provider)
- ❌ Privacy concerns (data sent to third party)

**Supported Providers:**
- OpenAI (GPT-4, GPT-4o, GPT-3.5-turbo)
- Google Gemini
- Anthropic Claude (coming soon)

**Configuration:**
```yaml
generator:
  mode: "api"
  api:
    provider: "openai"
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-4o-mini"
```

### 3. Custom Generator (Advanced)

**Pros:**
- ✅ 1.0x reward multiplier
- ✅ Full flexibility
- ✅ Can use any model or service

**Cons:**
- ❌ Requires custom implementation
- ❌ You maintain the code

**Configuration:**
```yaml
generator:
  mode: "custom"
  custom:
    script_path: "./custom/generate.py"
    endpoint: "http://localhost:8000/generate"
```

## Installation

### For Local GPU Generation

```bash
# Install transformers and quantization libraries
pip install transformers accelerate bitsandbytes

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: Flash Attention 2 for faster inference
pip install flash-attn
```

**Hardware Requirements:**

| Model | VRAM (4bit) | RAM | Storage |
|-------|-------------|-----|---------|
| Llama-3.1-8B | 6GB | 16GB | 5GB |
| Llama-3.1-70B | 40GB | 64GB | 40GB |
| Mixtral-8x7B | 30GB | 32GB | 30GB |
| Qwen2.5-7B | 5GB | 16GB | 4GB |

### For API Generation

```bash
# OpenAI
pip install openai
export OPENAI_API_KEY="sk-..."

# Gemini
pip install google-generativeai
export GEMINI_API_KEY="..."
```

## Configuration

### Step 1: Copy Example Config

```bash
cd /Users/xinyueyu/storyfi/storyfi-subnet
cp config/generator_config.yaml.example config/generator_config.yaml
cp .env.example .env
```

### Step 2: Edit Configuration

Edit `config/generator_config.yaml`:

```yaml
generator:
  mode: "local"  # or "api" or "custom"
  
  local:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    device: "cuda"
    quantization: "4bit"
    max_memory: "16GB"
    use_flash_attention: true
```

### Step 3: Set API Keys (if using API mode)

Edit `.env`:

```bash
OPENAI_API_KEY=sk-your-key-here
GEMINI_API_KEY=your-key-here
```

## Usage

### Start Miner with Local Generator

```bash
python neurons/miner.py \
    --netuid 42 \
    --wallet.name my_miner \
    --wallet.hotkey default \
    --logging.info
```

The miner will automatically:
1. Load `config/generator_config.yaml`
2. Initialize the configured generator (local/API)
3. Fall back to alternative if primary fails
4. Log which generator is being used

### Check Generator Status

When the miner starts, you'll see:

```
✅ Generator Mode: local
✅ Model: meta-llama/Llama-3.1-8B-Instruct
```

Or with fallback:

```
⚠️  Local mode failed, trying API fallback...
✅ Generator Mode: api
⚠️  Using fallback generator
```

## Fallback Chain

The system automatically tries fallback options:

```
Local GPU Model
    ↓ (if fails)
API Generator
    ↓ (if fails)
ERROR - No generator available
```

This ensures maximum uptime even if your preferred method fails.

## Reward Multipliers

Validators apply these multipliers to your rewards:

| Mode | Multiplier | Reason |
|------|-----------|---------|
| Local | **1.5x** | Promotes decentralization |
| API | **0.5x** | Discourages centralization |
| Custom | **1.0x** | Neutral |

**Example:**
- Base score: 80/100
- Local GPU: 80 × 1.5 = **120** (capped at 100)
- API: 80 × 0.5 = **40**

## Custom Generator Implementation

To implement a custom generator:

### Option 1: Script-based

Create `custom/generate.py`:

```python
#!/usr/bin/env python3
import json
import sys

# Read input from stdin
input_data = json.load(sys.stdin)

# Your generation logic here
result = your_generate_function(input_data)

# Output JSON to stdout
output = {
    "content": result
}
print(json.dumps(output))
```

Configure:

```yaml
generator:
  mode: "custom"
  custom:
    script_path: "./custom/generate.py"
```

### Option 2: HTTP Service

Run your own generation server:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    input_data = request.json
    result = your_generate_function(input_data)
    return jsonify({"content": result})

app.run(port=8000)
```

Configure:

```yaml
generator:
  mode: "custom"
  custom:
    endpoint: "http://localhost:8000/generate"
```

## Troubleshooting

### Issue: "Local generator failed"

**Possible causes:**
1. CUDA not installed or not detected
2. Insufficient VRAM
3. Model download failed

**Solutions:**
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check VRAM
nvidia-smi

# Clear cache and retry
rm -rf ~/.cache/huggingface/
```

### Issue: "API generator not available"

**Possible causes:**
1. API key not set
2. API key invalid
3. No internet connection

**Solutions:**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API manually
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Issue: "No generator available"

**This means both local and API failed!**

**Solutions:**
1. Check logs for specific error messages
2. Ensure at least one mode is properly configured
3. Test each mode individually

## Best Practices

### For Maximum Rewards

1. **Use local GPU generation** (1.5x multiplier)
2. Use 4bit quantization to reduce VRAM
3. Enable Flash Attention 2 if GPU supports it
4. Monitor generation quality and adjust models

### For Reliability

1. Configure both local and API modes
2. Set API key as backup (even if using local)
3. Monitor logs for fallback usage
4. Test your setup before going live

### For Cost Optimization

1. Prefer local models (no API costs)
2. If using API, use cheaper models (gpt-4o-mini)
3. Monitor API usage and costs
4. Consider custom generators for specific use cases

## Migration from v1.0.0

If you were using the old hardcoded OpenAI miner:

### Step 1: Create Config

```bash
cp config/generator_config.yaml.example config/generator_config.yaml
```

### Step 2: Move API Key to Config

Old (v1.0.0):
```python
# Hardcoded in miner.py
self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

New (v2.0.0):
```yaml
# config/generator_config.yaml
generator:
  mode: "api"
  api:
    provider: "openai"
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-4o-mini"
```

### Step 3: Restart Miner

```bash
# No code changes needed!
python neurons/miner.py --netuid 42 --wallet.name my_miner
```

The miner will automatically use the new generator system.

## Architecture Details

### Class Hierarchy

```
StoryGenerator (Abstract Base Class)
├── LocalModelGenerator
│   └── Uses HuggingFace Transformers
├── APIGenerator
│   ├── OpenAI support
│   ├── Gemini support
│   └── Custom endpoint support
└── CustomGenerator (Future)
    ├── Script execution
    └── HTTP client
```

### Key Files

- `generators/base.py` - Abstract base class
- `generators/local_generator.py` - Local GPU implementation
- `generators/api_generator.py` - API implementation
- `generators/loader.py` - Smart loader with fallback
- `config/generator_config.yaml` - Configuration file
- `neurons/miner.py` - Miner integration

### Data Flow

```
1. Validator sends request
   ↓
2. Miner receives StoryGenerationSynapse
   ↓
3. GeneratorLoader builds input_data
   ↓
4. Generator.generate(input_data)
   ↓
5. Returns {"generated_content": "..."}
   ↓
6. Miner parses and validates JSON
   ↓
7. Miner returns synapse.output_data
   ↓
8. Validator receives and scores
```

## Future Enhancements

- [ ] vLLM support for faster local inference
- [ ] Model caching and warm-up
- [ ] Multi-GPU support
- [ ] Anthropic Claude API support
- [ ] Custom generator registry
- [ ] Generator health monitoring
- [ ] Automatic model selection based on VRAM
- [ ] Prompt templates system

## References

- [SoulX Architecture Analysis](./docs/analysis/SOULX_ARCHITECTURE_ANALYSIS.md)
- [Bittensor Decentralization Philosophy](https://docs.bittensor.com)
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)

---

**Built with decentralization in mind** 🏗️
