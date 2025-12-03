# API Key Configuration Guide

## Quick Start

### Option 1: Using `.env` File (Recommended)

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys:**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Use in your code:**
   ```python
   from spatok.vlms.vlm_api import OpenAIVLM
   from spatok.vlms.config import get_config
   
   config = get_config()
   vlm = OpenAIVLM(api_key=config.get_openai_key())
   ```

**Advantages:**
- ✅ One file to sync across machines (just copy `.env`)
- ✅ Per-project configuration
- ✅ Already in `.gitignore` (won't be committed)
- ✅ Easy to update

### Option 2: Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
export XAI_API_KEY="xai-..."
```

Then reload: `source ~/.bashrc`

**Advantages:**
- ✅ Works system-wide
- ✅ Standard approach
- ❌ Clutters shell config
- ❌ Need to update on each machine

### Option 3: Config File in Home Directory

Create `~/.vlm_config.json`:

```json
{
  "openai_api_key": "sk-...",
  "anthropic_api_key": "sk-ant-...",
  "google_api_key": "AIza...",
  "xai_api_key": "xai-..."
}
```

Set secure permissions:
```bash
chmod 600 ~/.vlm_config.json
```

**Advantages:**
- ✅ Centralized across all projects
- ✅ Won't be accidentally committed
- ❌ Need to create on each machine

---

## Setup on Multiple Machines

### Recommended Workflow

1. **On first machine:**
   ```bash
   cd /home/miao/spatial-reasoning
   cp .env.example .env
   # Edit .env with your keys
   ```

2. **On additional machines:**
   ```bash
   # Copy .env from first machine
   scp machine1:~/spatial-reasoning/.env ~/spatial-reasoning/.env
   ```

That's it! The `.env` file will work on all machines.

---

## Configuration Priority

The config loader checks in this order:

1. `.env` file in project root
2. Environment variables
3. `~/.vlm_config.json`

Higher priority sources override lower ones.

---

## API Key Management

### Get Your API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/settings/keys
- **Google**: https://aistudio.google.com/app/apikey
- **X.AI**: https://console.x.ai/
- **Reka**: https://platform.reka.ai/

### Security Best Practices

1. **Never commit API keys to git** (already in `.gitignore`)
2. **Use restrictive file permissions:**
   ```bash
   chmod 600 .env
   chmod 600 ~/.vlm_config.json
   ```
3. **Rotate keys periodically**
4. **Use separate keys for development vs production**
5. **Monitor API usage** on provider dashboards

---

## Usage Examples

### Basic Usage

```python
from spatok.vlms.vlm_api import OpenAIVLM, ClaudeVLM
from spatok.vlms.config import get_config

# Automatic configuration loading
config = get_config()

# Use with any VLM
vlm = OpenAIVLM(api_key=config.get_openai_key())
response = vlm.call("What is AI?")
print(response)
```

### Check Configuration Status

```python
from spatok.vlms.config import get_config

config = get_config()
print(config)  # Shows which keys are configured

# Check individual keys
if config.get_openai_key():
    print("✓ OpenAI configured")
else:
    print("✗ OpenAI not configured")
```

### Runtime Configuration

```python
from spatok.vlms.config import get_config

config = get_config()

# Set keys at runtime (not persisted)
config.set('openai', 'sk-...')

# Save to file for future use
config.save_to_file()  # Saves to ~/.vlm_config.json
```

---

## Troubleshooting

### "API key not configured" error

1. Check if `.env` exists: `ls -la .env`
2. Check file contents: `cat .env` (be careful not to expose keys)
3. Verify environment variables: `env | grep API_KEY`
4. Check config status:
   ```python
   from spatok.vlms.config import get_config
   print(get_config())
   ```

### `.env` file not being read

1. Ensure `python-dotenv` is installed: `pip install python-dotenv`
2. Verify `.env` is in project root (same directory as `setup.py`)
3. Check file permissions: `ls -la .env`

### Keys working on one machine but not another

1. Copy `.env` file to the new machine
2. Or set environment variables on new machine
3. Or create `~/.vlm_config.json` on new machine
