# Model Cache Directory Usage Guide

## Overview

All VLM models now support specifying a custom cache directory where model checkpoints will be downloaded and stored. This is useful for:

- Organizing models in a specific location
- Using a shared model cache across multiple projects
- Storing models on a drive with more space
- Managing disk space more efficiently

## Directory Structure

When you specify a cache directory, models are saved in the following structure:

```
<cache_dir>/
└── hub/
    └── models--<org>--<model-name>/
        ├── snapshots/
        │   └── <commit-hash>/
        │       ├── model.safetensors (or model-*.safetensors for sharded models)
        │       ├── config.json
        │       ├── tokenizer_config.json
        │       └── ... (other model files)
        └── refs/
            └── main
```

### Example:
If you set `cache_dir="/data/vlm_models"` and download `Qwen/Qwen3-VL-8B-Instruct`:
```
/data/vlm_models/
└── hub/
    └── models--Qwen--Qwen3-VL-8B-Instruct/
        ├── snapshots/
        │   └── abc123.../
        │       ├── model-00001-of-00004.safetensors
        │       ├── model-00002-of-00004.safetensors
        │       ├── model-00003-of-00004.safetensors
        │       ├── model-00004-of-00004.safetensors
        │       ├── config.json
        │       └── ...
        └── refs/
            └── main
```

## Usage

### Option 1: Command Line Argument (Recommended for Testing)

```bash
# Test with a specific cache directory
python spatok/test/test_vlm_local.py \
    --mode noise \
    --model qwen3 \
    --model_cache_dir /data/vlm_models

# Test all models with custom cache
python spatok/test/test_vlm_local.py \
    --mode noise \
    --model all \
    --model_cache_dir /data/vlm_models
```

### Option 2: Environment Variables (Recommended for Production)

Set these before running your code:

```bash
export HF_HOME=/data/vlm_models
export TRANSFORMERS_CACHE=/data/vlm_models/hub

python spatok/test/test_vlm_local.py --mode noise --model qwen3
```

### Option 3: Programmatic Usage

```python
from spatok.vlms.vlm_local import Qwen3VLM

# Specify cache directory when initializing the model
vlm = Qwen3VLM(
    model_path="Qwen/Qwen3-VL-8B-Instruct",
    cache_dir="/data/vlm_models"
)

response = vlm.call(
    text_prompt="What's in this image?",
    image_input="path/to/image.jpg"
)
```

## Default Behavior

If no cache directory is specified, HuggingFace defaults are used:

- **Linux/Mac**: `~/.cache/huggingface/`
- **Windows**: `C:\Users\<username>\.cache\huggingface\`

## Best Practices

1. **Use Absolute Paths**: Always specify absolute paths for cache directories
   ```bash
   --model_cache_dir /data/vlm_models  # Good
   --model_cache_dir ~/vlm_models      # Also works (~ is expanded)
   --model_cache_dir ./models          # Avoid relative paths
   ```

2. **Shared Cache**: Use the same cache directory across projects to avoid duplicate downloads
   ```bash
   export HF_HOME=/data/shared/huggingface_cache
   ```

3. **Check Disk Space**: Large models can be 10-100GB+
   ```bash
   # Qwen3-VL-8B: ~16GB
   # Qwen3-VL-70B: ~140GB
   # Check available space first
   df -h /data/vlm_models
   ```

4. **Permissions**: Ensure the cache directory is writable
   ```bash
   mkdir -p /data/vlm_models
   chmod 755 /data/vlm_models
   ```

## Troubleshooting

### Issue: "Permission denied" when downloading

**Solution**: Check directory permissions
```bash
ls -ld /data/vlm_models
chmod 755 /data/vlm_models
```

### Issue: Models downloading to wrong location

**Solution**: Verify environment variables aren't set elsewhere
```bash
env | grep HF_HOME
env | grep TRANSFORMERS_CACHE
unset HF_HOME TRANSFORMERS_CACHE  # If needed
```

### Issue: Out of disk space

**Solution**: Check model size requirements before downloading
```bash
# Check space
df -h /data/vlm_models

# Clear old models if needed
rm -rf /data/vlm_models/hub/models--<old-model>/
```

## Model Size Reference

| Model | Approximate Size |
|-------|------------------|
| Qwen2-VL-7B | ~14 GB |
| Qwen2.5-VL-7B | ~14 GB |
| Qwen3-VL-8B | ~16 GB |
| Qwen3-VL-70B | ~140 GB |
| LLaVA-1.5-7B | ~13 GB |
| InternVL2-8B | ~16 GB |
| Phi-3-Vision | ~8 GB |
| MiniCPM-V | ~9 GB |

**Note**: Sizes are approximate and include model weights + tokenizer + config files.
