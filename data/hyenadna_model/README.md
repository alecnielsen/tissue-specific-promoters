# HyenaDNA Model Files

This directory contains the HyenaDNA model files needed for Modal deployment.

## Setup

Download the model files from HuggingFace:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download model files
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'LongSafari/hyenadna-medium-160k-seqlen-hf',
    local_dir='data/hyenadna_model',
    local_dir_use_symlinks=False  # Copy files, don't symlink
)
"
```

## Why Local Files?

Modal's HuggingFace cache proxy has issues with malformed URLs. By using local
model files that are copied into the Modal image, we avoid these issues.

## Required Files

After download, this directory should contain:
- `config.json` - Model configuration
- `configuration_hyena.py` - HyenaDNA config class
- `modeling_hyena.py` - HyenaDNA model class
- `model.safetensors` - Model weights (~57MB)

Note: `model.safetensors` is gitignored due to size. You must download it manually.
