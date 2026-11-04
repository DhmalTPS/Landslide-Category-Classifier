"""Download the classifier backbones into the image at build time (sizes are off on the free CPU Space)."""
from huggingface_hub import snapshot_download

MODELS = ["facebook/dinov2-base", "facebook/dinov2-large", "google/siglip2-base-patch16-224"]
for m in MODELS:
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    print("ok", m, flush=True)
