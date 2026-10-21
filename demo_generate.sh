#!/usr/bin/env bash
# LTX-2 demo — text-to-video with synchronized audio (DistilledPipeline).
#
# Prerequisites (GPU machine, Linux + CUDA, ~24 GB+ VRAM recommended):
#   uv sync --extra natten
#   hf auth login   # needs a Read token with "read gated repos" scope
#   hf download Lightricks/LTX-2.5 \
#       diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
#       text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
#       vae/ltx-2.5-video-vae-bf16.safetensors \
#       vae/ltx-2.5-audio-vae-bf16.safetensors \
#       latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
#       --local-dir models/ltx-2.5
#
# Usage:
#   ./demo_generate.sh [output.mp4] [extra ltx_pipelines flags...]
set -euo pipefail

OUTPUT="${1:-output.mp4}"
shift || true

uv run python -m ltx_pipelines.distilled \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --spatial-upsampler-path models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --num-frames 121 \
    --seed 42 \
    --output-path "$OUTPUT" \
    --prompt "A medium close-up shot features a man with a beard, wearing a green and white baseball cap and a light blue shirt over a white t-shirt. He looks directly into the camera with a focused expression, sniffs briefly, then speaks in a deep, satisfied voice: 'I think it's so good.' The camera is static with shallow depth of field; a softly blurred beige wall sits behind him. After a pause he continues: 'So good. So good. This got to be the best tool I've ever seen.'" \
    "$@"
