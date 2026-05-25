# neural-me
Personal LoRA training pipeline to generate AI images of myself using Stable Diffusion.

## What works
- Image preparation and resizing
- Auto-captioning with BLIP
- LoRA training with DreamBooth (1500 steps, bf16, T4 GPU)
- Image generation with Realistic Vision v5

## What needs improvement
- Likeness accuracy needs more training photos (38 is borderline)
- Need more steps (try 2000-3000 next time)
- Prompt engineering for consistent results

## Stack
- Python, HuggingFace Diffusers, Google Colab T4
- Base model: Realistic Vision V5.1
- Training: DreamBooth LoRA, 1500 steps
