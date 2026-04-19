# 🔬 Vision Privacy & Identity Lab

Fine-tune a Stable Diffusion model on your own portrait images using **Dreambooth + LoRA** — with automated image preprocessing, BLIP captioning, tattoo detection, and skin smoothing.

---

## Repository layout

```
IA-/
├── preprocessing.py      # Image preprocessing pipeline (crop, smooth, inpaint, caption)
├── train_colab.ipynb     # Google Colab training notebook (Dreambooth + LoRA)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess raw images

```bash
python preprocessing.py \
    --input_dir  raw_images/ \
    --output_dir dataset/ \
    --token      "sks person" \
    --size       512 \
    --smooth_skin
```

Add `--inpaint_tattoos` to also remove visible tattoos via Stable Diffusion inpainting (requires a CUDA GPU with ≥ 10 GB VRAM).

| Flag | Description |
|------|-------------|
| `--input_dir` | Folder containing raw portrait images |
| `--output_dir` | Destination for processed images + caption `.txt` files |
| `--token` | Unique activation token prepended to every caption (default: `sks person`) |
| `--size` | Output resolution in pixels, square (default: `512`) |
| `--smooth_skin` | Apply bilateral-filter skin smoothing |
| `--inpaint_tattoos` | Detect & inpaint tattoo regions via SD inpainting |

The pipeline:
1. **Face detection** (MediaPipe) → centre-crop to a padded square around the face
2. **Skin smoothing** *(optional)* – bilateral filter preserves edges while softening texture
3. **Tattoo inpainting** *(optional)* – HSV-based heuristic mask + SD inpaint pipeline
4. **BLIP captioning** – generates a natural-language caption, then prepends your unique token

Output per image: `<stem>.png` + `<stem>.txt` caption file ready for Dreambooth training.

### 3. Train in Google Colab

Open `train_colab.ipynb` in Google Colab (GPU runtime required):

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ap-xlr8/IA-/blob/main/train_colab.ipynb)

Fill in the **§ 2 – Configuration** cell with your paths and token, then run all cells.

**Memory-saving features enabled by default:**
- `gradient_checkpointing` – trades compute for VRAM
- `xformers` memory-efficient attention
- 8-bit Adam optimiser (`bitsandbytes`)
- `fp16` mixed-precision training

### 4. Generate images

After training finishes, the **§ 5 – Inference** cell loads the fine-tuned LoRA weights and generates portrait images using your custom prompts, e.g.:

```
a professional headshot of sks person, studio lighting, 4k
```

---

## Unique activation token

The unique token (default `sks`) is a rare word-piece that the model learns to associate exclusively with your subject.
It is automatically injected at the beginning of every caption during preprocessing and used as the `--instance_prompt` during training.

---

## Requirements

| Library | Purpose |
|---------|---------|
| `mediapipe` | Face detection |
| `Pillow` / `opencv-python` | Image I/O & filtering |
| `transformers` | BLIP captioning |
| `diffusers` | Stable Diffusion inpainting & training |
| `accelerate` | Distributed / mixed-precision training |
| `xformers` | Memory-efficient attention |
| `bitsandbytes` | 8-bit Adam optimiser |

---

## License

This project is provided for personal research and educational use only.
Always obtain proper consent before processing images of real individuals.
