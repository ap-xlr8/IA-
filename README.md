# Vision Privacy & Identity Lab

End-to-end pipeline for privacy-safe image preprocessing, DreamBooth / LoRA training, and custom image generation.

## Project Structure

```
IA-/
├── preprocessing.py            # Module 1 – image cleaning & privacy masks
├── caption_cleaner.py          # Module 2 – BLIP captioning + activation token
├── 01_image_preprocessing.ipynb   # Colab notebook – Module 1
├── 02_training_setup.ipynb        # Colab notebook – Module 2
├── 03_image_generation.ipynb      # Colab notebook – Module 3
└── requirements.txt
```

## Modules

### Module 1 – Image Preprocessing (`preprocessing.py`)

Cleans a folder of raw images for training:

| Step | Description |
|------|-------------|
| Scan | Recursively finds all `.jpg/.png/.webp` files |
| Resize | Centre-crop to 512 × 512 (configurable) |
| Face mask | MediaPipe face detection + Gaussian blur |
| Tattoo mask | Bounding-box inpainting (stub; plug in a custom model) |
| Export | Saves clean PNGs to the output directory |

```python
from preprocessing import adaptar_imagenes_crudas

adaptar_imagenes_crudas(
    input_dir="data/raw",
    output_dir="data/processed",
    target_size=(512, 512),
    apply_face_mask=True,
    apply_tattoo_mask=True,
)
```

### Module 2 – Training Setup (`caption_cleaner.py`)

Generates DreamBooth / LoRA-ready captions:

- **Unique activation token** – auto-generated UUID4-based string, or loaded from
  the `VLAB_ACTIVATION_TOKEN` environment variable.
- **BLIP auto-captioning** – uses `Salesforce/blip-image-captioning-base`.
- **Caption cleaning** – removes boilerplate phrases.
- Saves one `.txt` file per image (standard DreamBooth convention).

```python
from caption_cleaner import generar_token_activacion, procesar_directorio

token = generar_token_activacion()        # e.g. "vplid3f9a2b1c"
procesar_directorio("data/processed", token=token, clase="person", device="cuda")
```

### Module 3 – Image Generation (notebook)

`03_image_generation.ipynb` loads the trained LoRA adapter on top of
Stable Diffusion and generates new images using structured prompts that
include the activation token.

## VRAM Management (Colab)

`02_training_setup.ipynb` enables the following memory-saving options by
default when launching training:

| Option | Flag |
|--------|------|
| Gradient Checkpointing | `--gradient_checkpointing` |
| xformers attention | `--enable_xformers_memory_efficient_attention` |
| 8-bit Adam | `--use_8bit_adam` |
| FP16 mixed precision | `--mixed_precision fp16` |

## Quick Start (local)

```bash
pip install -r requirements.txt

# 1. Preprocess images
python preprocessing.py data/raw data/processed

# 2. Generate captions
python caption_cleaner.py data/processed --output-dir data/captions --device cpu
```

## Quick Start (Colab)

Open the notebooks in order:

1. `01_image_preprocessing.ipynb`
2. `02_training_setup.ipynb`
3. `03_image_generation.ipynb`

Each notebook installs its own dependencies and mounts Google Drive automatically.
