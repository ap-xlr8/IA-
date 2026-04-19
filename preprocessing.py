"""
Vision Privacy & Identity Lab – Preprocessing Pipeline
=======================================================
Adapts raw portrait images for Dreambooth / LoRA fine-tuning:

  1. Face detection & centre-crop (MediaPipe)
  2. Optional skin-smoothing (bilateral filter)
  3. Optional tattoo inpainting (Stable-Diffusion inpaint pipeline)
  4. Auto-captioning with BLIP + unique-token injection
  5. Saves processed images + captions to an output directory

Usage
-----
  python preprocessing.py \\
      --input_dir  raw_images/ \\
      --output_dir dataset/ \\
      --token      "sks person" \\
      --size       512 \\
      --smooth_skin \\
      --inpaint_tattoos

Requirements (see requirements.txt):
  pip install mediapipe Pillow transformers torch diffusers accelerate
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Face detection helpers (MediaPipe)
# ---------------------------------------------------------------------------

_face_detector: Optional[mp.solutions.face_detection.FaceDetection] = None


def _get_face_detector() -> mp.solutions.face_detection.FaceDetection:
    global _face_detector
    if _face_detector is None:
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,          # long-range model
            min_detection_confidence=0.5,
        )
    return _face_detector


def detect_face_bbox(
    image_rgb: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) of the largest face, or None if no face found."""
    detector = _get_face_detector()
    results = detector.process(image_rgb)
    if not results.detections:
        return None

    h, w = image_rgb.shape[:2]
    best = max(
        results.detections,
        key=lambda d: d.location_data.relative_bounding_box.width
        * d.location_data.relative_bounding_box.height,
    )
    bbox = best.location_data.relative_bounding_box
    x = max(0, int(bbox.xmin * w))
    y = max(0, int(bbox.ymin * h))
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)
    return (x, y, bw, bh)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def face_centered_crop(
    image: Image.Image,
    target_size: int = 512,
    padding_factor: float = 1.8,
) -> Image.Image:
    """
    Crop the image to a square centred on the detected face.
    Falls back to a centre crop if no face is detected.

    padding_factor > 1 expands the crop beyond the tight face bounding box
    so the head / neck / shoulders are included.
    """
    img_rgb = np.array(image.convert("RGB"))
    bbox = detect_face_bbox(img_rgb)

    w, h = image.size

    if bbox is not None:
        fx, fy, fw, fh = bbox
        cx = fx + fw // 2
        cy = fy + fh // 2
        # Make a padded square around the face centre
        side = int(max(fw, fh) * padding_factor)
        side = min(side, w, h)           # never exceed image dimensions
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        # Shift back if we went out of bounds
        if x2 - x1 < side:
            x1 = max(0, x2 - side)
        if y2 - y1 < side:
            y1 = max(0, y2 - side)
    else:
        # Centre-crop fallback
        side = min(w, h)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        x2 = x1 + side
        y2 = y1 + side

    cropped = image.crop((x1, y1, x2, y2))
    return cropped.resize((target_size, target_size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Skin smoothing
# ---------------------------------------------------------------------------

def smooth_skin(image: Image.Image, d: int = 9, sigma: int = 75) -> Image.Image:
    """
    Apply a bilateral filter to smooth skin tones while preserving edges.
    d         – filter diameter (larger = slower but smoother)
    sigma     – both sigmaColor and sigmaSpace
    """
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    smoothed = cv2.bilateralFilter(img_bgr, d, sigma, sigma)
    return Image.fromarray(cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Tattoo detection & inpainting
# ---------------------------------------------------------------------------

_inpaint_pipe = None  # loaded lazily


def _load_inpaint_pipeline():
    global _inpaint_pipe
    if _inpaint_pipe is None:
        try:
            from diffusers import StableDiffusionInpaintPipeline
            import torch

            _inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _inpaint_pipe = _inpaint_pipe.to(device)
            if device == "cuda":
                try:
                    _inpaint_pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
        except ImportError as exc:
            raise RuntimeError(
                "diffusers and torch are required for tattoo inpainting. "
                "Install them with: pip install diffusers torch"
            ) from exc
    return _inpaint_pipe


def build_tattoo_mask(image: Image.Image, threshold: int = 40) -> Image.Image:
    """
    Heuristic tattoo mask: detects high-contrast non-skin-coloured ink regions.

    This is a lightweight approximation using HSV colour space. For
    production use, replace with a dedicated segmentation model trained
    on tattoo data.

    Returns a grayscale PIL Image where white (255) = tattoo region.
    """
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Skin colour range in HSV
    lower_skin = np.array([0,   20, 70],  dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask  = cv2.inRange(img_hsv, lower_skin, upper_skin)

    # Dark-ink mask (low value + any hue)
    lower_ink = np.array([0,   0,   0], dtype=np.uint8)
    upper_ink = np.array([180, 255, threshold], dtype=np.uint8)
    ink_mask  = cv2.inRange(img_hsv, lower_ink, upper_ink)

    # Combine: ink regions that are *not* simply dark shadow on skin
    tattoo_raw = cv2.bitwise_and(ink_mask, cv2.bitwise_not(skin_mask))

    # Morphological clean-up: remove noise, fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tattoo_clean = cv2.morphologyEx(tattoo_raw, cv2.MORPH_CLOSE, kernel)
    tattoo_clean = cv2.morphologyEx(tattoo_clean, cv2.MORPH_OPEN,  kernel)

    # Dilate slightly so the inpainter has context
    tattoo_dilated = cv2.dilate(tattoo_clean, kernel, iterations=2)

    return Image.fromarray(tattoo_dilated)


def inpaint_tattoos(
    image: Image.Image,
    prompt: str = "smooth skin, natural skin tone, no tattoos",
    num_inference_steps: int = 30,
) -> Image.Image:
    """
    Detect tattoos via heuristic mask and inpaint them away.
    Returns the original image unchanged if no tattoo region is found.
    """
    mask = build_tattoo_mask(image)
    mask_arr = np.array(mask)

    if mask_arr.max() == 0:
        # No tattoo-like regions detected
        return image

    pipe = _load_inpaint_pipeline()
    result = pipe(
        prompt=prompt,
        image=image.convert("RGB"),
        mask_image=mask,
        num_inference_steps=num_inference_steps,
    ).images[0]
    return result


# ---------------------------------------------------------------------------
# BLIP captioning
# ---------------------------------------------------------------------------

_blip_processor = None
_blip_model     = None


def _load_blip():
    global _blip_processor, _blip_model
    if _blip_processor is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch

            model_id = "Salesforce/blip-image-captioning-base"
            _blip_processor = BlipProcessor.from_pretrained(model_id)
            _blip_model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _blip_model = _blip_model.to(device)
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for BLIP captioning. "
                "Install them with: pip install transformers torch"
            ) from exc
    return _blip_processor, _blip_model


def generate_caption(image: Image.Image, unique_token: str = "sks person") -> str:
    """
    Generate a natural-language caption with BLIP and prepend the unique token.

    The unique token is placed at the beginning so the model learns to
    associate the subject with that specific token string.
    """
    processor, model = _load_blip()
    import torch

    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(
        model.device, model.dtype if hasattr(model, "dtype") else torch.float32
    )
    out = model.generate(**inputs, max_new_tokens=50)
    base_caption = processor.decode(out[0], skip_special_tokens=True).strip()

    # Sanitise the caption: strip leading "a photo of" / "image of" boilerplate
    base_caption = re.sub(
        r"^(a\s+)?(photo|image|picture|portrait)\s+(of\s+)?",
        "",
        base_caption,
        flags=re.IGNORECASE,
    ).strip()

    return f"{unique_token}, {base_caption}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def process_dataset(
    input_dir: str,
    output_dir: str,
    unique_token: str = "sks person",
    target_size: int = 512,
    do_smooth_skin: bool = False,
    do_inpaint_tattoos: bool = False,
    caption_ext: str = ".txt",
) -> int:
    """
    Process all images in *input_dir* and write results to *output_dir*.

    Returns the number of successfully processed images.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"[WARN] No supported images found in {input_dir}")
        return 0

    processed = 0
    for img_file in sorted(image_files):
        try:
            image = Image.open(img_file).convert("RGB")
            print(f"[INFO] Processing {img_file.name} …", end=" ", flush=True)

            # 1. Face-centred crop + resize
            image = face_centered_crop(image, target_size=target_size)

            # 2. Skin smoothing (optional)
            if do_smooth_skin:
                image = smooth_skin(image)

            # 3. Tattoo inpainting (optional)
            if do_inpaint_tattoos:
                image = inpaint_tattoos(image)

            # 4. Save processed image
            out_img_path = output_path / (img_file.stem + ".png")
            image.save(out_img_path, format="PNG")

            # 5. Generate & save caption
            caption = generate_caption(image, unique_token=unique_token)
            out_cap_path = output_path / (img_file.stem + caption_ext)
            out_cap_path.write_text(caption, encoding="utf-8")

            print(f'done \u2192 {out_img_path.name}  |  caption: "{caption}"')
            processed += 1

        except Exception as exc:
            print(f"\n[ERROR] Failed to process {img_file.name}: {exc}", file=sys.stderr)

    print(f"\n[INFO] Processed {processed}/{len(image_files)} images → {output_path}")
    return processed


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision Privacy & Identity Lab – Image Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir",  required=True, help="Folder containing raw input images")
    parser.add_argument("--output_dir", required=True, help="Folder to write processed images + captions")
    parser.add_argument("--token",  default="sks person",
                        help="Unique activation token prepended to every caption")
    parser.add_argument("--size",   type=int, default=512,
                        help="Output image size (square)")
    parser.add_argument("--smooth_skin",    action="store_true",
                        help="Apply bilateral-filter skin smoothing")
    parser.add_argument("--inpaint_tattoos", action="store_true",
                        help="Detect and inpaint tattoo regions (CUDA recommended, requires ~10 GB VRAM)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        input_dir        = args.input_dir,
        output_dir       = args.output_dir,
        unique_token     = args.token,
        target_size      = args.size,
        do_smooth_skin   = args.smooth_skin,
        do_inpaint_tattoos = args.inpaint_tattoos,
    )
