"""
Vision Privacy & Identity Lab – Module 1
Preprocessing: image cleaning, resizing, face/tattoo privacy masks.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter

# MediaPipe is imported lazily so the module can be imported without GPU/camera
try:
    import mediapipe as mp

    _mp_face_detection = mp.solutions.face_detection
    _MEDIAPIPE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MEDIAPIPE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_OUTPUT_SIZE = (512, 512)
FACE_BLUR_STRENGTH = 51          # must be odd
TATTOO_MASK_COLOR = (0, 0, 0)    # black fill for detected tattoo regions


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR array."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(array: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def _resize_and_crop(image: Image.Image, target: tuple[int, int] = DEFAULT_OUTPUT_SIZE) -> Image.Image:
    """
    Resize + centre-crop an image to *target* dimensions while preserving
    aspect ratio (cover strategy, same as CSS background-size: cover).
    """
    tw, th = target
    iw, ih = image.size
    scale = max(tw / iw, th / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    return image.crop((left, top, left + tw, top + th))


# ── Face detection & privacy mask ─────────────────────────────────────────────

def aplicar_mascara_rostros(image: Image.Image, blur_strength: int = FACE_BLUR_STRENGTH) -> Image.Image:
    """
    Detect faces with MediaPipe and apply a Gaussian blur privacy mask over
    each bounding box.  Falls back gracefully when MediaPipe is unavailable.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (any mode; converted internally to RGB).
    blur_strength : int
        Kernel radius for the Gaussian blur (must be odd, >= 1).

    Returns
    -------
    PIL.Image.Image
        Image with face regions blurred.
    """
    if not _MEDIAPIPE_AVAILABLE:
        logger.warning("MediaPipe not available – skipping face masking.")
        return image

    blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]

    with _mp_face_detection.FaceDetection(
        model_selection=1,          # full-range model
        min_detection_confidence=0.5,
    ) as detector:
        results = detector.process(rgb)

    if not results.detections:
        return image

    output = rgb.copy()
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        roi = output[y1:y2, x1:x2]
        roi_blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
        output[y1:y2, x1:x2] = roi_blurred

    return Image.fromarray(output)


# ── Tattoo detection placeholder ───────────────────────────────────────────────

def detectar_tatuajes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Placeholder tattoo detector.  Returns a list of (x1, y1, x2, y2) bounding
    boxes for suspected tattoo regions.

    The current implementation is a stub that returns an empty list; replace
    the body of this function with a trained model (e.g. a YOLO/ONNX model
    fine-tuned on tattoo imagery) once available.
    """
    # TODO: integrate a fine-tuned tattoo detection model
    return []


def aplicar_inpainting_tatuajes(
    image: Image.Image,
    regions: list[tuple[int, int, int, int]],
    fill_color: tuple[int, int, int] = TATTOO_MASK_COLOR,
) -> Image.Image:
    """
    Apply an inpainting mask over the supplied bounding boxes.

    Currently fills regions with *fill_color* (black).  For production use,
    swap the fill for a diffusion-based inpainting call.

    Parameters
    ----------
    image : PIL.Image.Image
    regions : list of (x1, y1, x2, y2) tuples
    fill_color : RGB tuple used to fill each region

    Returns
    -------
    PIL.Image.Image
    """
    if not regions:
        return image

    output = image.copy().convert("RGB")
    pixels = output.load()

    for (x1, y1, x2, y2) in regions:
        for x in range(max(0, x1), min(output.width, x2)):
            for y in range(max(0, y1), min(output.height, y2)):
                pixels[x, y] = fill_color  # type: ignore[index]

    return output


# ── Main pipeline ──────────────────────────────────────────────────────────────

def adaptar_imagenes_crudas(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    target_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    apply_face_mask: bool = True,
    apply_tattoo_mask: bool = True,
    blur_strength: int = FACE_BLUR_STRENGTH,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Scan *input_dir* for raw images, apply privacy processing, and write
    cleaned copies to *output_dir*.

    Processing pipeline per image
    ──────────────────────────────
    1. Load image and convert to RGB.
    2. Resize + centre-crop to *target_size*.
    3. (Optional) Detect faces and blur them.
    4. (Optional) Detect tattoo regions and fill them.
    5. Export as PNG to *output_dir*.

    Parameters
    ----------
    input_dir : path-like
        Folder containing raw source images (searched recursively).
    output_dir : path-like
        Destination folder for processed images.
    target_size : (width, height)
        Output resolution in pixels.
    apply_face_mask : bool
        Whether to apply the face-blurring privacy mask.
    apply_tattoo_mask : bool
        Whether to detect and inpaint tattoo regions.
    blur_strength : int
        Gaussian blur kernel size for face masking (must be odd).
    skip_existing : bool
        Skip files that already exist in *output_dir*.

    Returns
    -------
    list of Path
        Paths of all successfully exported images.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        logger.warning("No images found in '%s'.", input_dir)
        return []

    exported: list[Path] = []

    for src_path in image_paths:
        out_name = src_path.stem + ".png"
        out_path = output_dir / out_name

        if skip_existing and out_path.exists():
            logger.info("Skipping existing file: %s", out_name)
            exported.append(out_path)
            continue

        try:
            image = Image.open(src_path).convert("RGB")
        except Exception as exc:
            logger.error("Cannot open '%s': %s", src_path, exc)
            continue

        # Step 2 – resize / crop
        image = _resize_and_crop(image, target_size)

        # Step 3 – face privacy mask
        if apply_face_mask:
            image = aplicar_mascara_rostros(image, blur_strength)

        # Step 4 – tattoo inpainting
        if apply_tattoo_mask:
            regions = detectar_tatuajes(image)
            image = aplicar_inpainting_tatuajes(image, regions)

        # Step 5 – export
        image.save(out_path, format="PNG")
        logger.info("Exported: %s", out_path)
        exported.append(out_path)

    logger.info("Done. %d / %d images processed.", len(exported), len(image_paths))
    return exported


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vision Privacy & Identity Lab – image preprocessor"
    )
    parser.add_argument("input_dir", help="Folder with raw images")
    parser.add_argument("output_dir", help="Destination for processed images")
    parser.add_argument("--size", nargs=2, type=int, default=[512, 512], metavar=("W", "H"))
    parser.add_argument("--no-face-mask", action="store_true")
    parser.add_argument("--no-tattoo-mask", action="store_true")
    parser.add_argument("--blur", type=int, default=FACE_BLUR_STRENGTH)
    args = parser.parse_args()

    adaptar_imagenes_crudas(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=tuple(args.size),
        apply_face_mask=not args.no_face_mask,
        apply_tattoo_mask=not args.no_tattoo_mask,
        blur_strength=args.blur,
    )
