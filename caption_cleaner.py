"""
Vision Privacy & Identity Lab – Module 2 helper
Caption generation and cleaning with BLIP + unique activation token.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Activation token ───────────────────────────────────────────────────────────
# A short, unique identifier that is prepended to every caption so that the
# DreamBooth / LoRA trainer can associate all images in this dataset with a
# single concept.  Generate once per training run and keep it stable.

_TOKEN_ENV_VAR = "VLAB_ACTIVATION_TOKEN"
_DEFAULT_TOKEN_PREFIX = "vplid"


def generar_token_activacion(longitud: int = 8) -> str:
    """
    Return a short, unique activation token for DreamBooth / LoRA training.

    The token is sourced (in order of precedence) from:
    1. The ``VLAB_ACTIVATION_TOKEN`` environment variable.
    2. A newly generated UUID4-based string (``<prefix><hex>``, e.g. ``vplid3f9a``).

    Parameters
    ----------
    longitud : int
        Number of hex characters appended to the prefix (2–32).

    Returns
    -------
    str
        A short, unique token such as ``vplid3f9a2b1c``.
    """
    env_token = os.environ.get(_TOKEN_ENV_VAR, "").strip()
    if env_token:
        return env_token

    longitud = max(2, min(longitud, 32))
    hex_part = uuid.uuid4().hex[:longitud]
    return f"{_DEFAULT_TOKEN_PREFIX}{hex_part}"


# ── BLIP captioning ────────────────────────────────────────────────────────────

def _load_blip_model(device: str = "cpu"):
    """
    Lazy-load the BLIP image-captioning model.
    Returns (processor, model) or raises ImportError if transformers is absent.
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for BLIP captioning. "
            "Install them with:  pip install transformers torch"
        ) from exc

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    model.eval()
    return processor, model


def generar_caption_blip(
    image_path: str | os.PathLike,
    device: str = "cpu",
    max_new_tokens: int = 50,
    _processor=None,
    _model=None,
) -> str:
    """
    Generate a descriptive caption for *image_path* using BLIP.

    Parameters
    ----------
    image_path : path-like
        Path to the image file.
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"`` …).
    max_new_tokens : int
        Maximum length of the generated caption.
    _processor, _model :
        Optional pre-loaded BLIP processor / model (avoids re-loading).

    Returns
    -------
    str
        Generated caption text (no activation token prepended here).
    """
    from PIL import Image
    import torch

    if _processor is None or _model is None:
        _processor, _model = _load_blip_model(device)

    image = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=max_new_tokens)

    caption = _processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


# ── Caption cleaning ───────────────────────────────────────────────────────────

_NOISE_PATTERNS = [
    r"\b(a photo of|an image of|a picture of|there is|there are)\b",
    r"\s{2,}",                   # multiple spaces
    r"^\s+|\s+$",                # leading / trailing whitespace
]


def limpiar_caption(texto: str) -> str:
    """
    Remove common BLIP boilerplate phrases and normalise whitespace.

    Parameters
    ----------
    texto : str
        Raw caption from BLIP or any other source.

    Returns
    -------
    str
        Cleaned caption.
    """
    for pattern in _NOISE_PATTERNS:
        texto = re.sub(pattern, " ", texto, flags=re.IGNORECASE)
    return texto.strip()


def construir_caption_entrenamiento(
    raw_caption: str,
    token: str,
    clase: str = "person",
) -> str:
    """
    Build the final training caption by prepending the activation token.

    Example
    -------
    >>> construir_caption_entrenamiento("smiling outdoors", "vplid3f9a", "person")
    'vplid3f9a person, smiling outdoors'

    Parameters
    ----------
    raw_caption : str
        Caption text (will be cleaned internally).
    token : str
        Unique activation token returned by ``generar_token_activacion``.
    clase : str
        Class word (e.g. ``"person"``, ``"man"``, ``"woman"``).

    Returns
    -------
    str
        Training-ready caption string.
    """
    cleaned = limpiar_caption(raw_caption)
    parts = [p for p in [token, clase, cleaned] if p]
    return ", ".join(parts)


# ── Batch caption generation ───────────────────────────────────────────────────

def procesar_directorio(
    image_dir: str | os.PathLike,
    output_dir: Optional[str | os.PathLike] = None,
    token: Optional[str] = None,
    clase: str = "person",
    device: str = "cpu",
    max_new_tokens: int = 50,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
) -> dict[str, str]:
    """
    Generate and save training captions for all images in *image_dir*.

    Each caption is saved as a ``.txt`` file with the same stem as the image
    (DreamBooth / LoRA convention).

    Parameters
    ----------
    image_dir : path-like
        Folder of processed images.
    output_dir : path-like, optional
        Where to write ``.txt`` files. Defaults to *image_dir*.
    token : str, optional
        Activation token; generated automatically if not supplied.
    clase : str
        Class word for the subject.
    device : str
        Torch device.
    max_new_tokens : int
        BLIP generation length.
    extensions : tuple of str
        Image extensions to process.

    Returns
    -------
    dict[str, str]
        Mapping of image filename → final training caption.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) if output_dir else image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if token is None:
        token = generar_token_activacion()
        logger.info("Activation token: %s", token)

    processor, model = _load_blip_model(device)
    captions: dict[str, str] = {}

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in extensions:
            continue

        raw = generar_caption_blip(
            img_path, device=device, max_new_tokens=max_new_tokens,
            _processor=processor, _model=model,
        )
        final = construir_caption_entrenamiento(raw, token, clase)
        captions[img_path.name] = final

        txt_path = output_dir / (img_path.stem + ".txt")
        txt_path.write_text(final, encoding="utf-8")
        logger.info("%s → %s", img_path.name, final)

    return captions


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vision Privacy & Identity Lab – caption generator"
    )
    parser.add_argument("image_dir", help="Directory of processed images")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--token", default=None, help="Activation token (auto-generated if omitted)")
    parser.add_argument("--clase", default="person")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    procesar_directorio(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        token=args.token,
        clase=args.clase,
        device=args.device,
    )
