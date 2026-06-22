"""Image validation checks run before model inference."""
from __future__ import annotations

import io

from fastapi import HTTPException
from PIL import Image

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_BYTES = 10 * 1024 * 1024   # 10 MB
MIN_DIM = 100
MAX_DIM = 5000
MAX_ASPECT_RATIO = 3.0          # width / height
MAX_SATURATION = 0.08           # 0–1 scale; above this = too colourful
MIN_GRAYSCALE_RATIO = 0.90      # fraction of pixels that must be near-grayscale


def _reject(message: str) -> None:
    raise HTTPException(status_code=400, detail={"error": True, "message": message})


def validate_xray_image(image_bytes: bytes, filename: str, content_type: str) -> None:
    """Run all validation checks in order. Raises HTTP 400 on the first failure."""

    # 1. File type
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if content_type not in ALLOWED_CONTENT_TYPES or ext not in ALLOWED_EXTENSIONS:
        _reject(
            f"Unsupported file type '{content_type}'. Only PNG and JPG/JPEG images are accepted."
        )

    # 2. File size
    if len(image_bytes) > MAX_BYTES:
        mb = len(image_bytes) / (1024 * 1024)
        _reject(f"File size {mb:.1f} MB exceeds the 10 MB limit.")

    # Load once for the remaining checks
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()                        # catch truncated / corrupt files
        img = Image.open(io.BytesIO(image_bytes))   # re-open after verify
    except Exception:
        _reject("Uploaded file could not be decoded as a valid image.")

    w, h = img.size

    # 3. Dimensions
    if w < MIN_DIM or h < MIN_DIM:
        _reject(
            f"Image is too small ({w}×{h} px). Minimum size is {MIN_DIM}×{MIN_DIM} px."
        )
    if w > MAX_DIM or h > MAX_DIM:
        _reject(
            f"Image is too large ({w}×{h} px). Maximum size is {MAX_DIM}×{MAX_DIM} px."
        )

    # 4. Aspect ratio
    if w / h > MAX_ASPECT_RATIO:
        _reject(
            f"Image aspect ratio ({w/h:.1f}:1) is too wide. "
            f"Chest X-rays should be no wider than {MAX_ASPECT_RATIO}:1."
        )

    # 5. Grayscale / saturation check (two complementary tests)
    rgb = img.convert("RGB")
    pixels = list(rgb.getdata())
    total = len(pixels)

    # 5a. Average HSV saturation
    hsv = img.convert("HSV")
    _, s, _ = hsv.split()
    avg_saturation = sum(s.getdata()) / total / 255.0
    if avg_saturation > MAX_SATURATION:
        _reject(
            "Image does not appear to be a chest X-ray "
            f"(average colour saturation {avg_saturation:.2f} exceeds threshold {MAX_SATURATION})."
        )

    # 5b. Pixel-level grayscale ratio — require most pixels to have R≈G≈B
    near_gray = sum(1 for r, g, b in pixels if max(r, g, b) - min(r, g, b) < 30)
    grayscale_ratio = near_gray / total
    if grayscale_ratio < MIN_GRAYSCALE_RATIO:
        _reject(
            "Image does not appear to be a chest X-ray "
            f"(only {grayscale_ratio:.0%} of pixels are near-grayscale; expected ≥{MIN_GRAYSCALE_RATIO:.0%})."
        )
