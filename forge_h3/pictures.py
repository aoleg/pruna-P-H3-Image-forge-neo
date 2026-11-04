"""The pictures of a P-H3-Image edit, from Forge Neo's own image inputs (ComfyUI PH3ImageConditioning).

In img2img the input image is <Picture 1>, the picture to edit; the gallery of Forge Neo's built-in ImageStitch
Integrated holds the references, <Picture 2> onwards. Each picture reaches the text encoder scaled within 1 MP and
the VAE scaled within the output area, aspect kept, sides on the 32 grid, never enlarged; the original picture is
taken, not the copy Forge resized to the output size.
"""

import base64
import io
import math
import re

import numpy as np
from PIL import Image

from .contracts import GRID, MAX_REFERENCES, H3Error

IMAGE_STITCH = "ImageStitch Integrated"
# what the text encoder sees of a picture
VISION_PIXELS = 1_000_000
# a short instruction gets the edit prefix; an edit brief from Pruna's prompt upsampler, or an instruction that
# already names the picture to edit, is encoded as it is
EDIT_PREFIX = "Edit <Picture 1>. "
EDIT_VERBATIM = re.compile(r"\s*(Edit <Picture \d+>|subject_definitions:)")

_warned_extra = False


def load(value):
    # a gallery item is (image, caption) in the UI; through the API it arrives as base64, with or without a data: prefix
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    if value is None or isinstance(value, Image.Image):
        return value
    if isinstance(value, str) and value:
        return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[-1])))
    return None


def stitch_gallery(p) -> list:
    """The images in ImageStitch's gallery, when its accordion is on."""
    runner = getattr(p, "scripts", None)
    for script in getattr(runner, "alwayson_scripts", []):
        if script.title() == IMAGE_STITCH:
            enable, gallery = (list(p.script_args[script.args_from:script.args_to]) + [False, None])[:2]
            if not enable or not gallery:
                return []
            images = [load(item) for item in gallery]
            return [image for image in images if image is not None]
    return []


def collect(p) -> list:
    """<Picture 1> is the img2img input image, the gallery follows; at most 8 pictures in all."""
    images = list(getattr(p, "init_images", None) or [])[:1] + stitch_gallery(p)
    if len(images) > MAX_REFERENCES:
        raise H3Error(f"P-H3-Image edits with up to {MAX_REFERENCES} pictures, the input image included; "
                      f"{len(images)} were given.")
    return images


def within_area(width: int, height: int, pixels: int) -> tuple[int, int]:
    """Aspect-preserving scale (down only) to `pixels`, rounded to the canvas grid."""
    scale = min(1.0, math.sqrt(pixels / (width * height)))
    return tuple(max(GRID, round(side * scale / GRID) * GRID) for side in (width, height))


def prepare(image: Image.Image, pixels: int):
    """(1, h, w, 3) float in [0, 1] within the pixel budget, the form the text encoder and the VAE take."""
    import torch
    image = image.convert("RGB")
    size = within_area(image.width, image.height, pixels)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)


def edit_prompt(prompt: str) -> str:
    """The instruction as the model reads it when pictures are present."""
    if EDIT_VERBATIM.match(prompt):
        return prompt.strip()
    return EDIT_PREFIX + prompt
