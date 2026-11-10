"""The H3 autoencoder as a Forge component, and the state-dict conversions it accepts.

Accepted files: the Comfy-Org single file (fp16), Kijai's int8 ConvRot video VAE (its decoder transformer linears are
quantized; Forge's mixed-precision operations load them), the original MiniMax FL2VA file, which lacks the latent
statistics buffers, and the two image VAEs, the same encoder with a single-frame decoder: P-H3-Image's (bf16 or int8)
and MiniMax-H3-Image's 2D one (it has no latent statistics either). The audio VAE of the video model is recognized
only to be refused: a still image never decodes it.
"""

import torch

from .video_vae import H3_IMAGE_DECODER_DIM, IMAGE_PROJ_OUT, LATENTS_MEAN, LATENTS_STD

VIDEO_KEYS = ("decoder.transformer_blocks.0.scale1", "encoder.down.5.block.0.conv1.weight")
AUDIO_KEY = "pre_block.attn.zero_k_bias"
PROJ_OUT = "decoder.proj_out.weight"
X_EMBEDDER = "decoder.x_embedder.weight"
QUANT_SUFFIXES = (".comfy_quant", ".weight_scale")

MISSING_VAE = "MiniMax H3 needs its autoencoder: select the H3 VAE under VAE / Text Encoder"


def _shape(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    return tuple(value["shape"]) if isinstance(value, dict) else tuple(value.shape)


def is_video_vae(sd) -> bool:
    return all(k in sd for k in VIDEO_KEYS)


def is_audio_vae(sd) -> bool:
    return AUDIO_KEY in sd


def is_image_vae(sd) -> bool:
    """An image VAE: its decoder makes one frame per latent (ComfyUI's test on the output projection)."""
    return PROJ_OUT in sd and _shape(sd[PROJ_OUT])[0] == IMAGE_PROJ_OUT


def is_h3_image_vae(sd) -> bool:
    """MiniMax-H3-Image's VAE: the 16-head 2D decoder (P-H3-Image's has 32 heads)."""
    return is_image_vae(sd) and X_EMBEDDER in sd and _shape(sd[X_EMBEDDER])[0] == H3_IMAGE_DECODER_DIM


def is_ph3_vae(sd) -> bool:
    return is_image_vae(sd) and not is_h3_image_vae(sd)


def is_quantized(sd: dict, prefix: str = "") -> bool:
    return any(k.startswith(prefix) and k.endswith(QUANT_SUFFIXES) for k in sd)


def convert_video_vae(sd: dict) -> dict:
    sd = dict(sd)
    sd.setdefault("latents_mean", torch.tensor(LATENTS_MEAN))
    sd.setdefault("latents_std", torch.tensor(LATENTS_STD))
    return sd


def video_layers(sd: dict, prefix: str = "") -> int:
    return sum(k.startswith(f"{prefix}decoder.transformer_blocks.") and k.endswith(".scale1") for k in sd)
