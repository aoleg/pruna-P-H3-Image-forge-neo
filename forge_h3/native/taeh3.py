"""madebyollin's taeh3, a tiny decoder for H3 latents, for Forge Neo's TAESD live preview.

Forge asks sd_vae_taesd.decoder_model() for a decoder when Settings > Live Previews > Live Preview Method is TAESD;
for H3 the extension answers with a PreviewDecoder (patches._hook_taesd). Any other method never reaches it.

taeh3 (taehv 62f7591, MIT) is a temporal TAEHV decoder: 24 latent channels, three 2x space upscales and a 2x2 pixel
shuffle (16x, as H3's VAEs), and a 2x time upscale in its last two stages, so one latent token gives four frames.
ComfyUI previews a MiniMax-H3-Image latent with it by decoding the one token and keeping the last of the four frames
(comfy/taesd/taehv.py, latent_preview.TAEHVPreviewerImpl). decode_last_frame() gives that same frame, but only the
frames its MemBlocks need pass through each stage: one at full size instead of four.

The decoder takes the latent as the sampler holds it (TAEHV uses no scale or shift; H3's normalization lives inside
its VAEs) and returns [0, 1] pixels, as Forge's TAESD path expects.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# madebyollin/taehv at 62f7591f (taeh3 added 5 August 2026, unchanged since)
URL = "https://github.com/madebyollin/taehv/raw/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors"
FILE_NAME = "taeh3.safetensors"
FOLDER = "VAE-taesd"  # Forge Neo's own folder for its tiny decoders, under models/

LATENT_CHANNELS = 24
PATCH_SIZE = 2
CHANNELS = (256, 128, 64, 64)
TIME_STRIDES = (1, 2, 2)

# the longest side of a preview: Forge copies every preview to the CPU, converts and encodes it, so a 16 MP picture
# is previewed at 2048x2048, area-averaged from the full decode
PREVIEW_SIDE = 2048


def conv(n_in, n_out, bias=True):
    return nn.Conv2d(n_in, n_out, 3, padding=1, bias=bias)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """A residual block that also sees the previous frame's input (zeros for the first frame)."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TGrow(nn.Module):
    """Time upscale: each frame becomes `stride` frames, the k-th from the k-th block of output channels."""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _nt, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


def decoder() -> nn.Sequential:
    """taehv.py's TAEHV decoder (base variant, ReLU) in taeh3's configuration; its indices are the file's keys."""
    n_f = CHANNELS
    blocks = [Clamp(), conv(LATENT_CHANNELS, n_f[0]), nn.ReLU()]
    for stage, stride in enumerate(TIME_STRIDES):
        width = n_f[stage]
        blocks += [MemBlock(width, width), MemBlock(width, width), MemBlock(width, width),
                   nn.Upsample(scale_factor=2), TGrow(width, stride), conv(width, n_f[stage + 1], bias=False)]
    blocks += [nn.ReLU(), conv(n_f[3], 3 * PATCH_SIZE ** 2)]
    return nn.Sequential(*blocks)


def decode_last_frame(model: nn.Sequential, latent: torch.Tensor) -> torch.Tensor:
    """One latent token [N, 24, h, w] -> the last of the frames it decodes to, [N, 3, 16h, 16w] in [0, 1].

    Frames run as the batch, N-major. A MemBlock's memory is the previous frame of the same picture. After the last
    MemBlock no frame sees another, so only the last frame goes on, and a later TGrow makes only its last frame.
    """
    blocks = list(model)
    last_memblock = max(i for i, b in enumerate(blocks) if isinstance(b, MemBlock))
    n, frames = latent.shape[0], 1
    x = latent
    for i, block in enumerate(blocks):
        if isinstance(block, MemBlock):
            if frames > 1:
                clip = x.view(n, frames, *x.shape[1:])
                past = F.pad(clip, (0, 0, 0, 0, 0, 0, 1, 0))[:, :frames].reshape(x.shape)
            else:
                past = torch.zeros_like(x)
            x = block(x, past)
            if i == last_memblock and frames > 1:
                x = x.view(n, frames, *x.shape[1:])[:, -1]
                frames = 1
        elif isinstance(block, TGrow) and i > last_memblock:
            # the last frame of this time upscale comes from the last block of output channels
            x = F.conv2d(x, block.conv.weight[-x.shape[1]:])
        elif isinstance(block, TGrow):
            x = block(x)
            frames *= block.stride
        else:
            x = block(x)
    return F.pixel_shuffle(x, PATCH_SIZE).clamp_(0, 1)


def preview_size(height: int, width: int, side: int = PREVIEW_SIDE) -> tuple[int, int]:
    """The preview's (height, width): the picture's own size, or shrunk to `side` on its long edge."""
    scale = min(1.0, side / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


class PreviewDecoder(nn.Module):
    """A packed H3 latent [1, 1, 1, N] -> the picture's preview [1, 3, H, W] in [0, 1], for Forge's TAESD path.

    shapes_of() gives the current generation's StreamShapes (or None); a latent that is not the current generation's
    goes to fallback(sample), which also returns [0, 1] pixels.
    """

    def __init__(self, shapes_of, fallback, side: int = PREVIEW_SIDE):
        super().__init__()
        self.decoder = decoder()
        self.shapes_of = shapes_of
        self.fallback = fallback
        self.side = side

    @torch.inference_mode()
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        from .streams import preview_frame

        shapes = self.shapes_of()
        frame = preview_frame(sample, shapes)
        if frame is sample or frame.ndim != 4 or frame.shape[1] != LATENT_CHANNELS:
            return self.fallback(sample)
        weight = self.decoder[1].weight
        pixels = decode_last_frame(self.decoder, frame.to(weight.device, weight.dtype)).float()
        size = preview_size(*pixels.shape[-2:], self.side)
        if size != tuple(pixels.shape[-2:]):
            pixels = F.interpolate(pixels, size=size, mode="area")
        return pixels


class NotTaeh3(ValueError):
    pass


def load(path: str, shapes_of, fallback, device=None, dtype=None) -> PreviewDecoder:
    from safetensors.torch import load_file

    state = load_file(path)
    if "decoder.1.weight" not in state:
        # Kijai's earlier 2D TAE (Kijai/MiniMax-H3-TAE) has the same file name and no "decoder." prefix
        raise NotTaeh3(f"{path} is not madebyollin's taeh3 (another TAE under the same name?); delete it and the "
                       f"preview downloads taeh3 again")
    model = PreviewDecoder(shapes_of, fallback)
    model.decoder.load_state_dict({k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")})
    return model.eval().requires_grad_(False).to(device=device or "cpu", dtype=dtype or torch.float32)


def path_in(models_path: str) -> str:
    return os.path.join(models_path, FOLDER, FILE_NAME)


class PreviewSource:
    """Loads taeh3 once per session, downloading it first if the file is missing.

    A failure is reported once and remembered, so the preview falls back to RGB without trying again at every step.
    """

    def __init__(self, models_path: str, download, open_decoder, report):
        self.path = path_in(models_path)
        self.download = download          # (path, url): Forge's sd_vae_taesd.download_model
        self.open_decoder = open_decoder  # (path) -> PreviewDecoder
        self.report = report              # (message)
        self.decoder = None
        self.failed = False

    def get(self):
        if self.decoder is None and not self.failed:
            try:
                if not os.path.exists(self.path):
                    self.download(self.path, URL)
                self.decoder = self.open_decoder(self.path)
            except Exception as e:
                self.failed = True
                self.report(f"[MiniMax H3] taeh3 preview unavailable, using the RGB preview: {e}")
        return self.decoder
