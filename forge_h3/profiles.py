"""The two kinds of still-image H3 model and their sampling recipes. Torch only, so CPU tests cover it."""

from dataclasses import dataclass

# the ManualSigmas grid of Pruna's workflows: 8 Euler steps at shift 1, no CFG
PH3_IMAGE_SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
PH3_IMAGE_SCHEDULER = "P-H3-Image"
# the hidden states of the 4B text encoder that Pruna's text connector reads (36 = after the final norm), as Forge's
# clip wrapper asks the encoder for them
PH3_IMAGE_HIDDEN_STATES = [13, 19, 25, 31, 36]


@dataclass(frozen=True)
class Profile:
    name: str          # the UI preset, and the key of the Settings options
    label: str
    sampler: str
    scheduler: str
    steps: int
    cfg: float
    shift: float       # the flow shift of both streams
    size: int          # the recommended side of a square picture
    text_encoder: str  # the text encoder variant that selects the profile (models.ModelInfo.variant)
    vae: str           # the VAE variant it needs


# an H3 checkpoint with its own text encoder and VAE, as a still-image model: the recipe of lodestones' chroma-h3
# fine-tune (its LoRA on any H3 checkpoint, about 30 Euler steps at CFG 3 and shift 3, around 1 MP); its few-step
# DMD LoRA wants 4 or 8 steps at CFG 1 instead
H3 = Profile(name="h3", label="MiniMax H3", sampler="Euler", scheduler="Simple", steps=30, cfg=3.0, shift=3.0,
             size=1024, text_encoder="h3", vae="h3")
# Pruna's P-H3-Image: its 4B text encoder and single-frame VAE, 8 MP output
PH3_IMAGE = Profile(name="ph3", label="P-H3-Image", sampler="Euler", scheduler=PH3_IMAGE_SCHEDULER, steps=8, cfg=1.0, shift=1.0,
                    size=2912, text_encoder="ph3-image", vae="ph3-image")

PROFILES = {profile.text_encoder: profile for profile in (H3, PH3_IMAGE)}


def for_text_encoder(variant: str) -> Profile:
    return PROFILES[variant]


def ph3_image_sigmas(n: int, sigma_min=0.0, sigma_max=1.0, device=None, **kwargs):
    """Forge scheduler function: Pruna's 8-step grid as it is, and for another step count the same curve resampled
    (linear interpolation over the step index, so a 4-step run keeps the grid's shape). Shift does not apply: the grid
    is given in the model's own sigma space, as ComfyUI's ManualSigmas is."""
    import torch
    grid = torch.tensor(PH3_IMAGE_SIGMAS, dtype=torch.float64)
    if n == len(grid) - 1:
        sigmas = grid
    else:
        positions = torch.linspace(0.0, len(grid) - 1, n + 1, dtype=torch.float64)
        lower = positions.floor().long().clamp(max=len(grid) - 2)
        sigmas = torch.lerp(grid[lower], grid[lower + 1], positions - lower)
    return sigmas.to(dtype=torch.float32, device=device)
