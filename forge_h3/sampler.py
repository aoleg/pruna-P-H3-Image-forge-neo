"""The "H3-Image LCM" sampler: LCM with a noise scale that falls across the steps. Torch only, so CPU tests cover it.

Each step takes the model's clean estimate and noises it afresh to the next sigma, as Forge's LCM does; the fresh noise
is scaled from the first noise step's value to the last one's, as ComfyUI's SamplerLCM does with s_noise and
s_noise_end. On MiniMax-H3-Image this is what removes the speckle and grit that Euler and Heun leave at the checkpoint's
own last sigmas below about 16 MP (at 16 MP Euler is clean too): Euler carries the model's own noise estimate into the
next step, and the low-sigma output heads turn it into speckle, while fresh noise is what they were distilled on.

The scale decides how much fine detail the picture gets. Above 1 the model is handed more noise than the step's sigma
says, removes only the sigma's worth and keeps the rest as texture; below 1 it gets less and smooths more. 1.10 at the
first noise step to 0.90 at the last is the community workflow's setting (Simple, 6 steps); Settings -> MiniMax H3 can
change both ends.
"""

import torch

SAMPLER = "H3-Image LCM"
S_NOISE_START = 1.10
S_NOISE_END = 0.90
S_NOISE_RANGE = (0.5, 2.0)
# the two Settings entries (scripts/forge_h3.py) and the infotext names they are written and pasted under
S_NOISE_START_SETTING = "h3_lcm_noise_first"
S_NOISE_END_SETTING = "h3_lcm_noise_last"
S_NOISE_START_INFOTEXT = "H3 LCM noise first"
S_NOISE_END_INFOTEXT = "H3 LCM noise last"


def noise_scale(step: int, noise_steps: int, start: float = S_NOISE_START, end: float = S_NOISE_END) -> float:
    """The noise multiplier of noise step `step` (0-based) of `noise_steps`: linear from start (the first step that adds
    noise) to end (the last step that adds noise). The final step goes to sigma 0 and adds none."""
    if noise_steps <= 1:
        return start
    return start + (end - start) * step / (noise_steps - 1)


def _clamped(value, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    low, high = S_NOISE_RANGE
    return min(max(value, low), high)


def noise_settings() -> tuple[float, float]:
    """The first and last noise scales from Settings -> MiniMax H3, the defaults outside Forge."""
    try:
        from modules import shared
        opts = shared.opts
    except (ImportError, AttributeError):
        return S_NOISE_START, S_NOISE_END
    return (_clamped(getattr(opts, S_NOISE_START_SETTING, S_NOISE_START), S_NOISE_START),
            _clamped(getattr(opts, S_NOISE_END_SETTING, S_NOISE_END), S_NOISE_END))


@torch.no_grad()
def sample_h3_image_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None,
                        s_noise=None, s_noise_end=None):
    """k-diffusion sampler: Forge passes the model wrapper, the start latent and the sigma list. Without a
    noise_sampler it draws from k-diffusion's default one, which Forge points at the image's seeded RNG, so a seed gives
    the same picture every time. s_noise / s_noise_end override the Settings values (tests and callers)."""
    extra_args = {} if extra_args is None else extra_args
    if noise_sampler is None:
        from k_diffusion.sampling import default_noise_sampler

        noise_sampler = default_noise_sampler(x)
    start, end = noise_settings()
    start = start if s_noise is None else float(s_noise)
    end = end if s_noise_end is None else float(s_noise_end)
    predictor = model.inner_model.predictor
    s_in = x.new_ones([x.shape[0]])
    steps = len(sigmas) - 1
    noise_steps = int((sigmas[1:] > 0).sum())
    added = 0
    for i in range(steps):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        x = denoised
        if sigmas[i + 1] > 0:
            noise = noise_sampler(sigmas[i], sigmas[i + 1]) * noise_scale(added, noise_steps, start, end)
            x = predictor.noise_scaling(sigmas[i + 1], noise, x)
            added += 1
    return x
