"""The blank reference: a black picture given to the model as an edit's <Picture 1>, for part of the denoising or all
of it. Plain logic, so CPU tests cover it.

A community tip for ComfyUI: connect an empty image to MiniMax-H3-Image's image conditioning. The picture then
reaches the model twice, as a vision block in the prompt the text encoder reads and as reference rows (its latent,
never denoised) in the DiT's sequence. On the RTX 5090 at 16 MP it gave denser pictures (more saturation, deeper
shadows, less pixel-level grit) and a different composition.

The from and to sliders are percentages of the denoising, as ControlNet's start and end are: each is turned into a
noise level with the model's own percent_to_sigma, and a step uses the blank reference when its sigma lies between
the two. On MiniMax-H3-Image's grid the sixths are its six steps (33% starts at the third, 50% at the fourth),
whatever the step count or the schedule. Half a percent of slack lets a slider at 33 or 67 mean the step at 1/3 or
2/3.

For part of the denoising both prompts are needed: the text encoder runs once with the picture and once without, and
Forge carries the two as one tensor, with-picture part first (blank_split tells them apart).
"""

from dataclasses import dataclass

# ComfyUI's EmptyImage is black; 1024 on a side is the smallest copy the model's edit sizing makes of a picture
BLANK_SIDE = 1024
BLANK_COLOR = (0, 0, 0)
SLACK = 0.5
INFOTEXT = "H3 Blank reference"


@dataclass(frozen=True)
class Window:
    """The part of the denoising that sees the blank reference, in percent."""
    start: float
    end: float

    @property
    def full(self) -> bool:
        return self.start <= 0.0 and self.end >= 100.0

    def label(self) -> str:
        return f"{self.start:g}-{self.end:g}%"


def window(enabled, start=0.0, end=100.0) -> Window | None:
    """The panel's checkbox and sliders as a Window, or None when the box is off, a value is unreadable or the
    range is empty."""
    if not enabled:
        return None
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return None
    start, end = min(max(start, 0.0), 100.0), min(max(end, 0.0), 100.0)
    if end <= start:
        return None
    return Window(start, end)


def parse(text) -> Window | None:
    """A Window from its infotext label ("33-100%")."""
    try:
        start, end = str(text).rstrip("%").split("-")
        return window(True, start, end)
    except ValueError:
        return None


def thresholds(win: Window, percent_to_sigma) -> tuple[float, float]:
    """(sigma_from, sigma_to): a step with sigma_to < sigma <= sigma_from uses the blank reference."""
    sigma_from = 1.0 if win.start - SLACK <= 0.0 else float(percent_to_sigma((win.start - SLACK) / 100.0))
    sigma_to = 0.0 if win.end >= 100.0 else float(percent_to_sigma((win.end - SLACK) / 100.0))
    return sigma_from, sigma_to


def active(sigma: float, sigma_from: float, sigma_to: float) -> bool:
    return sigma_to < sigma <= sigma_from + 1e-6


def blank_split(context_len: int, prefix_len: int) -> tuple[int, int]:
    """Lengths of the with-picture and the plain prompt in a conditioning that carries both. The with-picture one is
    the picture's prefix (its label and vision block) and the prompt's m tokens; the plain one is the same m tokens,
    or one pad token for an empty prompt."""
    rest = context_len - prefix_len
    if rest == 1:
        return prefix_len, 1
    if rest < 2 or rest % 2:
        raise ValueError(f"[MiniMax H3] a blank-reference conditioning of {context_len} tokens does not split after "
                         f"its {prefix_len}-token picture prefix")
    return prefix_len + rest // 2, rest // 2


def blank_picture():
    from PIL import Image
    return Image.new("RGB", (BLANK_SIDE, BLANK_SIDE), BLANK_COLOR)
