"""The H3-Image LCM sampler: LCM with a falling noise scale."""

import unittest

try:
    import torch
except ImportError:
    torch = None

import sys
import types
from unittest.mock import patch

from forge_h3.sampler import (
    S_NOISE_END,
    S_NOISE_START,
    SAMPLER,
    noise_scale,
    noise_settings,
)


class NoiseScaleTests(unittest.TestCase):
    def test_the_scale_falls_linearly_from_the_first_noise_step_to_the_last(self):
        self.assertEqual(SAMPLER, "H3-Image LCM")
        self.assertEqual((S_NOISE_START, S_NOISE_END), (1.10, 0.90))
        # Simple at 6 steps adds noise after the first five; the step to 0 adds none
        self.assertEqual([round(noise_scale(i, 5), 4) for i in range(5)], [1.10, 1.05, 1.00, 0.95, 0.90])
        self.assertEqual(noise_scale(0, 1), S_NOISE_START)
        self.assertEqual([noise_scale(i, 3, 1.5, 1.5) for i in range(3)], [1.5, 1.5, 1.5])

    def test_the_settings_are_read_and_clamped_to_the_slider_range(self):
        self.assertEqual(noise_settings(), (S_NOISE_START, S_NOISE_END))
        opts = types.SimpleNamespace(h3_lcm_noise_first=1.6, h3_lcm_noise_last=0.7)
        shared = types.SimpleNamespace(opts=opts)
        with patch.dict(sys.modules, {"modules": types.SimpleNamespace(shared=shared), "modules.shared": shared}):
            self.assertEqual(noise_settings(), (1.6, 0.7))
            opts.h3_lcm_noise_first, opts.h3_lcm_noise_last = 9.0, "x"
            self.assertEqual(noise_settings(), (2.0, S_NOISE_END))


class FlowPredictor:
    """Forge's flow noise scaling: sigma * noise + (1 - sigma) * latent."""

    @staticmethod
    def noise_scaling(sigma, noise, latent):
        return sigma * noise + (1.0 - sigma) * latent


class FakeModel:
    """A denoiser that always predicts the same clean latent and records what it was given."""

    def __init__(self, clean):
        self.clean = clean
        self.calls = []
        self.inner_model = type("Inner", (), {"predictor": FlowPredictor()})()

    def __call__(self, x, sigma, **kwargs):
        self.calls.append((x.clone(), float(sigma[0])))
        return self.clean.clone()


@unittest.skipIf(torch is None, "needs torch")
class SamplerTests(unittest.TestCase):
    def test_each_step_noises_the_clean_estimate_afresh_with_the_scaled_noise(self):
        from forge_h3.sampler import sample_h3_image_lcm

        clean = torch.full((1, 2, 3, 3), 0.5)
        noise = torch.ones_like(clean)
        sigmas = torch.tensor([1.0, 0.75, 0.25, 0.0])
        model = FakeModel(clean)
        drawn = []
        out = sample_h3_image_lcm(model, torch.randn_like(clean), sigmas,
                                  noise_sampler=lambda s, s_next: drawn.append((float(s), float(s_next))) or noise)
        # one model call per sigma before 0, fresh noise before every call but the first
        self.assertEqual([round(s, 4) for _, s in model.calls], [1.0, 0.75, 0.25])
        self.assertEqual(drawn, [(1.0, 0.75), (0.75, 0.25)])
        # three steps, two of them add noise: the scale runs 1.10 then 0.90; call 2 sees the clean estimate noised to
        # 0.75 with noise x 1.10, call 3 to 0.25 with x 0.90
        torch.testing.assert_close(model.calls[1][0], 0.75 * 1.10 * noise + 0.25 * clean)
        torch.testing.assert_close(model.calls[2][0], 0.25 * 0.90 * noise + 0.75 * clean)
        # the last step to 0 returns the clean estimate as it is
        torch.testing.assert_close(out, clean)

    def test_explicit_scales_override_the_settings(self):
        from forge_h3.sampler import sample_h3_image_lcm

        clean = torch.zeros(1, 1, 2, 2)
        noise = torch.ones_like(clean)
        model = FakeModel(clean)
        sample_h3_image_lcm(model, clean, torch.tensor([1.0, 0.5, 0.25, 0.0]), noise_sampler=lambda *_: noise,
                            s_noise=2.0, s_noise_end=0.5)
        torch.testing.assert_close(model.calls[1][0], 0.5 * 2.0 * noise)
        torch.testing.assert_close(model.calls[2][0], 0.25 * 0.5 * noise)

    def test_the_callback_sees_every_step(self):
        from forge_h3.sampler import sample_h3_image_lcm

        clean = torch.zeros(1, 1, 2, 2)
        seen = []
        sample_h3_image_lcm(FakeModel(clean), clean, torch.tensor([1.0, 0.5, 0.0]),
                            callback=lambda d: seen.append(d["i"]), noise_sampler=lambda *_: torch.zeros_like(clean))
        self.assertEqual(seen, [0, 1])


if __name__ == "__main__":
    unittest.main()
