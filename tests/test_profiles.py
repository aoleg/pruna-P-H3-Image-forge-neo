"""The model profiles and Pruna's 8-step schedule."""

import unittest

from forge_h3.profiles import (
    H3,
    PH3_IMAGE,
    PH3_IMAGE_SIGMAS,
    for_text_encoder,
    ph3_image_sigmas,
)

try:
    import torch
except ImportError:
    torch = None


class ProfileTests(unittest.TestCase):
    def test_the_text_encoder_picks_the_profile(self):
        self.assertIs(for_text_encoder("h3"), H3)
        self.assertIs(for_text_encoder("ph3-image"), PH3_IMAGE)
        self.assertEqual((PH3_IMAGE.name, PH3_IMAGE.steps, PH3_IMAGE.cfg, PH3_IMAGE.shift), ("ph3", 8, 1.0, 1.0))
        self.assertEqual((H3.name, H3.shift), ("h3", 12.0))


@unittest.skipIf(torch is None, "needs torch")
class ScheduleTests(unittest.TestCase):
    def test_eight_steps_give_prunas_grid_as_it_is(self):
        sigmas = ph3_image_sigmas(8, sigma_min=0.001, sigma_max=1.0)
        self.assertEqual(sigmas.dtype, torch.float32)
        torch.testing.assert_close(sigmas, torch.tensor(PH3_IMAGE_SIGMAS, dtype=torch.float32))

    def test_other_step_counts_resample_the_same_curve(self):
        for n in (4, 12, 20):
            with self.subTest(n=n):
                sigmas = ph3_image_sigmas(n, sigma_min=0.001, sigma_max=1.0)
                self.assertEqual(sigmas.shape, (n + 1,))
                self.assertEqual((float(sigmas[0]), float(sigmas[-1])), (1.0, 0.0))
                self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
        # the grid's own points stay on the curve: 16 steps pass through every 8-step sigma
        sixteen = ph3_image_sigmas(16)
        torch.testing.assert_close(sixteen[::2], torch.tensor(PH3_IMAGE_SIGMAS, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
