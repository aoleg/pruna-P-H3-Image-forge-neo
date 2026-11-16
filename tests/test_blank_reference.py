"""The blank reference's window: which steps see it, for any step count or schedule, and how the conditioning splits."""

import unittest

from forge_h3.blank_reference import (
    Window,
    active,
    blank_split,
    parse,
    thresholds,
    window,
)

GRID = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)


def percent_to_sigma(percent):
    # PredictionH3Image.percent_to_sigma: linear in the grid index, unsnapped
    if percent <= 0.0:
        return 1.0
    if percent >= 1.0:
        return 0.0
    pos = percent * 6
    i = min(int(pos), 5)
    return GRID[i] + (GRID[i + 1] - GRID[i]) * (pos - i)


def steps_on(win, sigmas):
    """1-based indices of the steps (sigmas before the closing 0) that use the blank reference."""
    sigma_from, sigma_to = thresholds(win, percent_to_sigma)
    return [i + 1 for i, s in enumerate(sigmas) if active(s, sigma_from, sigma_to)]


class WindowTests(unittest.TestCase):
    def test_the_sliders_become_a_window(self):
        self.assertEqual(window(True, 33, 100), Window(33.0, 100.0))
        self.assertIsNone(window(False, 33, 100))
        self.assertIsNone(window(True, 50, 50))
        self.assertIsNone(window(True, "x", 100))
        self.assertEqual(window(True, -5, 120), Window(0.0, 100.0))
        self.assertTrue(Window(0, 100).full)
        self.assertFalse(Window(33, 100).full)
        self.assertEqual(Window(33, 100).label(), "33-100%")
        self.assertEqual(parse("33-100%"), Window(33.0, 100.0))
        self.assertIsNone(parse(None))
        self.assertIsNone(parse("garbage"))

    def test_sixths_are_the_grid_steps_of_six(self):
        six = GRID[:-1]
        self.assertEqual(steps_on(Window(0, 100), six), [1, 2, 3, 4, 5, 6])
        self.assertEqual(steps_on(Window(33, 100), six), [3, 4, 5, 6])
        self.assertEqual(steps_on(Window(50, 100), six), [4, 5, 6])
        # a slider at 67 means the step at 2/3, though 67 is above 66.7
        self.assertEqual(steps_on(Window(67, 100), six), [5, 6])
        self.assertEqual(steps_on(Window(17, 100), six), [2, 3, 4, 5, 6])
        self.assertEqual(steps_on(Window(0, 50), six), [1, 2, 3])
        self.assertEqual(steps_on(Window(0, 67), six), [1, 2, 3, 4])
        self.assertEqual(steps_on(Window(33, 67), six), [3, 4])

    def test_another_step_count_switches_at_the_same_noise_level(self):
        # Simple at 8 steps on the grid: positions 6, 5.25, 4.5, ... 0.75
        eight = [percent_to_sigma(k / 8) for k in range(8)]
        on = steps_on(Window(50, 100), eight)
        self.assertTrue(all(eight[i - 1] <= 0.75 + 1e-6 for i in on))
        self.assertTrue(all(eight[i - 1] > 0.75 for i in range(1, 9) if i not in on))
        self.assertEqual(on, [5, 6, 7, 8])
        # a shifted 8-step curve (H3-Image 8-step) keeps the same rule: the steps at or below the threshold, which
        # sits half a percent of the denoising above sigma 0.75 (its step 6 is at 0.7526, inside that slack)
        curve = [5.069 * t / (1 + 4.069 * t) for t in [1 - k / 8 for k in range(8)]]
        sigma_from, _ = thresholds(Window(50, 100), percent_to_sigma)
        self.assertLess(sigma_from - 0.75, 0.005)
        on = steps_on(Window(50, 100), curve)
        self.assertEqual(on, [i + 1 for i, s in enumerate(curve) if s <= sigma_from])
        self.assertEqual(on, [6, 7, 8])


class SplitTests(unittest.TestCase):
    def test_the_two_prompts_split_by_length(self):
        # prefix 10 (label and vision block), prompt 4 tokens: 14 with the picture, 4 without
        self.assertEqual(blank_split(18, 10), (14, 4))
        # an empty prompt: the prefix alone, then one pad token
        self.assertEqual(blank_split(11, 10), (10, 1))
        with self.assertRaises(ValueError):
            blank_split(17, 10)
        with self.assertRaises(ValueError):
            blank_split(10, 10)


if __name__ == "__main__":
    unittest.main()
