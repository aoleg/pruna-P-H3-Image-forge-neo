"""The fused comfy-kitchen path of the DiT block: the same numbers as the eager path, on comfy-kitchen's CPU backend."""

import types
import unittest
from unittest.mock import patch

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class FusedTests(unittest.TestCase):
    def test_outside_forge_the_eager_path_is_taken_and_named(self):
        from forge_h3.native import kernels
        x = torch.zeros(2, 4)
        self.assertEqual(kernels.why_not(x, torch.bfloat16), "outside Forge")
        self.assertFalse(kernels.fused(x, torch.bfloat16))

    def test_the_modulated_norm_matches_the_eager_one_per_segment(self):
        from forge_h3.native import kernels
        from forge_h3.native.dit import _mod_scale_shift
        torch.manual_seed(0)
        norm = torch.nn.RMSNorm(32, eps=1e-6)
        with torch.no_grad():
            norm.weight.uniform_(0.8, 1.2)
        x = torch.randn(40, 32)
        shift, scale = torch.randn(6, 32) * 0.1, torch.randn(6, 32) * 0.1
        # a short text run, two rows of another tag, the picture's long segment and a reference
        segments = [(0, 5, 0), (5, 7, 2), (7, 33, 5), (33, 40, 4)]
        eager = _mod_scale_shift(norm(x), shift, scale, segments)
        fused = kernels.modulated_norm(norm, x, shift, scale, segments)
        torch.testing.assert_close(fused, eager, rtol=1e-5, atol=1e-5)

    def test_swiglu_matches_and_a_low_rank_lora_keeps_the_layer_forward(self):
        from forge_h3.native import kernels
        from forge_h3.native.dit import MLP
        torch.manual_seed(0)
        mlp = MLP(16, 24)
        x = torch.randn(7, 16)
        torch.testing.assert_close(mlp(x, fused=True), mlp(x))
        calls = []

        def stub(linear, inp, act):
            calls.append(act)
            gate, up = inp.chunk(2, dim=-1)
            return linear(torch.nn.functional.silu(gate) * up)

        with patch.object(kernels, "_forge_linear_input_act", stub):
            mlp(x, fused=True)
            self.assertEqual(calls, ["swiglu"])
            # an active low-rank LoRA hook on fc2 (native/lora.py) must not be skipped
            hook = types.SimpleNamespace(active=True)
            mlp.fc2.__dict__["forward"] = lambda inp, hook=hook, original=mlp.fc2.forward: original(inp)
            mlp.fc2.forward.active = True
            mlp(x, fused=True)
            self.assertEqual(calls, ["swiglu"])

    def test_a_whole_block_gives_the_same_output_fused_and_eager(self):
        from test_native import tiny_dit
        torch.manual_seed(1)
        model = tiny_dit(17)
        block = model.blocks[0]
        x = torch.randn(30, 256)
        t_emb = torch.randn(2, 16)
        mod_segments = [(0, 6, 0), (6, 30, 5)]
        angles = torch.rand(30, 96)
        from forge_h3.native.dit import rope_rotation_table
        rope = rope_rotation_table(angles, torch.float32)
        with torch.inference_mode():
            eager = block(x.clone(), t_emb, mod_segments, rope, {})
            fused = block(x.clone(), t_emb, mod_segments, rope, {"h3_fused": True})
        torch.testing.assert_close(fused, eager, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
