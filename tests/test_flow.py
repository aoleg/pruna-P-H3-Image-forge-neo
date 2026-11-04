"""Forge Neo's generation order replayed on CPU, with the real engine, VAE and DiT at toy sizes.

Forge's own order (modules/processing.py): scripts.before_process, model load, scripts.process, p.setup_conds
(get_learned_conditioning), then p.sample with process_before_every_sampling, decode_first_stage and
scripts.postprocess. Forge itself is replaced by forge_stubs and the fakes below.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import forge_stubs
from test_contracts import DIT, PH3_TE, PH3_VAE, TE, VAE, checkpoint

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

from forge_h3 import integration
from forge_h3.contracts import H3Error, raise_pending_error, set_pending_error

WIDTH, HEIGHT = 96, 64


class Txt2Img:
    def __init__(self):
        self.prompt, self.negative_prompt = "a bird takes off", ""
        self.n_iter, self.batch_size = 1, 1
        self.width, self.height = WIDTH, HEIGHT
        self.enable_hr = self.restore_faces = False
        self.image_mask = None
        self.subseed_strength, self.seed_resize_from_w, self.seed_resize_from_h = 0, -1, -1
        self.seeds, self.subseeds = [123], [0]
        self.distilled_cfg_scale, self.is_api = 3.5, False
        self.extra_generation_params, self.override_settings = {}, {}
        self.clear_prompt_cache = Mock()
        # Script: None; the H3 panel has no inputs of its own
        self.script_args = [0]


class Img2Img(Txt2Img):
    def __init__(self):
        super().__init__()
        self.init_images = []
        self.resize_mode, self.denoising_strength = 0, 0.75


class FakeTextEngine:
    """Records the pictures handed with the prompt; its vision block spans the first three tokens."""

    def __init__(self):
        self.images, self.vision_spans = [], []

    def __call__(self, texts, images=()):
        self.images = list(images)
        self.vision_spans = [(0, 3)] if images else []
        return [torch.randn(7, 48) for _ in texts]


def forge_modules(files):
    info = types.SimpleNamespace(filename=str(files[0]))
    opts = types.SimpleNamespace(sd_model_checkpoint="model", forge_additional_modules=[str(f) for f in files[1:]],
                                 forge_preset="sd")

    class ImageRNG:
        def __init__(self, shape, seeds, **kwargs):
            self.shape, self.seeds = shape, seeds

        def next(self):
            return torch.randn((len(self.seeds), *self.shape))

    modules = forge_stubs.module("modules")
    modules.processing = forge_stubs.module("modules.processing", StableDiffusionProcessingImg2Img=Img2Img)
    modules.shared = forge_stubs.module("modules.shared", opts=opts,
                                        state=types.SimpleNamespace(interrupted=False, skipped=False))
    modules.sd_models = forge_stubs.module("modules.sd_models", get_closet_checkpoint_match=lambda value: info)
    modules.rng = forge_stubs.module("modules.rng", ImageRNG=ImageRNG)
    presets = forge_stubs.module("modules_forge.presets")
    forge = forge_stubs.module("modules_forge", presets=presets,
                               main_entry=forge_stubs.module("modules_forge.main_entry", module_list={}))
    patches = forge_stubs.module("forge_h3.native.patches", begin_sampling=Mock(), end_sampling=Mock())
    return {"modules": modules, "modules.processing": modules.processing, "modules.shared": modules.shared,
            "modules.sd_models": modules.sd_models, "modules.rng": modules.rng,
            "modules_forge": forge,
            "modules_forge.presets": presets, "modules_forge.main_entry": forge.main_entry,
            "forge_h3.native.patches": patches}


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class FlowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = self.root = Path(tmp.name)
        files = [checkpoint(root / f"{name}.safetensors", tensors) for name, tensors in
                 (("model", DIT), ("encoder", TE), ("vae", VAE))]
        stubs = {**forge_stubs.engine_modules(), **forge_modules(files)}
        self.enterContext(patch.dict(sys.modules, stubs))
        for name in ("forge_h3.native.engine", "forge_h3.native.model", "forge_h3.native.text_engine",
                     "forge_h3.native.presets"):
            sys.modules.pop(name, None)
        self.addCleanup(set_pending_error, None)
        self.engine = self.make_engine()

    def make_engine(self):
        from test_native import tiny_dit

        from forge_h3.native.engine import MiniMaxH3Engine
        from forge_h3.native.video_vae import MiniMaxH3VideoVAE
        torch.manual_seed(0)
        vae = MiniMaxH3VideoVAE(ch=32, num_layers=1)
        with torch.no_grad():
            for p in vae.parameters():
                torch.nn.init.normal_(p, std=0.02)
        engine = object.__new__(MiniMaxH3Engine)
        dit = tiny_dit(17)
        engine.forge_objects = types.SimpleNamespace(
            vae=types.SimpleNamespace(patcher=None, device="cpu", vae_dtype=torch.float32,
                                      first_stage_model=vae.requires_grad_(False)),
            unet=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dit)),
            clip=types.SimpleNamespace(patcher=None))
        engine.text_processing_engine_h3 = FakeTextEngine()
        engine.is_h3, engine.shift, engine.generation = True, 12.0, None
        engine.references = []
        return engine

    def run_until_sampling(self, p):
        """before_process .. process_before_every_sampling, as Forge calls them; returns the conditioning."""
        integration.before_process(p)
        p.sd_model = self.engine
        integration.process(p)
        cond = self.engine.get_learned_conditioning([p.prompt])
        integration.before_sampling(p, torch.zeros(1, 4, HEIGHT // 8, WIDTH // 8))
        return cond

    def test_txt2img_samples_one_packed_still_image(self):
        p = Txt2Img()
        cond = self.run_until_sampling(p)
        self.assertEqual((p.h3_request.width, p.h3_request.height, p.h3_request.references), (WIDTH, HEIGHT, 0))
        self.assertEqual([m.role for m in p.h3_components.models], ["dit", "text_encoder", "vae"])
        shapes = self.engine.generation.shapes
        self.assertEqual((shapes.video, shapes.audio), ((1, 24, 1, HEIGHT // 16, WIDTH // 16), (1, 32, 2, 2)))
        self.assertEqual(tuple(p.modified_noise.shape), (1, 1, 1, shapes.size))
        self.assertEqual(p.rng.shape, (1, 1, shapes.size))
        self.assertEqual((self.engine.generation.refs, self.engine.generation.vision_spans), ([], []))
        # outside the h3 preset the Shift slider belongs to another model: H3 keeps its own
        self.assertEqual(p.distilled_cfg_scale, 12.0)
        # one sampler step through the real DiT, then the decode Forge saves
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertEqual(out.shape, p.modified_noise.shape)
        self.assertTrue(torch.isfinite(out).all())
        image = self.engine.decode_first_stage(p.modified_noise)
        self.assertEqual(tuple(image.shape), (1, 3, HEIGHT, WIDTH))
        self.assertTrue(float(image.min()) >= -1.0 and float(image.max()) <= 1.0)
        integration.postprocess(p, None)
        self.assertIsNone(self.engine.generation)
        self.assertIsNone(dit.generation)

    def test_p_h3_image_takes_its_own_shift_outside_its_preset(self):
        files = [checkpoint(self.root / f"{name}.safetensors", tensors) for name, tensors in
                 (("model", DIT), ("ph3_te", PH3_TE), ("ph3_vae", PH3_VAE))]
        sys.modules["modules.shared"].opts.forge_additional_modules = [str(f) for f in files[1:]]
        sys.modules["modules.shared"].opts.forge_preset = "h3"
        p = Txt2Img()
        self.run_until_sampling(p)
        self.assertEqual((p.h3_components.profile.name, p.distilled_cfg_scale), ("ph3", 1.0))
        self.assertEqual(p.extra_generation_params["H3 Model"], "P-H3-Image")
        sys.modules["modules.shared"].opts.forge_preset = "ph3"
        p = Txt2Img()
        p.distilled_cfg_scale = 2.0
        self.run_until_sampling(p)
        self.assertEqual(p.distilled_cfg_scale, 2.0)

    def test_the_h3_preset_keeps_the_slider_and_fills_the_api_default(self):
        sys.modules["modules.shared"].opts.forge_preset = "h3"
        p = Txt2Img()
        p.distilled_cfg_scale = 6.0
        self.run_until_sampling(p)
        self.assertEqual(p.distilled_cfg_scale, 6.0)
        p = Txt2Img()
        p.is_api = True
        sys.modules["modules.shared"].opts.h3_t2i_dcfg = 10.0
        self.run_until_sampling(p)
        self.assertEqual(p.distilled_cfg_scale, 10.0)

    def test_set_shift_takes_the_size_newer_forge_passes(self):
        # Forge Neo after d70373e calls set_shift(shift=..., width=..., height=...)
        self.engine.set_shift(shift=10.0, width=WIDTH, height=HEIGHT)
        self.assertEqual((self.engine.predictor_shift, self.engine.shift), (10.0, 10.0))
        self.engine.set_shift(0)
        self.assertEqual(self.engine.shift, 12.0)

    def test_reference_pictures_ride_as_condition_rows_before_the_targets(self):
        # the engine's picture path, which img2img editing will feed: <Picture 1> at its own size
        p = Txt2Img()
        integration.before_process(p)
        p.sd_model = self.engine
        integration.process(p)
        self.engine.set_references([torch.rand(1, 32, 128, 3)])
        cond = self.engine.get_learned_conditioning([p.prompt])
        self.assertEqual(len(self.engine.text_processing_engine_h3.images), 1)
        integration.before_sampling(p, torch.zeros(1, 4, HEIGHT // 8, WIDTH // 8))
        generation = self.engine.generation
        self.assertEqual([(r["kind"], r["latent_h"], r["latent_w"], tuple(r["latent"].shape)) for r in generation.refs],
                         [("image", 2, 8, (1, 24, 1, 2, 8))])
        self.assertEqual(generation.vision_spans, [(0, 3)])
        dit = self.engine.forge_objects.unet.model.diffusion_model
        out = dit(p.modified_noise, torch.tensor([900.0]), cond[0].unsqueeze(0))
        self.assertTrue(torch.isfinite(out).all())
        layout = next(iter(dit._layouts.values()))
        self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "audio", "video"])
        # the next request without pictures drops them and the cached conditioning that carried them
        p = Txt2Img()
        self.run_until_sampling(p)
        p.clear_prompt_cache.assert_called()
        self.assertEqual((self.engine.references, self.engine.generation.refs), ([], []))

    def test_img2img_is_refused_for_now(self):
        with self.assertRaisesRegex(H3Error, "img2img is not supported yet"):
            integration.before_process(Img2Img())
        # the pending error reaches the model, which Forge calls directly
        dit = self.engine.forge_objects.unet.model.diffusion_model
        with self.assertRaisesRegex(H3Error, "img2img"):
            dit(torch.zeros(1, 1, 1, 8), torch.tensor([900.0]), torch.zeros(1, 7, 48))
        with self.assertRaisesRegex(H3Error, "img2img"):
            self.engine.encode_first_stage(torch.zeros(1, 3, HEIGHT, WIDTH))

    def test_a_batch_of_pictures_needs_batch_count(self):
        p = Txt2Img()
        p.batch_size = 2
        with self.assertRaisesRegex(H3Error, "Batch Size to 1"):
            integration.before_process(p)
        p = Txt2Img()
        p.n_iter = 4
        integration.before_process(p)
        self.assertIsNotNone(p.h3_request)

    def test_other_checkpoints_are_left_alone(self):
        sys.modules["modules.sd_models"].get_closet_checkpoint_match = lambda value: types.SimpleNamespace(
            filename=str(checkpoint(self.root / "sd.safetensors", {"unet.weight": ("F16", [4, 4])})))
        p = Txt2Img()
        p.batch_size = 4
        integration.before_process(p)
        self.assertIsNone(p.h3_request)
        integration.process(p)
        integration.before_sampling(p, torch.zeros(1, 4, 8, 8))
        self.assertFalse(hasattr(p, "modified_noise"))

    def test_script_rejection_prints_one_line_and_stops_at_the_model(self):
        p = Txt2Img()
        p.width = 100
        with patch("builtins.print") as printed:
            integration.script_before_process(p)
        self.assertRegex(printed.call_args.args[0], r"^\[MiniMax H3\] H3 width and height must be multiples of 32")
        with self.assertRaisesRegex(H3Error, "multiples of 32"):
            raise_pending_error()
        # the model Forge loaded is checked too
        p = Txt2Img()
        integration.before_process(p)
        p.sd_model = types.SimpleNamespace()
        with self.assertRaisesRegex(H3Error, "did not load the H3 model"):
            integration.process(p)

    def test_nan_after_sampling_or_decoding_is_reported_instead_of_a_black_picture(self):
        self.run_until_sampling(Txt2Img())
        shapes = self.engine.generation.shapes
        latent = torch.zeros(1, 1, 1, shapes.size)
        latent[..., 3] = float("nan")
        with self.assertRaisesRegex(H3Error, "sampling produced NaN"):
            self.engine.decode_first_stage(latent)
        with self.assertRaisesRegex(RuntimeError, "does not belong"):
            self.engine.decode_first_stage(torch.zeros(1, 1, 1, 8))
        broken = types.SimpleNamespace(decode=lambda z: torch.full((1, 3, 1, 4, 4), float("inf"), dtype=z.dtype))
        self.engine.forge_objects.vae = types.SimpleNamespace(patcher=None, device="cpu", vae_dtype=torch.float16,
                                                              first_stage_model=broken)
        with self.assertRaisesRegex(H3Error, "VAE produced NaN.*--fp32-vae"):
            self.engine.decode_first_stage(torch.zeros_like(latent))


if __name__ == "__main__":
    unittest.main()
