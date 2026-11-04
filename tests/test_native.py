"""Native backend checks on CPU: layouts against the real checkpoint headers, small forward passes and conversions.

The fixture holds only tensor names, dtypes and shapes of the published files (no weights). Needs torch and
comfy-kitchen; skipped in an environment without them.
"""

import gzip
import json
import unittest
from pathlib import Path

try:
    import comfy_kitchen  # noqa: F401
    import torch
except ImportError:
    torch = None

HEADERS = Path(__file__).with_name("fixtures") / "h3_headers.json.gz"
AUX = (".comfy_quant", ".weight_scale", ".input_scale", ".weight_s_channel", ".weight_s_rel", ".weight_codebook")
DIT_FILES = ("minimax_h3_fl2va_pruned_int8_convrot", "minimax_h3_fl2va_int8_convrot",
             "minimax_h3_fl2va_pruned_w6a8", "minimax_h3_fl2va_pruned_fp8_scaled",
             "p-h3-image_bf16", "p-h3-image_w4a8")


def headers():
    # stored compactly as name -> [dtype, shape]; returned in the safetensors header form
    with gzip.open(HEADERS, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return {name: {k: v if k == "__metadata__" else {"dtype": v[0], "shape": v[1]} for k, v in header.items()}
            for name, header in data.items()}


def meta_state_dict(header):
    return {k: torch.empty(v["shape"], device="meta") for k, v in header.items() if k != "__metadata__"}


def layout_problems(module, header):
    """Missing, unexpected and mismatched keys of a module against a checkpoint header (quantization aside)."""
    file_keys = {k: v for k, v in header.items() if k != "__metadata__" and not k.endswith(AUX)}
    quantized = {k[: -len(".comfy_quant")] for k in header if k.endswith(".comfy_quant")}
    mine = {k: tuple(v.shape) for k, v in module.state_dict().items()}
    problems = [f"missing {k}" for k in sorted(set(mine) - set(file_keys))]
    problems += [f"unexpected {k}" for k in sorted(set(file_keys) - set(mine))]
    for k in sorted(set(mine) & set(file_keys)):
        shape = tuple(file_keys[k]["shape"])
        # quantized weights may be packed along the input axis; the output axis always matches
        same = shape[0] == mine[k][0] if k.rsplit(".", 1)[0] in quantized and k.endswith(".weight") else shape == mine[k]
        if not same:
            problems.append(f"shape {k}: {mine[k]} != {shape}")
    return problems


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class CheckpointLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()

    def test_dit_matches_every_published_build(self):
        from forge_h3.native.detect import detect
        from forge_h3.native.transformer import MiniMaxH3Model
        for name in DIT_FILES:
            with self.subTest(name):
                header = self.headers[name]
                config = detect(header, metadata=header.get("__metadata__"))
                with torch.device("meta"):
                    model = MiniMaxH3Model(**config)
                self.assertEqual(layout_problems(model, header), [])
                self.assertEqual(config["num_layers"], 50)
                self.assertEqual("adaln_curve_grid" in config, "pruned" in name or "p-h3" in name)
        # P-H3-Image's curve table is finer than Comfy-Org's pruned one; both are read from the shape
        from forge_h3.native.detect import detect
        self.assertEqual((detect(self.headers["p-h3-image_bf16"])["adaln_curve_grid"], detect(self.headers["p-h3-image_bf16"])["time_embed_dim"]), (4097, 16))
        self.assertEqual(detect(self.headers["minimax_h3_fl2va_pruned_int8_convrot"])["adaln_curve_grid"], 1025)

    def test_detection_ignores_other_models(self):
        from forge_h3.native.detect import detect
        self.assertIsNone(detect(self.headers["minimax_h3_video_vae_fp16"]))

    def test_comfy_org_vae_matches(self):
        from forge_h3.native.video_vae import MiniMaxH3VideoVAE
        with torch.device("meta"):
            self.assertEqual(layout_problems(MiniMaxH3VideoVAE(), self.headers["minimax_h3_video_vae_fp16"]), [])

    def test_ph3_image_vae_matches_both_published_builds(self):
        from forge_h3.native import vae
        from forge_h3.native.video_vae import PH3ImageVAE
        for name in ("p-h3-image_vae_bf16", "p-h3-image_vae_int8_convrot"):
            with self.subTest(name), torch.device("meta"):
                header = self.headers[name]
                self.assertTrue(vae.is_video_vae(header) and vae.is_ph3_vae(header))
                self.assertEqual(layout_problems(PH3ImageVAE(num_layers=vae.video_layers(header)), header), [])
        self.assertFalse(vae.is_ph3_vae(self.headers["minimax_h3_video_vae_fp16"]))

    def test_text_encoders_are_told_apart(self):
        from forge_h3.models import H3, PH3_IMAGE, _role, _variant
        for name, variant in (("qwen3vl_32b_minimax_h3_int8_convrot", H3), ("p-h3-image_text_encoder_bf16", PH3_IMAGE)):
            keys = set(self.headers[name]) - {"__metadata__"}
            self.assertEqual((_role(keys), _variant("text_encoder", self.headers[name], keys)), ("text_encoder", variant))
        for name, variant in (("minimax_h3_video_vae_fp16", H3), ("p-h3-image_vae_bf16", PH3_IMAGE)):
            keys = set(self.headers[name]) - {"__metadata__"}
            self.assertEqual((_role(keys), _variant("vae", self.headers[name], keys)), ("vae", variant))
        # the 4B encoder: 36 layers of 2560, the vision tower of Forge Neo's own 4B model, and the connector
        te = self.headers["p-h3-image_text_encoder_bf16"]
        self.assertEqual(te["model.language_model.embed_tokens.weight"]["shape"], [151936, 2560])
        self.assertEqual(max(int(k.split(".")[3]) for k in te if k.startswith("model.language_model.layers.")) + 1, 36)
        self.assertEqual(te["text_connector.proj.weight"], {"dtype": "F32", "shape": [5120, 5 * 2560]})

    def test_original_minimax_vae_converts_to_the_same_layout(self):
        from forge_h3.native import vae
        video = vae.convert_video_vae(meta_state_dict(self.headers["minimax_original_video_vae"]))
        expected = {k: tuple(v["shape"]) for k, v in self.headers["minimax_h3_video_vae_fp16"].items() if k != "__metadata__"}
        self.assertEqual({k: tuple(v.shape) for k, v in video.items()}, expected)

    def test_vae_roles_are_recognized(self):
        from forge_h3.native import vae
        for name in ("minimax_h3_video_vae_fp16", "minimax_original_video_vae"):
            self.assertTrue(vae.is_video_vae(self.headers[name]))
            self.assertFalse(vae.is_audio_vae(self.headers[name]))
        self.assertTrue(vae.is_audio_vae({"pre_block.attn.zero_k_bias": None}))
        self.assertEqual(vae.video_layers(self.headers["minimax_h3_video_vae_fp16"]), 36)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class ConversionTests(unittest.TestCase):
    def test_quantized_vae_keeps_its_quantization(self):
        from forge_h3.native import vae
        sd = {"decoder.blocks.0.attn.to_qkv.comfy_quant": torch.zeros(1),
              "decoder.blocks.0.attn.to_qkv.weight_scale": torch.ones(1)}
        self.assertTrue(vae.is_quantized(sd))
        self.assertFalse(vae.is_quantized({"decoder.blocks.0.attn.to_qkv.weight": torch.ones(1)}))
        converted = vae.convert_video_vae({"decoder.proj_out.comfy_quant": torch.zeros(1)})
        self.assertIn("decoder.proj_out.comfy_quant", converted)
        self.assertEqual(sorted(converted)[-2:], ["latents_mean", "latents_std"])


def tiny_dit(curve):
    from forge_h3.native.transformer import MiniMaxH3Model
    torch.manual_seed(0)
    # 128-dim heads: the rope rotates 96 of them, as in the full model
    model = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                           attention_head_dim=128, ffn_hidden_size=96, text_dim=48, time_embed_hidden_size=256,
                           time_embed_dim=16, adaln_curve_grid=curve)
    with torch.no_grad():
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)
        model.rope.inv_freq.copy_(1.0 / (10000 ** (torch.arange(16) / 16)))
        if curve:
            torch.nn.init.normal_(model.adaln_t_table)
    return model.requires_grad_(False)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class ForwardTests(unittest.TestCase):
    def test_streams_round_trip_through_the_packed_latent(self):
        from forge_h3.native.streams import still_shapes
        shapes = still_shapes(width=96, height=64)
        self.assertEqual(shapes.video, (1, 24, 1, 4, 6))
        self.assertEqual(shapes.audio, (1, 32, 2, 2))
        video, audio = torch.randn(shapes.video), torch.randn(shapes.audio)
        packed = shapes.pack(video, audio)
        self.assertEqual(packed.shape, (1, 1, 1, video.numel() + audio.numel()))
        self.assertEqual(shapes.size, video.numel() + audio.numel())
        v, a = shapes.unpack(packed)
        self.assertTrue(torch.equal(v, video) and torch.equal(a, audio))

    def test_both_adaln_forms_run_and_refined_text_is_reused(self):
        from forge_h3.native.layout import PackedLayout
        for curve in (None, 17):
            with self.subTest(curve=curve), torch.inference_mode():
                model = tiny_dit(curve)
                video, audio, context = torch.randn(1, 24, 1, 6, 10), torch.randn(1, 32, 2, 2), torch.randn(1, 7, 48)
                t = torch.tensor([700.0])
                payload = {"layout": PackedLayout(7, 1, 6, 10, 2)}
                raw = model.forward_streams([video, audio], t, context, minimax_payload=payload)
                refined = model.forward_streams([video, audio], t, model.preprocess_text_embeds(context), minimax_payload=payload)
                self.assertEqual([tuple(o.shape) for o in raw], [(1, 24, 1, 6, 10), (1, 32, 2, 2)])
                self.assertTrue(all(torch.isfinite(o).all() for o in raw))
                self.assertTrue(all(torch.allclose(a, b, atol=1e-5) for a, b in zip(raw, refined)))

    def test_packed_forward_matches_the_stream_forward(self):
        from forge_h3.native.streams import Generation, StreamShapes
        with torch.inference_mode():
            model = tiny_dit(17)
            shapes = StreamShapes(video=(1, 24, 1, 6, 10), audio=(1, 32, 2, 2))
            model.generation = Generation(shapes=shapes, seed=1)
            video, audio, context = torch.randn(shapes.video), torch.randn(shapes.audio), torch.randn(2, 7, 48)
            t = torch.tensor([700.0, 700.0])
            packed = model(torch.cat([shapes.pack(video, audio)] * 2), t, context)
            v, a = model.forward_streams([video, audio], t[:1], context[:1])
            self.assertTrue(torch.allclose(packed[:1], shapes.pack(v, a), atol=1e-5))
            layout = next(iter(model._layouts.values()))
            self.assertEqual([k for _, _, k in layout.segments], ["text", "audio", "video"])
            # the targets start right after the text; the audio rows sit on the video frame's time step
            (ta, _, _), (tv, _, _) = layout.segments[1:]
            self.assertEqual((layout.position_ids[ta, 0].item(), layout.position_ids[tv, 0].item()), (7.0, 7.0))

    def test_live_preview_takes_the_video_frame(self):
        from forge_h3.native.streams import StreamShapes, preview_frame
        shapes = StreamShapes(video=(1, 24, 1, 4, 6), audio=(1, 32, 2, 2))
        video = torch.arange(24 * 4 * 6, dtype=torch.float32).reshape(shapes.video)
        packed = shapes.pack(video, torch.zeros(shapes.audio))
        frame = preview_frame(packed, shapes)
        self.assertEqual(tuple(frame.shape), (1, 24, 4, 6))
        self.assertTrue(torch.equal(frame, video[:, :, 0]))
        # any other latent, or no generation, is left for Forge as it is
        other = torch.zeros(1, 4, 8, 8)
        self.assertIs(preview_frame(other, shapes), other)
        self.assertIs(preview_frame(packed, None), packed)

    def test_text_token_tags_mark_the_vision_blocks(self):
        from forge_h3.native.streams import text_token_tags
        self.assertIsNone(text_token_tags(7, []))
        self.assertEqual(text_token_tags(9, [(1, 4), (6, 12)]).tolist(), [1, 0, 0, 0, 1, 1, 0, 0, 0])

    def test_packed_forward_with_references_matches_the_stream_forward(self):
        from forge_h3.native.layout import PackedLayout
        from forge_h3.native.streams import Generation, StreamShapes, text_token_tags
        with torch.inference_mode():
            model = tiny_dit(17)
            shapes = StreamShapes(video=(1, 24, 1, 6, 10), audio=(1, 32, 2, 2))
            # reference pictures keep their own size: a portrait and a landscape reference for a landscape picture
            refs = [{"kind": "image", "latent_h": 8, "latent_w": 4, "latent": torch.randn(1, 24, 1, 8, 4)},
                    {"kind": "image", "latent_h": 6, "latent_w": 10, "latent": torch.randn(1, 24, 1, 6, 10)}]
            spans = [(0, 3), (4, 6)]
            video, audio, context = torch.randn(shapes.video), torch.randn(shapes.audio), torch.randn(1, 9, 48)
            t = torch.tensor([700.0])
            model.generation = Generation(shapes=shapes, seed=1, refs=refs, vision_spans=spans)
            packed = model(shapes.pack(video, audio), t, context)
            layout = model._layout(9, shapes, refs=refs)
            self.assertEqual([k for _, _, k in layout.segments], ["text", "ref_img", "ref_img", "audio", "video"])
            # each picture packs on its own grid, one time step apart, and the targets start after them
            (a1, b1, _), (a2, b2, _) = [s for s in layout.segments if s[2] == "ref_img"]
            self.assertEqual((b1 - a1, b2 - a2), (8 * 4 // 4, 6 * 10 // 4))
            self.assertEqual(layout.position_ids[a1, 0].item(), 9.0)
            self.assertEqual(layout.position_ids[a2, 0].item(), 10.0)
            tv, _, _ = next(s for s in layout.segments if s[2] == "video")
            self.assertEqual(layout.position_ids[tv, 0].item(), 11.0)
            payload = {"seed": 1, "refs": refs, "cond_video_latents": [r["latent"] for r in refs],
                       "text_token_tags": text_token_tags(9, spans), "layout": PackedLayout(9, 1, 6, 10, 2, refs=refs)}
            v, a = model.forward_streams([video, audio], t, context, minimax_payload=payload)
            self.assertTrue(torch.allclose(packed, shapes.pack(v, a), atol=1e-5))
            # the references change the prediction (the toy DiT couples weakly: a reference moves it by about 1e-4),
            # and other references get their own layout
            model.generation = Generation(shapes=shapes, seed=1)
            plain = model(shapes.pack(video, audio), t, context)
            self.assertGreater(float((packed - plain).abs().max()), 1e-6)
            self.assertIsNot(model._layout(9, shapes, refs=refs[:1]), layout)

    def test_the_layout_cache_is_bounded(self):
        from forge_h3.native.streams import StreamShapes
        from forge_h3.native.transformer import LAYOUT_CACHE_SIZE
        model = tiny_dit(None)
        shapes = StreamShapes(video=(1, 24, 1, 6, 10), audio=(1, 32, 2, 2))
        for text_len in range(LAYOUT_CACHE_SIZE + 2):
            model._layout(text_len + 1, shapes)
        self.assertLessEqual(len(model._layouts), LAYOUT_CACHE_SIZE)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class VaeTests(unittest.TestCase):
    def tiny_vae(self):
        from forge_h3.native.video_vae import MiniMaxH3VideoVAE
        torch.manual_seed(0)
        model = MiniMaxH3VideoVAE(ch=32, num_layers=1)
        with torch.no_grad():
            for p in model.parameters():
                torch.nn.init.normal_(p, std=0.02)
        return model.requires_grad_(False)

    def test_a_picture_is_one_latent_frame(self):
        model = self.tiny_vae()
        with torch.inference_mode():
            z = model.encode(torch.rand(1, 3, 1, 64, 96) * 2 - 1)
            self.assertEqual(tuple(z.shape), (1, 24, 1, 4, 6))
            self.assertEqual(model.decode_output_shape(z.shape), (1, 3, 1, 64, 96))
            out = model.decode(z)
            self.assertEqual(tuple(out.shape), (1, 3, 1, 64, 96))
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)
            # ComfyUI's rule for a lone latent: decoded as a chunk of five copies, the first real frame kept
            raw = model.latents_mean.view(1, -1, 1, 1, 1) + z * model.latents_std.view(1, -1, 1, 1, 1)
            chunk = model._finalize_pixels(model._decode_pixels(raw.repeat(1, 1, 5, 1, 1)))
            torch.testing.assert_close(out, chunk[:, :, 3:4])
            # a picture without the frame axis takes the same path
            self.assertTrue(torch.equal(model.encode(torch.zeros(1, 3, 64, 96)), model.encode(torch.zeros(1, 3, 1, 64, 96))))

    def test_the_ph3_image_vae_samples_its_posterior_and_decodes_one_frame(self):
        from forge_h3.native.video_vae import PH3_IMAGE_FRAME_POSITION, PH3ImageVAE
        torch.manual_seed(0)
        model = PH3ImageVAE(ch=32, num_layers=1)
        with torch.no_grad():
            for p in model.parameters():
                torch.nn.init.normal_(p, std=0.02)
        model.requires_grad_(False)
        self.assertNotIn("decoder.mask_token", model.state_dict())
        self.assertEqual(tuple(model.decoder.proj_out.weight.shape), (3 * 16 * 16, 2048))
        with torch.inference_mode():
            picture = torch.rand(1, 3, 1, 64, 96) * 2 - 1
            z = model.encode(picture)
            self.assertEqual(tuple(z.shape), (1, 24, 1, 4, 6))
            # the posterior sample is seeded: the same picture gives the same latent, and not the mean
            self.assertTrue(torch.equal(z, model.encode(picture)))
            mean = (model.quant_conv(model.encoder(model._normalize_pixels(picture))).float().chunk(2, dim=1)[0]
                    - model.latents_mean.view(1, -1, 1, 1, 1)) / model.latents_std.view(1, -1, 1, 1, 1)
            self.assertFalse(torch.allclose(z, mean))
            out = model.decode(z)
            self.assertEqual(tuple(out.shape), (1, 3, 1, 64, 96))
            self.assertTrue(0.0 <= float(out.min()) and float(out.max()) <= 1.0)
            ids = model.decoder._token_ids(1, 4, 6, "cpu", torch.float32)
            self.assertEqual(tuple(ids.shape), (1, 24, 3))
            self.assertTrue(bool((ids[..., 0] == PH3_IMAGE_FRAME_POSITION).all()))
            with self.assertRaisesRegex(ValueError, "single-frame"):
                model.decode(torch.zeros(1, 24, 2, 4, 6))
            with self.assertRaisesRegex(ValueError, "single pictures"):
                model.encode(torch.zeros(1, 3, 5, 64, 96))

    def test_video_frame_grid_still_round_trips(self):
        from forge_h3.native.video_vae import latent_frames, pixel_frames
        model = self.tiny_vae()
        with torch.inference_mode():
            z = model.encode(torch.rand(1, 3, 5, 64, 96) * 2 - 1)
            self.assertEqual(z.shape[2], latent_frames(5))
            self.assertEqual(tuple(model.decode(z).shape), (1, 3, pixel_frames(z.shape[2]), 64, 96))

    def test_encoder_layers_work_with_offloaded_weights(self):
        # issue #4: when VRAM is short Forge keeps these weights in fp32 system RAM and turns on manual cast; a
        # float64 input stands in for the GPU one (both make plain torch layers fail on the stored weights)
        from forge_h3.native.video_vae import CausalConv3d, TemporalIsolatedGroupNorm
        torch.manual_seed(0)
        conv, norm = CausalConv3d(32, 32, kernel_size=3, padding=1), TemporalIsolatedGroupNorm(8, 32)
        reference_conv, reference_norm = CausalConv3d(32, 32, kernel_size=3, padding=1), TemporalIsolatedGroupNorm(8, 32)
        reference_conv.load_state_dict(conv.state_dict())
        reference_norm.load_state_dict(norm.state_dict())
        reference_conv.double()
        reference_norm.double()
        for layer in (conv, norm):
            self.assertEqual((layer.weight_function, layer.bias_function), ([], []))
            layer.parameters_manual_cast = True
        x = torch.randn(1, 32, 2, 8, 8, dtype=torch.float64)
        with torch.inference_mode():
            torch.testing.assert_close(norm(x), reference_norm(x))
            torch.testing.assert_close(conv(x, pre_norm=norm, residual=x),
                                       reference_conv(x, pre_norm=reference_norm, residual=x))
        self.assertEqual({conv.weight.dtype, norm.weight.dtype}, {torch.float32})


if __name__ == "__main__":
    unittest.main()
