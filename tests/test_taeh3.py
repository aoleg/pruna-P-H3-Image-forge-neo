"""taeh3, the TAESD live preview of H3 pictures: layout against the published file, the frame it decodes, the hook's
source. CPU only, random weights."""

import os
import tempfile
import unittest
from unittest import mock

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

from test_native import headers


def full_decode(model, latent):
    """Every frame of one latent token, as taehv.py's parallel decode makes them: [N, 4, 3, 16h, 16w]."""
    from forge_h3.native.taeh3 import PATCH_SIZE, MemBlock

    n = latent.shape[0]
    x = latent
    for block in model:
        if isinstance(block, MemBlock):
            clip = x.view(n, -1, *x.shape[1:])
            past = F.pad(clip, (0, 0, 0, 0, 0, 0, 1, 0))[:, :clip.shape[1]].reshape(x.shape)
            x = block(x, past)
        else:
            x = block(x)
    x = F.pixel_shuffle(x, PATCH_SIZE).clamp(0, 1)
    return x.view(n, -1, *x.shape[1:])


def random_decoder(seed=0):
    from forge_h3.native import taeh3

    torch.manual_seed(seed)
    model = taeh3.decoder().eval()
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    return model


@unittest.skipIf(torch is None, "needs torch")
class Taeh3DecoderTests(unittest.TestCase):
    def test_decoder_matches_the_published_file(self):
        from forge_h3.native import taeh3

        header = headers()["taeh3"]
        mine = {f"decoder.{k}": tuple(v.shape) for k, v in taeh3.decoder().state_dict().items()}
        theirs = {k: tuple(v["shape"]) for k, v in header.items() if k.startswith("decoder.")}
        self.assertEqual(mine, theirs)
        self.assertEqual({v["dtype"] for v in header.values()}, {"F16"})

    def test_last_frame_equals_the_full_decode_comfyui_keeps(self):
        from forge_h3.native import taeh3

        model = random_decoder()
        latent = torch.randn(2, 24, 6, 5)
        with torch.no_grad():
            frames = full_decode(model, latent)
            last = taeh3.decode_last_frame(model, latent)
        # one token decodes to four frames; ComfyUI and taehv.py trim the first three
        self.assertEqual(tuple(frames.shape), (2, 4, 3, 96, 80))
        self.assertEqual(tuple(last.shape), (2, 3, 96, 80))
        torch.testing.assert_close(last, frames[:, -1], rtol=0, atol=1e-6)
        self.assertGreater(last.std().item(), 0.05)
        self.assertGreaterEqual(last.min().item(), 0)
        self.assertLessEqual(last.max().item(), 1)

    def test_preview_size_caps_the_long_side(self):
        from forge_h3.native.taeh3 import PREVIEW_SIDE, preview_size

        self.assertEqual(PREVIEW_SIDE, 2048)
        self.assertEqual(preview_size(4096, 4096), (2048, 2048))
        self.assertEqual(preview_size(2432, 3648), (1365, 2048))
        self.assertEqual(preview_size(1024, 1536), (1024, 1536))


@unittest.skipIf(torch is None, "needs torch")
class PreviewDecoderTests(unittest.TestCase):
    def preview(self, shapes, side=2048):
        from forge_h3.native import taeh3

        fallback = mock.Mock(return_value="rgb")
        model = taeh3.PreviewDecoder(lambda: shapes, fallback, side=side)
        model.decoder.load_state_dict(random_decoder().state_dict())
        return model, fallback

    def test_the_packed_latent_previews_its_picture(self):
        from forge_h3.native import taeh3
        from forge_h3.native.streams import still_shapes

        shapes = still_shapes(128, 96)  # latent 6x8, with P-H3's audio stub
        video = torch.randn(1, 24, 1, 6, 8)
        packed = shapes.pack(video, torch.randn(1, 32, 2, 2))
        model, fallback = self.preview(shapes)
        pixels = model(packed)
        fallback.assert_not_called()
        self.assertEqual(tuple(pixels.shape), (1, 3, 96, 128))
        self.assertEqual(pixels.dtype, torch.float32)
        with torch.no_grad():
            torch.testing.assert_close(pixels, taeh3.decode_last_frame(model.decoder, video[:, :, 0]))

    def test_a_large_picture_is_previewed_smaller(self):
        from forge_h3.native.streams import still_shapes

        shapes = still_shapes(256, 128, audio=False)
        packed = shapes.pack(torch.randn(1, 24, 1, 8, 16), torch.zeros(1, 32, 2, 0))
        model, _ = self.preview(shapes, side=64)
        self.assertEqual(tuple(model(packed).shape), (1, 3, 32, 64))

    def test_a_latent_of_another_generation_goes_to_the_rgb_preview(self):
        from forge_h3.native.streams import still_shapes

        model, fallback = self.preview(still_shapes(128, 96))
        foreign = torch.randn(1, 1, 1, 77)
        self.assertEqual(model(foreign), "rgb")
        fallback.assert_called_once_with(foreign)
        model, fallback = self.preview(None)
        self.assertEqual(model(foreign), "rgb")


@unittest.skipIf(torch is None, "needs torch")
class LoadTests(unittest.TestCase):
    def test_load_reads_the_decoder_and_refuses_another_tae(self):
        from safetensors.torch import save_file

        from forge_h3.native import taeh3

        state = {f"decoder.{k}": v.half() for k, v in random_decoder().state_dict().items()}
        state["encoder.0.weight"] = torch.zeros(64, 12, 3, 3, dtype=torch.float16)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "taeh3.safetensors")
            save_file(state, path)
            model = taeh3.load(path, lambda: None, None, "cpu", torch.float32)
            self.assertEqual(model.decoder[1].weight.dtype, torch.float32)
            torch.testing.assert_close(model.decoder[1].weight, state["decoder.1.weight"].float())
            # Kijai's 2D TAE: no "decoder." prefix
            save_file({"1.weight": torch.zeros(96, 24, 3, 3)}, path)
            with self.assertRaisesRegex(taeh3.NotTaeh3, "not madebyollin's taeh3"):
                taeh3.load(path, lambda: None, None)


class PreviewSourceTests(unittest.TestCase):
    def test_the_file_is_downloaded_once_into_forges_taesd_folder(self):
        from forge_h3.native import taeh3

        with tempfile.TemporaryDirectory() as models:
            target = os.path.join(models, "VAE-taesd", "taeh3.safetensors")
            downloaded = []

            def download(path, url):
                downloaded.append((path, url))
                os.makedirs(os.path.dirname(path))
                open(path, "wb").close()

            source = taeh3.PreviewSource(models, download, lambda path: f"decoder for {path}", mock.Mock())
            self.assertEqual(source.get(), f"decoder for {target}")
            self.assertEqual(source.get(), f"decoder for {target}")
            self.assertEqual(downloaded, [(target, taeh3.URL)])
            self.assertIn("62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors", taeh3.URL)

    def test_a_present_file_is_not_downloaded(self):
        from forge_h3.native import taeh3

        with tempfile.TemporaryDirectory() as models:
            os.makedirs(os.path.join(models, "VAE-taesd"))
            open(os.path.join(models, "VAE-taesd", "taeh3.safetensors"), "wb").close()
            download = mock.Mock()
            source = taeh3.PreviewSource(models, download, lambda path: "decoder", mock.Mock())
            self.assertEqual(source.get(), "decoder")
            download.assert_not_called()

    def test_a_failure_is_reported_once_and_the_rgb_preview_stays(self):
        from forge_h3.native import taeh3

        with tempfile.TemporaryDirectory() as models:
            download = mock.Mock(side_effect=OSError("offline"))
            report = mock.Mock()
            source = taeh3.PreviewSource(models, download, lambda path: "decoder", report)
            self.assertIsNone(source.get())
            self.assertIsNone(source.get())
            download.assert_called_once()
            report.assert_called_once()
            self.assertIn("offline", report.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
