# P-H3-Image and MiniMax H3 stills for Forge Neo

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
![Status](https://img.shields.io/badge/Status-P--H3--Image_tested-brightgreen)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

An extension for [Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) that runs the MiniMax H3 architecture as a **still-image model**: Pruna's [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy) for text-to-image and editing, and H3 fine-tunes such as lodestones' [chroma-h3](https://huggingface.co/lodestones/chroma-h3) through their LoRAs, with the checkpoint selector, presets and Generate button you already know.

> [!Note]
> **A personal note and recommendation, after testing P-H3-Image generation and editing on an RTX 5090.** P-H3-Image is much better as an **edit-and-upscale** model than as a text-to-image model. It is a very good image restoration and upscaling model: in edit mode it does 8 MP and more without running out of memory, in 8 steps, in seconds. Text-to-image is rather meh, because small details come out soft. Give it a picture to work from.
>
> **Still untested:** chroma-h3, and the stock MiniMax H3 checkpoint with Pruna's P-H3-Image LoRA.
>
> This is a fork of [minimax-h3-forge-neo](https://github.com/eduardoabreu81/minimax-h3-forge-neo) with everything video and audio removed: no clips, no sound, no FFmpeg, no keyframes, guides or motion control. The extension needs an up-to-date **Forge Neo** (the `neo` branch, with comfy-kitchen 0.2.37 or later). On an older version it stays disabled and tells you so in the console; Forge Neo itself keeps working as usual.

## What it does

- **P-H3-Image text-to-image** at 8 MP in 8 steps, no CFG, with Pruna's own schedule: select the P-H3-Image checkpoint, text encoder and VAE, pick the `ph3` UI preset and generate.
- **P-H3-Image edits** in img2img: the input image is `<Picture 1>`, the ImageStitch Integrated gallery holds up to 7 references, and the prompt names them.
- **H3 fine-tunes as still models**: an H3 checkpoint with its 32B text encoder and video VAE, plus the fine-tune's LoRA through Forge's `<lora:name:1>` syntax. chroma-h3's recipe is the `h3` preset.
- **Forge's own machinery**: its loader, samplers, memory management, quantized formats (int8, w4a8, w6a8, fp8, GGUF) and LoRA handling. Nothing is installed into Forge's Python environment.

A generation is one video latent frame plus the two-step audio stub the model expects, sampled together and decoded as a picture; the audio is never decoded, so no audio VAE is needed.

## Installation

1. In Forge Neo, open **Extensions → Install from URL**.
2. Paste `https://github.com/aoleg/pruna-P-H3-Image-forge-neo`, click **Install** and restart the WebUI.
3. Check the console at startup for `[MiniMax H3] native backend enabled`.
4. Download one set of model files below.

### P-H3-Image

Files from [PrunaAI/P-H3-Image-Comfy](https://huggingface.co/PrunaAI/P-H3-Image-Comfy). Pick one file per part; precisions can be mixed.

| Part | Folder | bf16 | int8 (`int8_convrot`) | w4a8 |
| --- | --- | ---: | ---: | ---: |
| Checkpoint `p-h3-image_*` | `models/Stable-diffusion` | 40.5 GB | 21.2 GB | 13.4 GB |
| Text encoder `p-h3-image_text_encoder_*` | `models/text_encoder` | 9.1 GB | 5.1 GB | 3.6 GB |
| VAE `p-h3-image_vae_*` | `models/VAE` | 5.6 GB | 3.1 GB | (use int8) |

Pruna's own figures for ComfyUI: the bf16 set peaks at about 56 GB of VRAM at 8 MP, the int8 set at 31 GB, the w4a8 set with the int8 VAE at 21 GB. Forge Neo offloads what does not fit, at a cost in speed.

If you already have a MiniMax H3 FL2VA checkpoint, the 2 GB `loras/p-h3-image_lora_bf16.safetensors` applies P-H3-Image to it: select the H3 checkpoint, the P-H3-Image text encoder and VAE, and add `<lora:p-h3-image_lora_bf16:1>` to the prompt.

The **prompt upsampler** of Pruna's workflows (a Qwen3.5-9B text generator) is not part of this extension. P-H3-Image is sensitive to prompting: write full, specific prompts, or run the upsampler elsewhere and paste its output.

### chroma-h3 and other H3 fine-tunes

| Part | File | Folder |
| --- | --- | --- |
| H3 checkpoint | any FL2VA checkpoint, for example [minimax_h3_fl2va_pruned_int8_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) (19.5 GiB) | `models/Stable-diffusion` |
| Text encoder | [qwen3vl_32b_minimax_h3_int8_convrot](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) (25.3 GiB), Merserk's INT4 or unsloth's GGUF | `models/text_encoder` |
| Video VAE | [minimax_h3_video_vae_fp16](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) (4.9 GiB) or Kijai's int8 | `models/VAE` |
| chroma-h3 fine-tune | [chroma-h3-comfyui.safetensors](https://huggingface.co/lodestones/chroma-h3/tree/main/lora) (2.9 GB) | `models/Lora` |
| chroma-h3 few-step | [chroma-h3_vspace_dmd8-comfyui.safetensors](https://huggingface.co/lodestones/chroma-h3/tree/main/lora) (6.6 GB) | `models/Lora` |

Use the `-comfyui` LoRA files; the diffusers-format ones in the same folder do not load in Forge. The audio VAE of the video model is not needed and is refused if selected.

## Settings

Selecting an H3 checkpoint applies the matching profile's sampler, schedule and CFG, and the UI presets carry the rest:

| | `ph3` preset: P-H3-Image | `h3` preset: H3 fine-tunes |
| --- | --- | --- |
| Sampler, schedule | Euler, **P-H3-Image** (Pruna's 8-step grid) | Euler, Simple |
| Steps, CFG | 8, 1 | 30, 3 (the DMD LoRA: 8, 1) |
| Shift | 1 | 3 |
| Size | about 8 MP: 2912×2912, 3360×2496, 3872×2176 and the 32-multiples around them | about 1 MP |

The P-H3-Image schedule is the model's own sigma grid and ignores Shift; another step count resamples the same curve. Batch Size stays 1 (use Batch Count); Hires. fix and the Refiner are hidden while an H3 checkpoint is selected.

## Editing with P-H3-Image

**Settings → MiniMax H3 → [H3] Enable Reference** (on by default, like Forge's Klein and Krea 2 settings) decides what img2img does. On, it is a P-H3-Image edit, with the P-H3-Image text encoder and VAE selected:

- the input image is `<Picture 1>`, the picture to edit;
- turn on **ImageStitch Integrated** and add the references to its gallery: `<Picture 2>`, `<Picture 3>`, up to 8 pictures in all;
- write the instruction and name the pictures ("put the hat from <Picture 2> on the person in <Picture 1>"); a short instruction gets `Edit <Picture 1>. ` in front, an instruction that already starts that way, or an edit brief from Pruna's upsampler, is used as it is;
- Width and Height set the output; edits work best at about 4 MP (2048×2048, 2368×1760, 2720×1536). Resize mode and Denoising strength are not used: the model generates the whole picture from noise, conditioned on the pictures, each scaled within the output area for the VAE and within 1 MP for the text encoder.

Off, img2img is Forge's own with any H3 model: the input image is encoded, noised at Denoising strength and denoised on the profile's schedule. Other H3 models have no edit mode, so with the setting on img2img is refused for them.

## Known unknowns

- chroma-h3 and the stock H3 checkpoint with Pruna's LoRA have not been run yet. chroma-h3's own inference layout is not published; the extension runs it as P-H3-Image is run (one latent frame, a two-step audio stub). If that is not how it was trained, the pictures will say so.
- A lone latent through the stock video VAE is decoded as a chunk of copies, keeping the first real frame, as ComfyUI does now; the earlier rule (decode the latent alone, keep its last frame) may look different.
- P-H3-Image's text-to-image is soft in small detail at every size tried. The sampling path matches ComfyUI's stage by stage, and the launch flags that lower precision made no difference, so the comparison left is the same prompt in ComfyUI.

## Local tests (developers)

```bash
python -m unittest discover -s tests -v
python -m ruff check .
```

The tests use the real tensor names and shapes of the published files (`tests/fixtures/h3_headers.json.gz`, no weights) and stand-ins for Forge Neo (`tests/forge_stubs.py`). They do not replace a GPU run.

## Credits

- **[minimax-h3-forge-neo](https://github.com/eduardoabreu81/minimax-h3-forge-neo)** by Eduardo Abreu, the video extension this fork starts from
- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)** by MiniMax
- **[Pruna AI](https://huggingface.co/PrunaAI/P-H3-Image-Comfy)**: P-H3-Image
- **[lodestones](https://huggingface.co/lodestones/chroma-h3)**: chroma-h3
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)**: reference implementation of both models
- **[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3)**: the H3 model files
- **[Kijai](https://huggingface.co/Kijai)**: the INT8 video VAE and the W4A8 checkpoints
- **[Merserk](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)**: the INT4 text encoder
- **[unsloth](https://huggingface.co/unsloth/MiniMax-H3-GGUF)**: GGUF checkpoints

## License

AGPL-3.0, see [LICENSE](LICENSE). The model weights have their own terms: P-H3-Image and MiniMax H3 are under the MiniMax H3 Community License, which excludes some regions and requires "Powered by MiniMax H3" on products built with them; the Qwen3-VL parts of P-H3-Image's text encoder are Apache 2.0.
