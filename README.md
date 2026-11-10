# MiniMax-H3-Image, P-H3-Image and H3 stills for Forge Neo

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
![Status](https://img.shields.io/badge/Status-P--H3--Image_tested-brightgreen)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

An extension for [Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) that runs the MiniMax H3 architecture as a **still-image model**: MiniMax's own [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) and Pruna's [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy) for text-to-image and editing, and H3 fine-tunes such as lodestones' [chroma-h3](https://huggingface.co/lodestones/chroma-h3) through their LoRAs, with the checkpoint selector, presets and Generate button you already know.

> [!Note]
> **A personal note and recommendation, after testing P-H3-Image generation and editing on an RTX 5090.** P-H3-Image is much better as an **edit-and-upscale** model than as a text-to-image model. It is a very good image restoration and upscaling model: in edit mode it does 8 MP and more without running out of memory, in 8 steps, in seconds. Text-to-image is rather meh, because small details come out soft. Give it a picture to work from.
>
> **Still untested:** MiniMax-H3-Image (added in 0.9.0 from the published files and ComfyUI's code, not run on a GPU yet), chroma-h3, and the stock MiniMax H3 checkpoint with Pruna's P-H3-Image LoRA.
>
> This is a fork of [minimax-h3-forge-neo](https://github.com/eduardoabreu81/minimax-h3-forge-neo) with everything video and audio removed: no clips, no sound, no FFmpeg, no keyframes, guides or motion control. The extension needs an up-to-date **Forge Neo** (the `neo` branch, with comfy-kitchen 0.2.37 or later). On an older version it stays disabled and tells you so in the console; Forge Neo itself keeps working as usual.

## What it does

- **MiniMax-H3-Image**, MiniMax's image model: text-to-image and edits with up to 16 pictures, 4 to 20 MP, in 6 steps with no CFG. Select its checkpoint, the H3 text encoder and its VAE, and pick the `h3image` UI preset.
- **P-H3-Image text-to-image** at 8 MP in 8 steps, no CFG, with Pruna's own schedule: select the P-H3-Image checkpoint, text encoder and VAE, pick the `ph3` UI preset and generate.
- **P-H3-Image edits** in img2img: the input image is `<Picture 1>`, the ImageStitch Integrated gallery holds up to 7 references, and the prompt names them.
- **H3 fine-tunes as still models**: an H3 checkpoint with its 32B text encoder and video VAE, plus the fine-tune's LoRA through Forge's `<lora:name:1>` syntax. chroma-h3's recipe is the `h3` preset.
- **Forge's own machinery**: its loader, samplers, memory management, quantized formats (int8, w4a8, w6a8, fp8, GGUF) and LoRA handling. Nothing is installed into Forge's Python environment.

A generation is one video latent frame, decoded as a picture. The audio-video checkpoints (P-H3-Image, H3) also get the two-step audio stub they expect, sampled with the picture and never decoded; MiniMax-H3-Image has no audio stream at all. No audio VAE is needed.

## Installation

1. In Forge Neo, open **Extensions → Install from URL**.
2. Paste `https://github.com/aoleg/pruna-P-H3-Image-forge-neo`, click **Install** and restart the WebUI.
3. Check the console at startup for `[MiniMax H3] native backend enabled`.
4. Download one set of model files below.

### MiniMax-H3-Image

Files from [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image).

| Part | File | Size | Folder |
| --- | --- | ---: | --- |
| Checkpoint | `diffusion_models/minimax_h3_image_pruned_bf16`, `_int8_convrot` or `_w4a8` | 40.2 / 21.0 / 12.6 GB | `models/Stable-diffusion` |
| Text encoder | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot` (the MiniMax H3 text encoder; Merserk's INT4 and unsloth's GGUF work too) | 27.1 GB | `models/text_encoder` |
| VAE | `vae/minimax_h3_image_vae_fp16` | 1.2 GB | `models/VAE` |

The checkpoint is the H3 DiT without the audio layers, with six output heads, one per sampling step: 6 steps, CFG 1, no negative prompt. It works from about 4 MP to 20 MP and is best around 16 MP (4096×4096, 4736×3552, 5024×3360, 5472×3072, 6272×2688 and their portrait forms); ComfyUI's node defaults to 2048×2048.

The checkpoint asks for comfy-kitchen's INT8 attention, which ComfyUI then uses; in Forge Neo that is the `--use-ck-attention` launch flag. Without it the model runs on Forge's default attention, slower and with more VRAM at these sizes.

**Experimental:** P-H3-Image's much smaller text encoder (`p-h3-image_text_encoder_*`, 3.6 to 9.1 GB) is accepted in place of the 32B one. Pruna distilled it to stand in for the H3 encoder, and there are reports that MiniMax-H3-Image follows it, but the checkpoint was not trained with it. The console, the panel and the infotext mark such a generation as experimental. Sampling, sizes and the VAE stay MiniMax-H3-Image's.

Not used here: the `qwen3.5_9b_w4a8_gptq` prompt enhancer (the model expects a detailed English description of the whole picture, so write one or expand the prompt elsewhere), and the `nvfp4_awq` text encoder, which this extension still refuses because it encoded prompts wrongly in Forge Neo when last tried.

Its license is the MiniMax-H3-Image Community License, not MiniMax H3's: read it before commercial use.

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

| | `h3image` preset: MiniMax-H3-Image | `ph3` preset: P-H3-Image | `h3` preset: H3 fine-tunes |
| --- | --- | --- | --- |
| Sampler, schedule | Euler, Simple | Euler, **P-H3-Image** (Pruna's 8-step grid) | Euler, Simple |
| Steps, CFG | 6, 1 | 8, 1 | 30, 3 (the DMD LoRA: 8, 1) |
| Shift | not used (slider hidden) | 1 | 3 |
| Size | 4 to 20 MP, best about 16 MP | about 8 MP: 2912×2912, 3360×2496, 3872×2176 | about 1 MP |

MiniMax-H3-Image samples on the sigma grid stored in its checkpoint: every Forge schedule is mapped onto that grid, and Simple at 6 steps gives exactly the six sigmas its output heads were made for. Other step counts work (a step then blends the heads it spans), other schedules bend the spacing. The P-H3-Image schedule is that model's own sigma grid and ignores Shift; another step count resamples the same curve. Batch Size stays 1 (use Batch Count); Hires. fix and the Refiner are hidden while an H3 checkpoint is selected.

## Editing

**Settings → MiniMax H3 → [H3] Enable Reference** (on by default, like Forge's Klein and Krea 2 settings) decides what img2img does. On, it is an edit, with MiniMax-H3-Image or P-H3-Image:

- the input image is `<Picture 1>`, the picture to edit;
- turn on **ImageStitch Integrated** and add the references to its gallery: `<Picture 2>`, `<Picture 3>` and so on, 16 pictures in all for MiniMax-H3-Image, 8 for P-H3-Image;
- write the instruction and name the pictures ("put the hat from <Picture 2> on the person in <Picture 1>");
- Width and Height set the output. Resize mode and Denoising strength are not used: the model generates the whole picture from noise, conditioned on the pictures.

The two models take the pictures their own way, as their ComfyUI nodes do:

| | MiniMax-H3-Image | P-H3-Image |
| --- | --- | --- |
| Picture for the VAE | short edge brought into 1024 to 2048 | within the output area, never enlarged |
| Picture for the text encoder | short edge 1024 | within 1 MP |
| Prompt | used as written | a short instruction gets `Edit <Picture 1>. ` in front; one that already starts that way, or an edit brief from Pruna's upsampler, is used as it is |
| Good output size | 4 MP and up | about 4 MP (2048×2048, 2368×1760, 2720×1536) |

Every reference adds its own tokens to the sequence: sixteen large pictures are a much longer run than one.

Off, img2img is Forge's own with any H3 model: the input image is encoded, noised at Denoising strength and denoised on the profile's schedule. Other H3 models have no edit mode, so with the setting on img2img is refused for them.

## Known unknowns

- MiniMax-H3-Image with P-H3-Image's text encoder is an experiment on hearsay: nothing here has compared it with the 32B encoder.
- MiniMax-H3-Image has not been run on a GPU here. Its checkpoint and VAE match the extension's classes tensor for tensor, the head selection and sigma grid follow ComfyUI's code, and the offline tests pass; the first pictures are still the test.
- The working-memory reservation is an estimate (the console prints the measured figure on the first step of every generation); 16 MP is twice the tokens of anything tried so far.
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
- **[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)** and **[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)** by MiniMax
- **[Pruna AI](https://huggingface.co/PrunaAI/P-H3-Image-Comfy)**: P-H3-Image
- **[lodestones](https://huggingface.co/lodestones/chroma-h3)**: chroma-h3
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)**: reference implementation of the models
- **[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)**: the MiniMax-H3-Image and H3 model files
- **[Kijai](https://huggingface.co/Kijai)**: the INT8 video VAE and the W4A8 checkpoints
- **[Merserk](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)**: the INT4 text encoder
- **[unsloth](https://huggingface.co/unsloth/MiniMax-H3-GGUF)**: GGUF checkpoints

## License

AGPL-3.0, see [LICENSE](LICENSE). The model weights have their own terms: MiniMax-H3-Image is under the MiniMax-H3-Image Community License (separate authorization above a revenue threshold or for hosted services); P-H3-Image and MiniMax H3 are under the MiniMax H3 Community License, which excludes some regions and requires "Powered by MiniMax H3" on products built with them; the Qwen3-VL parts of P-H3-Image's text encoder are Apache 2.0.
