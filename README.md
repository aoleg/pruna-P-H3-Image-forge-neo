# MiniMax-H3-Image for Forge Neo

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
![Status](https://img.shields.io/badge/Status-MiniMax--H3--Image_tested-brightgreen)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

An extension for [Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) that runs MiniMax's [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) for text-to-image and editing, with the checkpoint selector, presets and Generate button you already know. It also runs Pruna's [P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy), which this extension was first written for, and accepts P-H3-Image's small text encoder in place of MiniMax's 32B one.

> [!Note]
> **A personal note and recommendation, after testing both models on an RTX 5090.** MiniMax-H3-Image supersedes P-H3-Image. With the same prompt, seeds and text encoder, MiniMax-H3-Image wrote the requested numbers correctly in every picture, followed the described poses more closely and gave less fine grain. P-H3-Image dropped or misspelled a number in 5 of 16 pictures and added more garbled signs.
>
> **The part of P-H3-Image that is still worth having is its text encoder.** At 5 GB in INT8 against 27 GB, it stands in for the 32B H3 encoder, and MiniMax-H3-Image follows it well. All my MiniMax-H3-Image tests used it. It is marked experimental because MiniMax did not train the checkpoint with it.
>
> P-H3-Image stays supported. It is still a good **edit-and-upscale** model: in edit mode it does 8 MP and more without running out of memory, in 8 steps, in seconds.
>
> See [Settings](#settings) for the MiniMax-H3-Image schedules I use instead of MiniMax's own. **Still untested:** MiniMax-H3-Image's edit mode and 16 MP, and the stock MiniMax H3 checkpoint with Pruna's P-H3-Image LoRA.
>
> The repository keeps its old name for now.
>
> This is a fork of [minimax-h3-forge-neo](https://github.com/eduardoabreu81/minimax-h3-forge-neo) with everything video and audio removed: no clips, no sound, no FFmpeg, no keyframes, guides or motion control. The extension needs an up-to-date **Forge Neo** (the `neo` branch, with comfy-kitchen 0.2.37 or later). On an older version it stays disabled and tells you so in the console; Forge Neo itself keeps working as usual.

## What it does

- **MiniMax-H3-Image**, MiniMax's image model: text-to-image and edits with up to 16 pictures, 4 to 20 MP, in 6 to 8 steps with no CFG, on schedules made for it. Select its checkpoint, the H3 text encoder and its VAE, and pick the `h3image` UI preset.
- **P-H3-Image's text encoder with MiniMax-H3-Image** (experimental): 3.6 to 9.1 GB in place of 27 GB, selected like any other text encoder.
- **P-H3-Image text-to-image** at 8 MP in 8 steps, no CFG, with Pruna's own schedule: select the P-H3-Image checkpoint, text encoder and VAE, pick the `ph3` UI preset and generate.
- **P-H3-Image edits** in img2img: the input image is `<Picture 1>`, the ImageStitch Integrated gallery holds up to 7 references, and the prompt names them.
- **Forge's own machinery**: its loader, samplers, memory management, quantized formats (int8, w4a8, w6a8, fp8, GGUF) and LoRA handling. Nothing is installed into Forge's Python environment.

A generation is one latent frame, decoded as a picture. P-H3-Image's checkpoint has the audio-video layout of MiniMax H3, so it also gets the two-step audio stub it expects, sampled with the picture and never decoded; MiniMax-H3-Image has no audio stream at all. The video VAE and the audio VAE of MiniMax H3 are not used, and are refused if selected.

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

The checkpoint is the H3 DiT without the audio layers, with six output heads, one per step of MiniMax's own 6-step schedule: CFG 1, no negative prompt. It works from about 4 MP to 20 MP and is best around 16 MP (4096×4096, 4736×3552, 5024×3360, 5472×3072, 6272×2688 and their portrait forms); ComfyUI's node defaults to 2048×2048.

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

### MiniMax H3 itself

Up to 0.9.x an `h3` preset ran a MiniMax H3 video checkpoint with its own 32B text encoder and video VAE as a still-image model, for fine-tune LoRAs such as chroma-h3. It was never tested on a GPU, and MiniMax-H3-Image makes it redundant, so 0.10.0 removes it together with the video VAE. An H3 audio-video checkpoint still runs behind the P-H3-Image text encoder and VAE, as described above; with the 32B text encoder it is refused with a message that says what to select.

## Settings

Selecting a checkpoint applies the matching profile's sampler, schedule and CFG, and the UI presets carry the rest:

| | `h3image` preset: MiniMax-H3-Image | `ph3` preset: P-H3-Image |
| --- | --- | --- |
| Sampler, schedule | Euler, **H3-Image 8-step** | Euler, **P-H3-Image** (Pruna's 8-step grid) |
| Steps, CFG | 8, 1 | 8, 1 |
| Shift | not used (slider hidden) | 1 |
| Size | 4 to 20 MP, best about 16 MP | about 8 MP: 2912×2912, 3360×2496, 3872×2176 |

### The H3-Image schedules

MiniMax-H3-Image has three schedules of its own, each named for the step count it was tuned at:

| Schedule | Steps | What it is | Result |
| --- | ---: | --- | --- |
| **H3-Image 8-step** | 8 | a shifted flow curve from sigma 1 to 0.42, then 0 | the default, and the crispest |
| **H3-Image 8-step softer** | 8 | the same 0.42 end with a slower start: each sigma halfway between the 8-step curve and Pruna's grid | smoother skin and fewer dark specks, faces still defined |
| **H3-Image 6-step** | 6 | MiniMax's own six-point grid with its last point lifted from 0.232 to 0.42 | clean, three quarters of the time |

The first sigmas decide how smooth the picture is, and the last one barely matters. From crisp to smooth, the order is H3-Image 8-step, H3-Image 8-step softer, then the P-H3-Image schedule. The three give different compositions on the same seed.

**Heun** at 8 steps (15 model evaluations against Euler's 8) gives sharper faces than Euler on all three, with more small dark specks that look like dust on a scanned slide. Euler stays the default because it is twice as fast.

Each runs at any other step count too: the list is resampled between its two fixed ends, so 10 or 12 steps of the 8-step schedule sample the same curve more finely and never go below its last sigma. More steps are different, not better: 12 steps gave more skin texture and blemishes than 8. The step count also changes the composition, because the first sigmas choose it.

These schedules are a personal pick from same-seed tests on an RTX 5090 (Euler, CFG 1, 6.4 MP, the INT8 checkpoint with P-H3-Image's text encoder). What decides the picture is the lowest sigma the model is evaluated at:

| Last sigma before 0 | Schedules that end there | Picture |
| --- | --- | --- |
| 0.76 and up | Krea 2 Reference, Flux2 (at this size) | under-denoised, down to a blur |
| 0.53 | P-H3-Image, MiniMax's grid without its last point (5 steps) | clean, smoother, a little more grain |
| 0.27 to 0.42 | the three H3-Image schedules, Linear Quadratic, Qwen Image 2.1 Turbo | clean and crisp, nearly the same picture across the range |
| 0.232 | Simple at 6 steps, which is **MiniMax's reference recipe** | speckle |
| below 0.2 | Simple, SGM Uniform, Beta, Bong Tangent and Normal at 8 steps | gritty: leathery skin, noise in flat areas |

Every Forge schedule is mapped onto the sigma grid stored in the checkpoint, and Simple at 6 steps gives exactly the six sigmas its output heads were made for, so the reference recipe stays one schedule and step count away. Why its last step fails here is not known: the port matches MiniMax's pipeline piece by piece, and the cause may be the model, its INT8 file or the port. The P-H3-Image schedule is Pruna's own sigma grid and ignores Shift; another step count resamples the same curve. Batch Size stays 1 (use Batch Count); Hires. fix and the Refiner are hidden while an H3 checkpoint is selected.

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

Off, img2img is Forge's own with either model: the input image is encoded, noised at Denoising strength and denoised on the profile's schedule. With MiniMax-H3-Image the schedules stop at sigma 0.42 or above, so even a low Denoising strength starts from a heavily noised picture: gentle refinement is not what this mode does.

## Known unknowns

- MiniMax-H3-Image with P-H3-Image's text encoder is an experiment: one same-seed comparison gave a similar composition with different small details, and all the schedule tests above used it.
- MiniMax-H3-Image runs, with either text encoder. Its reference recipe (Euler, Simple, 6 steps) leaves speckle in its last step, at sigma 0.232, which the H3-Image schedules avoid. The output heads and the time-embedding table were checked numerically against MiniMax's own weights and the sampler against its pipeline, so the open comparisons are the same prompt and seed in ComfyUI, and a checkpoint that is not INT8. It has not been tried at 16 MP or for edits.
- The working-memory reservation is an estimate (the console prints the measured figure on the first step of every generation); 16 MP is twice the tokens of anything tried so far.
- The stock MiniMax H3 checkpoint with Pruna's LoRA has not been run yet.
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
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)**: reference implementation of the models
- **[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)**: the MiniMax-H3-Image and H3 model files
- **[Kijai](https://huggingface.co/Kijai)**: the W4A8 checkpoints
- **[Merserk](https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot)**: the INT4 text encoder
- **[unsloth](https://huggingface.co/unsloth/MiniMax-H3-GGUF)**: GGUF checkpoints

## License

AGPL-3.0, see [LICENSE](LICENSE). The model weights have their own terms: MiniMax-H3-Image is under the MiniMax-H3-Image Community License (separate authorization above a revenue threshold or for hosted services); P-H3-Image and MiniMax H3 are under the MiniMax H3 Community License, which excludes some regions and requires "Powered by MiniMax H3" on products built with them; the Qwen3-VL parts of P-H3-Image's text encoder are Apache 2.0.
