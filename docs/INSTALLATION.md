# Installation and troubleshooting

Version 0.12.2. The H3 architecture runs on Forge Neo's own loader, samplers and memory management; the extension installs no Python package. See the [README](../README.md) for the short version.

## Requirements

- **Forge Neo**, `neo` branch. The extension imports and registers against revision `286a244e` (3 November 2026); the native backend was last run on a GPU as the video extension at `97b26fb`. Python 3.13, Torch 2.13 or 2.14 with CUDA, comfy-kitchen 0.2.37 or later with its CUDA backend.
- **Disk:** MiniMax-H3-Image needs its checkpoint (12.6 to 40.2 GB), the H3 32B text encoder (12 to 27 GB) and a 1.2 GB VAE. About 20 GB for the smallest P-H3-Image set (w4a8 checkpoint, w4a8 text encoder, int8 VAE), 30 GB for the int8 set, 55 GB for bf16.
- **System RAM:** Forge loads the checkpoint, text encoder and VAE together. The video extension needed 32 GB of RAM with 16 GB of VRAM and the smaller files; P-H3-Image's files are about half the size of H3's.

## Install

1. In Forge Neo, open **Extensions → Install from URL**.
2. Paste `https://github.com/aoleg/pruna-P-H3-Image-forge-neo`, click **Install** and restart the WebUI.
3. Check the console at startup:
   - `[MiniMax H3] native backend enabled (torch ...)`: ready.
   - `[MiniMax H3] This Forge Neo is too old for the extension...`: the extension changed nothing and stays disabled. Update Forge Neo (`git pull` on the `neo` branch) and restart. The lines below the message list what is missing.

### Coming from minimax-h3-forge-neo

The video extension and this one register the same model family: keep one of them installed. The video VAE, the audio VAE, FFmpeg and the `H3 FFmpeg executable` setting are no longer used. The `h3` UI preset is gone since 0.10.0 (MiniMax H3 with its own 32B text encoder is not run as a still-image model); the presets are `h3image` and `ph3`.

## Model files

### MiniMax-H3-Image

From [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image).

| Part | File | Size | Folder |
| --- | --- | ---: | --- |
| Checkpoint | `diffusion_models/minimax_h3_image_pruned_bf16.safetensors` | 40.2 GB | `models/Stable-diffusion` |
| | `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` | 21.0 GB | |
| | `diffusion_models/minimax_h3_image_pruned_w4a8.safetensors` | 12.6 GB | |
| Text encoder | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.1 GB | `models/text_encoder` |
| VAE | `vae/minimax_h3_image_vae_fp16.safetensors` | 1.2 GB | `models/VAE` |

The text encoder is MiniMax H3's own, so a file you already have for H3 works, Merserk's INT4 and unsloth's GGUF included. Experimental: `p-h3-image_text_encoder_*` (P-H3-Image's 4B encoder with its text connector) is accepted in its place and marked as experimental in the console, the panel and the infotext; the recipe stays MiniMax-H3-Image's. Also experimental: `p-h3-image_vae_*` in place of `minimax_h3_image_vae_fp16`, with either text encoder; both VAEs share the H3 encoder and latents, and only the decoder differs. `qwen3vl_32b_minimax_h3_nvfp4_awq` is refused (see below). `qwen3.5_9b_w4a8_gptq` is the prompt enhancer and is not used.

The checkpoint is recognized by its tensors: the H3 DiT without the audio stream, with six stacked output heads. The sigma grid of the heads is read from the file's metadata; a file without it (a repack, a GGUF) gets the published grid. The VAE is recognized by its 2D decoder (24 blocks, 16 heads) and has to be selected with this checkpoint and no other.

### P-H3-Image

From [PrunaAI/P-H3-Image-Comfy](https://huggingface.co/PrunaAI/P-H3-Image-Comfy).

| Part | File | Size | Folder |
| --- | --- | ---: | --- |
| Checkpoint | `diffusion_models/p-h3-image_bf16.safetensors` | 40.5 GB | `models/Stable-diffusion` |
| | `diffusion_models/p-h3-image_int8_convrot.safetensors` | 21.2 GB | |
| | `diffusion_models/p-h3-image_w4a8.safetensors` | 13.4 GB | |
| Text encoder | `text_encoders/p-h3-image_text_encoder_bf16.safetensors` | 9.1 GB | `models/text_encoder` |
| | `text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors` | 5.1 GB | |
| | `text_encoders/p-h3-image_text_encoder_w4a8.safetensors` | 3.6 GB | |
| VAE | `vae/p-h3-image_vae_bf16.safetensors` | 5.6 GB | `models/VAE` |
| | `vae/p-h3-image_vae_int8_convrot.safetensors` | 3.1 GB | |
| LoRA for an H3 checkpoint | `loras/p-h3-image_lora_bf16.safetensors` | 2.0 GB | `models/Lora` |

The prompt upsampler files (`text_encoders/p-h3-image_prompt_upsampler_*`) are not used.

The checkpoint has the tensor layout of Comfy-Org's pruned FL2VA files, so any MiniMax H3 audio-video checkpoint (pruned, int8, w6a8, fp8, Kijai's W4A8, GGUF Q2_K to Q8_0) is accepted in its place with Pruna's LoRA. The extension knows P-H3-Image by its **text encoder** (Pruna's text connector) and requires the P-H3-Image VAE with it, whose decoder makes one frame per latent. The two parts must be selected together.

### Refused files

The NVFP4 AWQ text encoder is refused (it encodes prompts wrongly in Forge Neo). The video VAE and the audio VAE of MiniMax H3 are refused: still images use neither. MiniMax H3 with its own 32B text encoder is refused: it is the video model. llama.cpp GGUF text encoders, which keep the vision tower in a separate file, are refused; unsloth's file carries it inside.

Files are recognized by their tensor layout, not by their name.

## Selecting the components

1. Pick the UI preset: **h3image** for MiniMax-H3-Image (the **H3-Image LCM** sampler, Simple, 6 steps by default; MiniMax's reference is Euler, Simple, 6 steps), **ph3** for P-H3-Image. If you used the h3image preset before 0.12.0, Forge may have saved its old sampler, schedule and steps: set them under **Settings → Presets → H3IMAGE**. For Euler and Heun, the schedules **H3-Image 8-step**, **8-step softer** and **6-step** are named for the step count they were tuned at and resample to any other count; see the README for what each is. The preset sets sampler, schedule, steps, CFG and Shift; selecting an H3 checkpoint also re-applies the profile's sampler, schedule and CFG, whatever preset is active.
2. Select the checkpoint in the checkpoint selector.
3. Under **VAE / Text Encoder**, select the text encoder and the VAE, one of each. The **Components** part of the **MiniMax H3** panel lists what was recognized and which model it is.
4. Set Width and Height: multiples of 32, at least 64. MiniMax-H3-Image works from 4 to 20 MP and is best around 16 MP (4096×4096, 4736×3552, 5024×3360, 5472×3072, 6272×2688). P-H3-Image renders best at about 8 MP (2912×2912 and the sizes in its README) and edits at about 4 MP.
5. For a LoRA, add it to the prompt: `<lora:p-h3-image_lora_bf16:1>` puts P-H3-Image on a MiniMax H3 checkpoint. On a quantized checkpoint the extension keeps the INT8 matmuls and adds the LoRA as a low-rank term; set **Diffusion in Low Bits** to **Automatic (fp16 LoRA)** if Forge reports it cannot merge the LoRA.

## img2img

**Settings → MiniMax H3 → [H3] Enable Reference** is on by default: img2img is an edit, with MiniMax-H3-Image or P-H3-Image. Off, img2img is Forge's standard one with either model: the input image is encoded and noised at Denoising strength. The H3-Image schedules stop at sigma 0.42 or above, so with MiniMax-H3-Image even a low Denoising strength starts from a heavily noised picture.

With the setting on, the input image is `<Picture 1>`; **ImageStitch Integrated**'s gallery holds `<Picture 2>` onwards. The output is Width × Height; Resize mode and Denoising strength are not used, and the latent-upscale resize mode is refused.

- **MiniMax-H3-Image:** 16 pictures in all. Each goes to the VAE with its short edge brought into 1024 to 2048 and to the text encoder at a 1024 short edge; the prompt is used as written, each picture introduced to the model as `Picture N: `.
- **P-H3-Image:** 8 pictures in all, each within the output area for the VAE and within 1 MP for the text encoder. The instruction gets `Edit <Picture 1>. ` in front unless it already starts with `Edit <Picture` or `subject_definitions:`.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| The MiniMax H3 panel does not appear | Select an H3 or P-H3-Image checkpoint; the panel shows only for them. If the console says `H3 native controls were not found`, update Forge Neo and restart. |
| `Select the H3 text encoder / VAE...` | Select both components under **VAE / Text Encoder**, one of each. |
| `The MiniMax-H3-Image VAE goes with the MiniMax-H3-Image checkpoint` | Select the MiniMax-H3-Image parts together: its checkpoint, `qwen3vl_32b_minimax_h3_*` and `minimax_h3_image_vae_fp16` (or, experimentally, P-H3-Image's text encoder and VAE). |
| `... asks for comfy-kitchen's INT8 attention` (a note, not an error) | Start Forge Neo with `--use-ck-attention` to run MiniMax-H3-Image as ComfyUI does. |
| `MiniMax H3 PDD heads need the sampler's sigma schedule` | The sampler in use does not hand its schedule to the model. Use Euler. |
| `... is the H3 video VAE` | Still images use an image VAE: select `p-h3-image_vae_*` (P-H3-Image) or `minimax_h3_image_vae_fp16` (MiniMax-H3-Image). |
| `An H3 audio-video checkpoint makes still images with the P-H3-Image text encoder` | The selected checkpoint is MiniMax H3 or P-H3-Image with the 32B text encoder. Select `p-h3-image_text_encoder_*` and `p-h3-image_vae_*`, or select the MiniMax-H3-Image checkpoint. |
| Gritty skin, noise in flat areas, or a NaN error with MiniMax-H3-Image | The schedule evaluates the model at a low sigma. Use an **H3-Image** schedule. |
| `... is the H3 audio VAE: still images do not use it` | Deselect it. |
| `Unrecognized H3 component` | A selected module is not an H3 or P-H3-Image file. Deselect it. |
| `H3 width and height must be multiples of 32` | Use multiples of 32, at least 64. |
| `H3 generates one picture at a time` | Set Batch Size to 1 and use Batch Count. |
| `The img2img input image did not reach H3` | Set **Settings → VAE → VAE for Encoding** to **Full**. |
| The process is killed while loading or adding a LoRA | Not enough system RAM. Close other programs, restart Forge before changing the text encoder, and prefer the smaller files. |

Report other errors on the [issue tracker](https://github.com/aoleg/pruna-P-H3-Image-forge-neo/issues) with the console output.

## Local tests (developers)

```bash
python -m unittest discover -s tests -v
python -m ruff check .
```

The native backend tests need Torch and comfy-kitchen; Gradio is optional. They use the real tensor names and shapes of the published files (`tests/fixtures/h3_headers.json.gz`, no weights) and stand-ins for Forge Neo (`tests/forge_stubs.py`). They do not replace a GPU run.

## Source provenance

- ComfyUI `0228a9d` (10 November 2026): `comfy/ldm/minimax/` (`MiniMaxH3ImageModel`, `sampling.py`, the image VAEs), `comfy/text_encoders/minimax.py`, `comfy_extras/nodes_minimax_h3.py` (`MiniMaxH3ImageConditioning`, `PH3ImageConditioning`), `comfy/sd.py`; P-H3-Image was ported from `5f415e3`, the original video port followed `e9027f2`.
- Forge Neo `286a244e` (import and registration), `97b26fb` and `d70373e` (the video extension's GPU tests and source reading).
- PrunaAI/P-H3-Image-Comfy, `main` as of 4 November 2026, for the tensor layouts and recipes.
- Comfy-Org/MiniMax-H3-Image and MiniMaxAI/MiniMax-H3-Image, `main` as of 11 November 2026.
