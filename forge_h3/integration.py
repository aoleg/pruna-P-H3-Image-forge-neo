"""H3 inside Forge's own txt2img flow, through script callbacks.

Forge loads the model (native/patches.py registers it), samples and decodes as usual. The callbacks here check the
request, give the sampler the packed video+audio noise of one still image, and let Forge save the picture.
"""

from pathlib import Path

from . import pictures
from .contracts import GenerationRequest, H3Error, set_pending_error
from .models import PH3_IMAGE, inspect_model, resolve_components

# StableDiffusionProcessing.distilled_cfg_scale when a request does not set it
API_DEFAULT_DISTILLED_CFG = 3.5
# img2img Resize mode "Just resize (latent upscale)": it would interpolate the placeholder latent
LATENT_UPSCALE = 3


def checkpoint_info(value):
    # the same lookup Forge applies to a checkpoint override: alias, then title substring, with or without the hash
    from modules import sd_models
    return sd_models.get_closet_checkpoint_match(value)


def select_h3(p):
    """The selected checkpoint when it is an H3 diffusion model; read from its header, before Forge loads it."""
    from modules import shared
    overrides = getattr(p, "override_settings", {})
    info = checkpoint_info(overrides.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint))
    if info is None:
        return None
    item = inspect_model(info.filename)
    return info if item is not None and item.role == "dit" else None


def module_paths(values):
    from modules_forge import main_entry
    result = []
    for value in values or []:
        # Forge Neo keys its list by file name; with "Show filenames without folder" off the dropdown sends the path
        # under models/VAE or models/text_encoder, which Forge itself reduces to the file name (modules_change)
        path = main_entry.module_list.get(value) or main_entry.module_list.get(Path(value).name, value)
        if not Path(path).is_file():
            raise H3Error(f"Selected H3 component is unavailable: {Path(value).name}. Refresh the model list.")
        result.append(str(Path(path).resolve()))
    return result


def validate_processing(p):
    if getattr(p, "batch_size", 1) != 1:
        raise H3Error("H3 generates one picture at a time: set Batch Size to 1 and use Batch Count.")
    if getattr(p, "enable_hr", False) or getattr(p, "txt2img_upscale", False):
        raise H3Error("Disable Hires. fix for H3 generation.")
    if getattr(p, "restore_faces", False):
        raise H3Error("Disable Restore faces for H3 generation.")
    if getattr(p, "image_mask", None) is not None:
        raise H3Error("H3 inpainting is not implemented.")
    if (getattr(p, "subseed_strength", 0) > 0 or getattr(p, "seed_resize_from_w", -1) > 0
            or getattr(p, "seed_resize_from_h", -1) > 0):
        raise H3Error("Disable variation seed and seed resize for H3.")
    if getattr(p, "script_args", ()) and p.script_args[0] not in (0, None, "None"):
        raise H3Error("Select Script: None for H3 generation. Script combinations are not validated yet.")


def before_process(p):
    """Before Forge loads the model: check the request and set up the H3 generation."""
    p.h3_request = None
    set_pending_error(None)
    try:
        _before_process(p)
    except H3Error as error:
        set_pending_error(error)
        raise


def script_before_process(p):
    """before_process as Forge's script runner calls it: a rejected request prints one line instead of the traceback
    Forge logs for any exception; the error stays pending and stops the generation when the model is first called."""
    try:
        before_process(p)
    except H3Error as error:
        print(f"[MiniMax H3] {error}")


def validate_img2img(p, components):
    if components.text_encoder.variant != PH3_IMAGE:
        raise H3Error(f"{components.profile.label} has no edit mode: img2img needs P-H3-Image (its text encoder and "
                      "VAE). Use txt2img.")
    if not getattr(p, "init_images", None):
        raise H3Error("Add an input image for P-H3-Image in img2img: it becomes <Picture 1>, the picture to edit.")
    if getattr(p, "resize_mode", 0) == LATENT_UPSCALE:
        raise H3Error("P-H3-Image does not take the latent upscale resize mode. Choose another Resize mode.")


_precision_noted = False


def note_precision_flags():
    """Once per session: Forge flags that run below the precision of the models' ComfyUI reference, which an
    8-step distilled model shows in its fine detail."""
    global _precision_noted
    if _precision_noted:
        return
    _precision_noted = True
    try:
        from backend.args import args
    except ImportError:
        return
    notes = []
    if getattr(args, "use_ck_attention", False):
        notes.append("--use-ck-attention runs INT8 attention in the DiT and the VAE decoder")
    if getattr(args, "fast_fp16", False):
        notes.append("--fast-fp16 accumulates fp16 matmuls in fp16 (the VAE decoder, a fp16 text encoder)")
    if notes:
        print("[MiniMax H3] note: " + "; ".join(notes) + ". The reference runs full precision there: compare without "
              "the flag if small details come out garbled.")


def _before_process(p):
    p.h3_references = []
    info = select_h3(p)
    if info is None:
        return
    from modules import processing, shared
    validate_processing(p)
    note_precision_flags()
    overrides = getattr(p, "override_settings", {})
    components = resolve_components(info.filename, module_paths(overrides.get("forge_additional_modules", shared.opts.forge_additional_modules)))
    refs = []
    if isinstance(p, processing.StableDiffusionProcessingImg2Img):
        validate_img2img(p, components)
        # the original pictures, each within the output area for the VAE (ComfyUI PH3ImageConditioning); the text
        # encoder's 1 MP copies are made from these by the engine
        refs = [pictures.prepare(image, p.width * p.height) for image in pictures.collect(p)]
        # the model generates the whole picture from noise, conditioned on <Picture 1>: the input image is never noised
        p.denoising_strength = 1.0
        p.prompt = pictures.edit_prompt(p.prompt)
    request = GenerationRequest(width=p.width, height=p.height, references=len(refs))
    p.h3_request = request
    p.h3_components = components
    p.h3_references = refs
    edit = f", editing <Picture 1> with {request.references} pictures" if refs else ""
    print(f"[MiniMax H3] {components.profile.label}: still image at {request.width}x{request.height}{edit}")


def process(p):
    request = getattr(p, "h3_request", None)
    if request is None:
        return
    if not getattr(p.sd_model, "is_h3", False):
        raise H3Error("Forge did not load the H3 model. Check the VAE / Text Encoder selection and the console.")
    from modules import shared

    engine = p.sd_model
    refs = getattr(p, "h3_references", [])
    # Forge caches the conditioning by prompt, which knows nothing of the pictures before it
    if refs or engine.condition_images():
        p.clear_prompt_cache()
    engine.set_references(refs)
    if request.references:
        p.extra_generation_params["H3 Edit pictures"] = request.references

    profile = p.h3_components.profile
    p.extra_generation_params["H3 Model"] = profile.label
    if getattr(shared.opts, "forge_preset", None) != profile.name:
        # outside the profile's preset the slider is another model's Shift or Distilled CFG; keep the profile's own
        # shift (and infotext)
        p.distilled_cfg_scale = profile.shift
    elif getattr(p, "is_api", False) and p.distilled_cfg_scale == API_DEFAULT_DISTILLED_CFG:
        # an API request that leaves distilled_cfg_scale out gets Forge's Flux default; use the preset's Shift
        p.distilled_cfg_scale = getattr(shared.opts, f"{profile.name}_t2i_dcfg", profile.shift)


def before_sampling(p, noise):
    from .native import patches
    request = getattr(p, "h3_request", None)
    if request is None:
        return
    from modules import rng
    shape = p.sd_model.prepare(request.width, request.height, int(p.seeds[0]))
    # Forge made p.rng for an image latent; the samplers that add noise on the way (ancestral, SDE, res_multistep)
    # draw from it too, so it has to give the packed shape
    p.rng = rng.ImageRNG(shape, p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)
    p.modified_noise = p.rng.next().to(device=noise.device, dtype=noise.dtype)
    if getattr(p, "init_latent", None) is not None:
        # img2img samples from init_latent at full denoise: the packed start is pure noise
        import torch
        p.init_latent = torch.zeros_like(p.modified_noise)
    patches.begin_sampling()


def after_sampling(p):
    from .native import patches
    if getattr(p, "h3_request", None) is not None:
        patches.end_sampling()


def postprocess(p, processed):
    """After Forge's own outputs: drop the decoded picture."""
    if getattr(p, "h3_request", None) is None:
        return
    from .native import patches
    patches.end_sampling()
    p.sd_model.release_generation()
