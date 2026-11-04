"""Small H3 accordion and native-control events for Forge's Gradio interface."""

import html
import logging

import gradio as gr

from .contracts import H3Error
from .integration import checkpoint_info
from .models import ROLE_LABELS, inspect_model

COMPONENTS = {}
NATIVE_CONTROLS = ("sampling", "scheduler", "cfg_scale")
# Forge's InputAccordions that do nothing for H3, by elem_id after the tab: Hires. fix, and the Refiner when
# Settings shows it (Forge Neo's processing script names it just "enable"); hidden and turned off while an H3
# checkpoint is selected, shown again after
HIDDEN_ACCORDIONS = ("hr", "enable")
PANELS = []
logger = logging.getLogger("forge_h3")


def _in_blocks():
    # Forge also creates components outside the UI build (on page load); events can only be bound inside it
    return gr.context.Context.root_block is not None


def capture(component, **kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id:
        COMPONENTS[elem_id] = component
        if elem_id in ("setting_sd_model_checkpoint", "setting_sd_modules", "forge_ui_preset"):
            bind_all(elem_id)


class Panel:
    def __init__(self, is_img2img):
        self.tab = "img2img" if is_img2img else "txt2img"
        self.is_img2img = is_img2img
        self.saved = gr.State({"active": False})
        with gr.Accordion("MiniMax H3", open=False, visible=False, elem_id=f"{self.tab}_h3_panel") as self.accordion:
            self.status = gr.Markdown("Select the H3 text encoder and VAE in VAE / Text Encoder.")
            with gr.Accordion("Components", open=False):
                self.summary = gr.Markdown("")
        self.bound = False
        self.attempts = []
        PANELS.append(self)

    @property
    def inputs(self):
        # the panel has no generation inputs of its own
        return []

    @property
    def needed(self):
        ids = [f"{self.tab}_{name}" for name in NATIVE_CONTROLS]
        return ids + ["setting_sd_model_checkpoint", "setting_sd_modules"]

    def bind(self, trigger="?"):
        if self.bound:
            return
        if not _in_blocks():
            self.attempts.append(f"{trigger}: outside Blocks")
            return
        ids = [f"{self.tab}_{name}" for name in NATIVE_CONTROLS]
        needed = self.needed
        if any(name not in COMPONENTS for name in needed):
            self.attempts.append(f"{trigger}: waiting for {', '.join(n for n in needed if n not in COMPONENTS)}")
            logger.debug("Waiting for H3 native controls for %s.", self.tab)
            return
        native = [COMPONENTS[name] for name in ids]
        checkpoint = COMPONENTS["setting_sd_model_checkpoint"]
        modules = COMPONENTS["setting_sd_modules"]
        preset = COMPONENTS.get("forge_ui_preset")
        defaults = [{key: getattr(c, key) for key in ("minimum", "maximum", "step", "label", "visible", "choices", "interactive")
                     if hasattr(c, key)} for c in native]
        preset_input = [preset] if preset is not None else []
        hidden = [(COMPONENTS[key], COMPONENTS[f"{key}-checkbox"]) for key in (f"{self.tab}_{name}" for name in HIDDEN_ACCORDIONS)
                  if key in COMPONENTS and f"{key}-checkbox" in COMPONENTS]
        inputs = [checkpoint, modules, self.saved] + native + preset_input
        outputs = ([self.accordion, self.status, self.summary, self.saved] + native + preset_input
                   + [component for pair in hidden for component in pair])

        def update(value, module_values, saved, *values):
            values = values[:len(native)]
            info = checkpoint_info(value)
            error = ""
            active = False
            if info:
                try:
                    item = inspect_model(info.filename)
                    active = item is not None and item.role == "dit"
                except H3Error as exc:
                    error = str(exc)
            saved = dict(saved or {"active": False})
            entering = active and not saved.get("active")
            leaving = not active and saved.get("active")
            if entering:
                saved["native"] = [dict(config, value=value) for config, value in zip(defaults, values)]
            updates = [gr.update() for _ in native]
            if entering:
                sampler, scheduler, cfg = preset_settings(self.tab)
                updates[0] = gr.update(value=sampler)
                updates[1] = gr.update(value=scheduler)
                updates[2] = gr.update(value=cfg) if cfg is not None else gr.update()
            elif leaving:
                updates = [gr.update(**config) for config in saved.get("native", [])]
                saved.pop("native", None)
            saved["active"] = active
            summary = ""
            if active:
                try:
                    from .integration import module_paths
                    from .models import resolve_components
                    components = resolve_components(info.filename, module_paths(module_values))
                    summary = "  \n".join(f"**{ROLE_LABELS[m.role].title()}:** {html.escape(m.path.name)} ({m.quantization})"
                                          for m in components.models)
                    error = ""
                except H3Error as exc:
                    error = str(exc)
            status = html.escape(error) if error else "H3 makes one still image per generation."
            return ([gr.update(visible=active), status, summary, saved] + updates
                    + ([gr.update()] if preset is not None else []) + hidden_updates(len(hidden), active, leaving))

        for event in (checkpoint.change, modules.change):
            event(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        gr.context.Context.root_block.load(update, inputs=inputs, outputs=outputs, queue=False, show_progress=False)
        self.bound = True
        logger.info("H3 native controls connected for %s.", self.tab)


def preset_settings(tab):
    """The sampler, schedule and CFG saved for the h3 preset in Settings, which start from the reference workflows'
    (every Forge sampler and schedule works); a CFG of 0 there keeps the current one (None), as Forge's presets do."""
    kind = "i2i" if tab == "img2img" else "t2i"
    try:
        from modules import shared
        opts = shared.opts
    except (ImportError, AttributeError):
        opts = None
    sampler = getattr(opts, f"h3_{kind}_sampler", None) or "Res Multistep"
    scheduler = getattr(opts, f"h3_{kind}_scheduler", None) or "Simple"
    cfg = getattr(opts, f"h3_{kind}_cfg", None)
    if cfg is None:
        cfg = 1.0
    return sampler, scheduler, cfg if cfg > 0 else None


def hidden_updates(count, active, leaving):
    """For each hidden accordion: the accordion, then its checkbox (the value Forge reads)."""
    if active:
        return [gr.update(visible=False), gr.update(value=False)] * count
    if leaving:
        return [gr.update(visible=True), gr.update()] * count
    return [gr.update(), gr.update()] * count


def bind_all(trigger="ui_tabs"):
    for panel in PANELS:
        panel.bind(trigger)
    return []


def check_bindings(*_args):
    # Forge also builds panel instances it never renders; a tab is fine as long as one of its panels is bound
    for tab in dict.fromkeys(panel.tab for panel in PANELS):
        panels = [panel for panel in PANELS if panel.tab == tab]
        if any(panel.bound for panel in panels):
            continue
        missing = sorted({name for panel in panels for name in panel.needed if name not in COMPONENTS})
        attempts = [attempt for panel in panels for attempt in panel.attempts]
        logger.error("H3 native controls were not found for %s (missing: %s; attempts: %s); restart Forge after "
                     "updating it.", tab, ", ".join(missing) or "none", "; ".join(attempts) or "none")


def reset():
    COMPONENTS.clear()
    PANELS.clear()
