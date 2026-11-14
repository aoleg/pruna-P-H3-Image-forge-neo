"""Forge Neo discovers this always-on script when the extension is installed."""

import sys
import traceback
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import script_callbacks, scripts, shared

from forge_h3 import integration, sampler, ui
from forge_h3.native import compat

ENABLED = False

_problems = compat.check()

if _problems:
    compat.report(_problems)
else:
    try:
        from forge_h3.native import patches

        patches.apply()
        ENABLED = True
    except Exception:
        print("[MiniMax H3] failed to enable, Forge Neo is unchanged:")
        traceback.print_exc()


class Script(scripts.Script):
    sorting_priority = 10
    create_group = False

    def title(self):
        return "MiniMax H3"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if ENABLED else False

    def ui(self, is_img2img):
        self.panel = ui.Panel(is_img2img)
        return self.panel.inputs

    def before_process(self, p, *args):
        integration.script_before_process(p)

    def process(self, p, *args):
        integration.process(p)

    def process_before_every_sampling(self, p, *args, **kwargs):
        integration.before_sampling(p, kwargs["noise"])

    def post_sample(self, p, ps, *args):
        integration.after_sampling(p)

    def postprocess(self, p, processed, *args):
        integration.postprocess(p, processed)


def settings():
    # as Forge Neo's klein_do_reference, krea2_do_reference and the like
    section = ("forge_h3", "MiniMax H3")
    shared.opts.add_option(integration.REFERENCE_SETTING,
                           shared.OptionInfo(True, "[H3] Enable Reference", section=section)
                           .info("enable Edit ; disable img2img")
                           .info("pin to <b>Quicksettings</b> is recommended if changed often")
                           .info("Edit needs P-H3-Image"))
    import gradio as gr

    shared.opts.add_option(integration.OUTPUT_SCALE_SETTING,
                           shared.OptionInfo("1", "[H3] Output scale", gr.Radio,
                                             {"choices": list(integration.OUTPUT_SCALES)}, section=section)
                           .info("shrinks the decoded picture with an exact area filter, for less pixel-level "
                                 "grit; from 16 MP, 3/4 gives 9 MP, 2/3 gives 7 MP, 1/2 gives 4 MP"))
    shared.opts.add_option(integration.RESOLUTION_SHIFT_SETTING,
                           shared.OptionInfo(0.0, "[H3] Resolution shift", gr.Slider,
                                             {"minimum": 0.0, "maximum": 1.0, "step": 0.05}, section=section)
                           .info("experimental, MiniMax-H3-Image only: moves its sigma grid with the picture size, "
                                 "lower below 16 MP and higher above, as Krea 2 and Flux move their shift; 0 keeps "
                                 "MiniMax's fixed grid, 0.5 is SD3's rule (at 4 MP the shift halves); no gain on the "
                                 "stock checkpoint in tests, more grit with Euler at 4 MP"))
    low, high = sampler.S_NOISE_RANGE
    noise_slider = {"minimum": low, "maximum": high, "step": 0.1}
    shared.opts.add_option(sampler.S_NOISE_START_SETTING,
                           shared.OptionInfo(sampler.S_NOISE_START, "[H3] LCM noise, first step", gr.Slider, noise_slider,
                                             section=section, infotext=sampler.S_NOISE_START_INFOTEXT)
                           .info("H3-Image LCM: the fresh noise added after the first step, as a multiple of the step's "
                                 "sigma; above 1 adds fine detail, below 1 smooths"))
    shared.opts.add_option(sampler.S_NOISE_END_SETTING,
                           shared.OptionInfo(sampler.S_NOISE_END, "[H3] LCM noise, last step", gr.Slider, noise_slider,
                                             section=section, infotext=sampler.S_NOISE_END_INFOTEXT)
                           .info("H3-Image LCM: the same at the last step that adds noise; the steps between run "
                                 "linearly from first to last. Set both alike for a flat s_noise"))


script_callbacks.on_ui_settings(settings)
# a rebuilt UI starts with no panels: the ones of a previous build can never bind again
script_callbacks.on_before_ui(ui.reset)
script_callbacks.on_after_component(ui.capture)
script_callbacks.on_ui_tabs(ui.bind_all)
script_callbacks.on_app_started(ui.check_bindings)
script_callbacks.on_script_unloaded(ui.reset)
