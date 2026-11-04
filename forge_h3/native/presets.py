"""The "h3" and "ph3" UI presets, added at runtime: PresetArch gains a member per profile and the preset's options
are registered through Forge Neo's own presets.register, so they match every other preset (as in the Qwen-Image 2.1
extension). The P-H3-Image schedule joins Forge's scheduler list the same way.
"""

from enum import Enum

from modules_forge import presets

from ..profiles import H3, PH3_IMAGE, PH3_IMAGE_SCHEDULER, ph3_image_sigmas

PROFILES = (H3, PH3_IMAGE)
PRESETS = tuple(profile.name for profile in PROFILES)


def _add_enum_member(enum_cls: type[Enum], name: str) -> Enum:
    if name in enum_cls.__members__:
        return enum_cls[name]

    value = max(m.value for m in enum_cls) + 1
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)

    # EnumType.__setattr__ refuses new members, so go through type
    type.__setattr__(enum_cls, name, member)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member
    if hasattr(enum_cls, "_hashable_values_"):
        enum_cls._hashable_values_.append(value)
    return member


def _is_preset_option(key: str) -> bool:
    return any(key.startswith(f"{name}_") or key.endswith(f"_{name}") for name in PRESETS)


def register() -> None:
    for profile in PROFILES:
        arch = _add_enum_member(presets.PresetArch, profile.name)
        presets.SAMPLERS[arch] = profile.sampler
        presets.SCHEDULERS[arch] = profile.scheduler
        presets.STEPS[arch] = profile.steps
        presets.CFG[arch] = profile.cfg
        presets.SHIFT[arch] = profile.shift

    from modules import shared

    templates: dict = {}
    presets.register(templates)
    for key, info in templates.items():
        if _is_preset_option(key) and key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)


def register_scheduler() -> None:
    """Pruna's 8-step grid as a Forge schedule, before the UI lists the schedules."""
    from modules import sd_schedulers

    if any(s.label == PH3_IMAGE_SCHEDULER for s in sd_schedulers.all_schedulers):
        return
    scheduler = sd_schedulers.Scheduler("ph3_image", PH3_IMAGE_SCHEDULER, ph3_image_sigmas)
    sd_schedulers.all_schedulers.append(scheduler)
    if scheduler.label not in getattr(sd_schedulers.shared.opts, "hide_schedulers", ()):
        sd_schedulers.schedulers.append(scheduler)
        sd_schedulers.schedulers_map[scheduler.name] = scheduler
        sd_schedulers.schedulers_map[scheduler.label] = scheduler
