"""Task registry."""

from __future__ import annotations

import importlib

_REGISTRY = {
    "e1_simpson": ("witnesses.tasks.e1_simpson", "E1Simpson"),
    "e2_dags": ("witnesses.tasks.e2_dags", "E2Dags"),
    "e2c_regimes": ("witnesses.tasks.e2c_regimes", "E2cRegimes"),
    "e3_twin": ("witnesses.tasks.e3_twin", "E3Twin"),
    "e4_identify": ("witnesses.tasks.e4_identify", "E4Identify"),
    "e5a_recurrence": ("witnesses.tasks.e5_recurrence", "E5aRestartClock"),
    "e5b_recurrence": ("witnesses.tasks.e5_recurrence", "E5bNoiseCF"),
}


def get_task(name: str, **kwargs):
    mod, cls = _REGISTRY[name]
    return getattr(importlib.import_module(mod), cls)(**kwargs)
