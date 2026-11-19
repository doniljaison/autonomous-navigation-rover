"""SLAM parameters: a plain namespace built from config, with overrides (replay tuning)."""

from types import SimpleNamespace


def from_config(**overrides) -> SimpleNamespace:
    from phase3_onboard import config
    d = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    for k, v in overrides.items():
        if k not in d:
            raise KeyError(f"unknown parameter {k}")
        d[k] = v
    return SimpleNamespace(**d)
