"""Configuration loading: config/default.yaml deep-merged with config/local.yaml or $BTC5M_CONFIG."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """dict with attribute access and path resolution helpers."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) else v

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def load_config() -> Config:
    base = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    override_path = os.environ.get("BTC5M_CONFIG") or str(ROOT / "config" / "local.yaml")
    if Path(override_path).exists():
        base = _deep_merge(base, yaml.safe_load(Path(override_path).read_text(encoding="utf-8")) or {})
    if base.get("mode") != "paper":
        raise ValueError("Only mode=paper is supported. This system never trades.")
    return Config(base)


def ca_bundle() -> str | bool:
    """TLS verification target. Honors standard env vars; falls back to certifi/system."""
    for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        if os.environ.get(k) and Path(os.environ[k]).exists():
            return os.environ[k]
    if Path("/root/.ccr/ca-bundle.crt").exists():  # sandbox proxy bundle (dev environment)
        return "/root/.ccr/ca-bundle.crt"
    return True
