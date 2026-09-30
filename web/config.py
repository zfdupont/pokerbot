"""Runtime configuration for the poker web service (env driven)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TOML = os.path.join(_REPO_ROOT, "sixmax", "configs", "default.toml")
_DEFAULT_ORIGINS = ("https://zfdupont.com",)


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    return default if raw is None or raw == "" else int(raw)


@dataclass(frozen=True)
class Config:
    checkpoint: str
    config_toml: str
    stack: int
    small_blind: int
    session_ttl_s: int
    cors_origins: tuple[str, ...]

    @property
    def big_blind(self) -> int:
        return self.small_blind * 2


def load_config(env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    origins = env.get("POKERBOT_CORS_ORIGINS")
    cors = (_DEFAULT_ORIGINS if not origins
            else tuple(o.strip() for o in origins.split(",") if o.strip()))
    return Config(
        checkpoint=env.get("POKERBOT_CHECKPOINT", ""),
        config_toml=env.get("POKERBOT_CONFIG_TOML", _DEFAULT_TOML),
        stack=_int(env, "POKERBOT_STACK", 200),
        small_blind=_int(env, "POKERBOT_SMALL_BLIND", 1),
        session_ttl_s=_int(env, "POKERBOT_SESSION_TTL", 3600),
        cors_origins=cors,
    )
