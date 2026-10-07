"""Shared pieces of the evaluation scripts: paths, env loading, run config."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
FORK_DIR = EVAL_DIR.parent
SERVER_DIR = FORK_DIR.parent / "server"
RUNS_DIR = EVAL_DIR / "runs"
RESULTS_DIR = EVAL_DIR / "results"

# Env names the pipeline and judge read. Only these are taken from --env.
_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "FRED_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "TIINGO_API_KEY",
    "LLM_PROVIDER",
    "DEEP_THINK_LLM",
    "QUICK_THINK_LLM",
    "PRICE_VENDOR",
)


def load_env(path: str | None) -> dict[str, str]:
    """Read the listed keys from an env file into os.environ (values are
    never printed). Returns the non-secret settings for the record."""
    if not path:
        return {}
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key in _ENV_KEYS and value:
            values[key] = value
    os.environ.update(values)
    return {k: v for k, v in values.items() if not k.endswith("_KEY")}


def golden_set() -> dict:
    return json.loads((EVAL_DIR / "golden_set.json").read_text(encoding="utf-8"))


def platform_config() -> dict:
    """The config a platform (free) run executes with, built by the server's
    own builder so the evaluation can't drift from production."""
    sys.path.insert(0, str(SERVER_DIR))
    from app.runconfig import build_config_overrides
    from app.settings import Settings
    from tradingagents.default_config import DEFAULT_CONFIG

    settings = Settings(
        _env_file=None,
        llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
        deep_think_llm=os.environ.get("DEEP_THINK_LLM", "claude-sonnet-5"),
        quick_think_llm=os.environ.get("QUICK_THINK_LLM", "claude-haiku-4-5"),
        price_vendor=os.environ.get("PRICE_VENDOR", "yfinance"),
    )
    config = dict(DEFAULT_CONFIG)
    config.update(build_config_overrides(settings))
    config["memory_log_path"] = None  # platform runs are stateless
    config["checkpoint_enabled"] = False
    return config
