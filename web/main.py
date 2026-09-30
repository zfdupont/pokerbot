"""Entry point: load the extension + checkpoint, then serve."""
from __future__ import annotations

import argparse

import uvicorn

from web.config import Config, load_config


def build_production_app(config: Config):
    from web.extension import load_sixmax
    load_sixmax()
    from agents.sixmax_agent import SixmaxAgent, SixmaxDeployStrategy
    from web.app import build_app

    if not config.checkpoint:
        raise SystemExit("POKERBOT_CHECKPOINT is not set")
    strategy = SixmaxDeployStrategy.load(config.checkpoint, config.config_toml)
    bot_factory = lambda: SixmaxAgent.from_strategy(  # noqa: E731
        strategy, config.config_toml)
    return build_app(config, bot_factory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve pokerbot-web")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    app = build_production_app(load_config())
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
