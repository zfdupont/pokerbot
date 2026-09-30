from web.config import load_config


def test_defaults():
    cfg = load_config(env={})
    assert cfg.stack == 200
    assert cfg.small_blind == 1
    assert cfg.big_blind == 2
    assert cfg.session_ttl_s == 3600
    assert "https://zfdupont.com" in cfg.cors_origins
    assert cfg.config_toml.endswith("sixmax/configs/default.toml")


def test_env_overrides():
    cfg = load_config(env={
        "POKERBOT_STACK": "100",
        "POKERBOT_SMALL_BLIND": "2",
        "POKERBOT_SESSION_TTL": "60",
        "POKERBOT_CHECKPOINT": "/checkpoints/hu.bin",
        "POKERBOT_CORS_ORIGINS": "https://a.example, https://b.example",
    })
    assert cfg.stack == 100
    assert cfg.small_blind == 2
    assert cfg.big_blind == 4
    assert cfg.session_ttl_s == 60
    assert cfg.checkpoint == "/checkpoints/hu.bin"
    assert cfg.cors_origins == ("https://a.example", "https://b.example")
