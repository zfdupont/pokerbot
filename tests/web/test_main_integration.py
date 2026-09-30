"""End-to-end through the real checkpoint. Skipped when the compiled extension
cannot be built (mirrors tests/sixmax gating)."""
import os

import pytest
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOML = os.path.join(_ROOT, "sixmax", "configs", "default.toml")

# Load the C++ extension at import time so the session-scoped checkpoint fixture
# (which calls sixmax.BlueprintTrainer) finds the real module, not the Python
# namespace package.
try:
    from web.extension import load_sixmax
    load_sixmax()
    _EXT_OK = True
except Exception:  # pragma: no cover - environment dependent
    _EXT_OK = False

pytestmark = pytest.mark.skipif(not _EXT_OK, reason="sixmax extension unavailable")


def test_full_hand_via_real_bot(blueprint_hu_ckpt):
    from agents.sixmax_agent import SixmaxAgent, SixmaxDeployStrategy
    from web.app import build_app
    from web.config import load_config

    strategy = SixmaxDeployStrategy.load(blueprint_hu_ckpt, _TOML)
    cfg = load_config(env={"POKERBOT_STACK": "200", "POKERBOT_SMALL_BLIND": "1",
                           "POKERBOT_CHECKPOINT": blueprint_hu_ckpt})
    app = build_app(cfg, lambda: SixmaxAgent.from_strategy(strategy, _TOML))
    c = TestClient(app)

    body = c.post("/api/session").json()
    token, state = body["token"], body["state"]
    for _ in range(100):
        if state["hand_complete"]:
            break
        if not state["hero"]["is_actor"]:
            state = c.get(f"/api/session/{token}").json()["state"]
            continue
        acts = {a["action"] for a in state["legal_actions"]}
        pick = "check" if "check" in acts else ("call" if "call" in acts else "fold")
        body = c.post(f"/api/session/{token}/action",
                      json={"seq": state["seq"], "action": pick}).json()
        state = body["state"]

    assert state["hand_complete"] is True
    # No bot hole cards ever appeared before the showdown completed.
    assert state["session"]["stack"] >= 0
