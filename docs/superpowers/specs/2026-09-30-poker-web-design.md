# Poker Web — Heads-Up Play Against the Bot

**Date:** 2026-09-30
**Branch:** `feature/poker-web` (to be created)
**Status:** Draft — awaiting review

## Overview

Add an interactive heads-up game to the personal portfolio site (`blogfolio`,
`zfdupont.com`) where a visitor plays a cash session against the trained bot.
The bot is a **dedicated heads-up six-max blueprint** — the existing tabular C++
`BlueprintTrainer` (`sixmax/`), retrained with `num_players = 2`. The game runs in
the **live engine** (`game/poker.py`) and is served by a small **FastAPI** service
that the portfolio page calls over REST.

This introduces the first web UI / HTTP service for the project. The service is a
*host* (like `scripts/openpoker_bot.py`): it imports `game/poker.py` and
`agents/sixmax_agent.py` and is never imported by `cfr/`, `neural_cfr/`, or
`sixmax/`. The `/.mex` invariant wording will be updated during GROW to name the
new `web/` host directory.

## Motivation

The trained bot currently exists only as CLI (`scripts/play.py`) and as an
outbound connector to openpoker.ai. A visitor to the portfolio cannot play it.
Serving a heads-up game on `zfdupont.com` showcases the project directly and is
the natural "display this project" surface.

Six-max blueprints are a poor heads-up opponent: heads-up is a structurally
different game (much wider ranges, higher aggression), and the heads-up slice of
a 143k-infoset six-max table is thinly visited. A dedicated heads-up retrain
concentrates all visits on that slice, converges faster, and yields a smaller
checkpoint. The engine, trainer, and deploy bridge already support 2 players, so
this is a config/CLI change plus a training run, not a code change.

## Confirmed Decisions

| Decision | Choice |
|---|---|
| Bot | Six-max tabular blueprint, **retrained heads-up** (`num_players = 2`) |
| Session model | **Cash session** — stack persists across hands within a visit, P/L tracked |
| Persistence | Survives page reload within the browser (localStorage token + TTL'd in-memory store) |
| UI location | A `/poker` page **inside `blogfolio`** (Next.js) talking to a separate Python service |
| Transport | **REST** (FastAPI) |
| Experience | Single strong bot, **polished** UI (animations, felt styling, sounds) |
| Service exposure | `poker.zfdupont.com` subdomain behind the droplet's nginx (mirrors `wnba.zfdupont.com`) |
| Checkpoint delivery | **Mount** on the droplet (not baked into the image, never committed) |

## Architecture

```
blogfolio:  zfdupont.com/poker            (Next.js client component)
      │  REST/JSON, CORS from https://zfdupont.com
      ▼
nginx on droplet:  poker.zfdupont.com  →  127.0.0.1:8100
      ▼
pokerbot-web container  (FastAPI + uvicorn, 1 worker)
   ├── sixmax.so                       linux/amd64, built by CI
   ├── SixmaxAgent + hu_blueprint.bin  loaded once, shared across sessions
   ├── game/poker.py PokerGame         the live engine (heads-up ordering fixed)
   └── in-memory session store         token → Session, idle TTL
```

Cross-origin: the page is served from `zfdupont.com`, the API from
`poker.zfdupont.com`. The service sets CORS to allow `https://zfdupont.com` (and
`http://localhost:*` in dev).

## Backend Service — new `web/` in `pokerbot`

```
web/
  __init__.py
  app.py        FastAPI app + routes
  config.py     env config (checkpoint path, TTL, CORS origins, blinds/stack)
  session.py    Session dataclass + TTL store + sweeper
  game_runner.py  wraps PokerGame: apply a human action, play the bot + streets,
                  build the ordered event log and the client-facing state snapshot
  extension.py  load sixmax.so honoring SIXMAX_SO_PATH (no Buck2 at runtime)
```

Entry point: `web/app.py` run under uvicorn (`uvicorn web.app:app --host 0.0.0.0
--port 8100 --workers 1`).

### Why a single worker

The session store is an in-process dict. Running more than one uvicorn worker
would route requests for the same token to different stores. v1 pins **1 worker**.
Scale-out path (documented, not built): move the store to Redis and keep sessions
by token.

### Extension loading

`agents/sixmax_agent.py` currently force-builds via Buck2 at import. In the
container the `.so` is prebuilt at `/opt/sixmax.so` and `SIXMAX_SO_PATH` is set
(the existing convention used by `scripts/train_sixmax.py` and
`docker/Dockerfile`). `web/extension.py` force-loads `SIXMAX_SO_PATH` (or falls
back to a Buck2 build in local dev) and registers it in `sys.modules["sixmax"]`
*before* importing `agents.sixmax_agent`, mirroring `tests/sixmax/conftest.py`.

### The live-engine heads-up fix (required)

`game/poker.py` plays heads-up **nonstandard** today:

- `_post_blinds` uses `sb = (button+1) % n`, `bb = (button+2) % n`; at `n == 2`
  that makes the **BB the `button`** player and the SB the other.
- `_first_to_act` returns `(button+1) % n` for preflop and postflop alike; at
  `n == 2` the **same player (SB) acts first on every street**.
- `_assign_positions` uses `list(Position)[-n:]`; at `n == 2` it labels seats
  `CO` / `BTN` — neither is `SB` / `BB`.

The sixmax engine and `canonical_live_after` / `SixmaxAgent` assume **standard**
heads-up (button = SB, first-to-act preflop = button, postflop = BB). A heads-up
blueprint served through the live engine today would therefore receive wrong
position / `after` context and silently misplay.

Fix: special-case `n == 2` in `_post_blinds` (`sb = button`, `bb = (button+1)`),
`_first_to_act` (`preflop = button`, `postflop = button+1`), and
`_assign_positions` (`[SB, BB]`). `n > 2` behavior is unchanged. Verify with
`scripts/eval_hu_sanity.py`. (The tabular `CFRAgent` is position-blind — it
hardcodes `player=0` — so this does not regress it.)

### Game configuration

Hold 100BB effective using integer chips: `starting_stack = 200`,
`small_blind = 1` (⇒ `big_blind = 2`). `SixmaxAgent` rescales by `big_blind` to
the `starting_stack=100, big_blind=1` training frame, so this is exact.

### API contract

All endpoints return JSON. Errors use standard HTTP codes with
`{"error": "<code>"}`.

| Method + path | Request | Response |
|---|---|---|
| `POST /api/session` | `{}` | `{token, state, events}` (deals the first hand) |
| `GET /api/session/{token}` | — | `{token, state, events: []}` |
| `POST /api/session/{token}/action` | `{action, amount?}` | `{events, state}` |
| `POST /api/session/{token}/next-hand` | `{}` | `{events, state}` |
| `POST /api/session/{token}/rebuy` | `{}` | `{events, state}` |
| `DELETE /api/session/{token}` | — | `{cashed_out: true, net}` |
| `GET /health` | — | `{status: "ok"}` |

`action` ∈ `{"fold", "check", "call", "bet", "all_in"}`; `amount` is the raise-to
(total street commitment, matching `game/poker.py`'s convention) and is required
only for `bet`. Illegal actions return `409 {"error": "illegal_action"}`.

#### `state` snapshot (client-facing)

```jsonc
{
  "hand_id": 7,
  "street": "preflop",              // preflop | flop | turn | river | showdown
  "community_cards": ["Ah","Kd","2c"],
  "pot": 6,
  "current_bet": 4,
  "hero": {
    "seat": 0, "position": "SB",
    "hole_cards": ["As","Ks"],
    "stack": 196, "current_bet": 4,
    "is_actor": true
  },
  "bot": { "stack": 198, "current_bet": 2 },
  "legal_actions": [
    {"action": "fold"},
    {"action": "call", "amount": 2},
    {"action": "bet", "min": 4, "max": 196},
    {"action": "all_in", "amount": 196}
  ],
  "hand_complete": false,
  "result": null,                   // populated when hand_complete
  "session": { "start_stack": 200, "stack": 196, "net": -4, "hands_played": 7 }
}
```

**The bot's hole cards are never included** until `showdown` (and only then if
the hand reaches showdown). This is enforced in `game_runner.py` and covered by a
test.

#### `events` (ordered, for animation)

```jsonc
{"type": "hand_start", "hand_id": 7, "button": 0, "hero_pos": "SB"}
{"type": "hole_cards", "seat": 0, "cards": ["As","Ks"]}
{"type": "action", "seat": 1, "action": "call", "amount": 2}
{"type": "street", "name": "flop", "board": ["Ah","Kd","2c"]}
{"type": "showdown", "seat": 0, "cards": ["As","Ks"]}
{"type": "hand_result", "winners": [{"seat": 0, "amount": 6}]}
```

After a completed hand the client shows the result and calls
`/next-hand`; if the hero's stack is `0`, the client offers `/rebuy`.

### Session store

- `token` = `secrets.token_urlsafe(24)`; `Session` holds the `PokerGame`, the
  hero `Player`, the current hand's hero stack, `start_stack`, `hands_played`,
  and `last_seen`.
- Idle TTL (~1h, configurable). A background sweeper drops expired sessions and
  their games. `POST /api/session` with no token always creates a new one; the
  client is responsible for resuming via `GET`/`action` with its stored token.
- Concurrency note: individual sessions are single-owner (one browser); the
  store guards mutations with a lock.

## Frontend — `blogfolio`

New route `app/poker/page.tsx` (client component), plus local components under
`app/poker/`:

- `PokerTable` — felt, seats, board, pot, dealer button.
- `Card` — CSS/SVG card face + back, flip animation.
- `Seat` — stack, current bet, hole cards (bot hidden until showdown).
- `ActionBar` — fold / check / call / bet (with a sizing control bounded by
  `min`/`max`) / all-in.
- `ActionLog` — scrolling history of `events`.
- `SessionBar` — start stack, current stack, net P/L, hands played, "cash out".

Behavior:

- On mount, read the token from `localStorage`; if present, `GET
  /api/session/{token}` to resume, else `POST /api/session`.
- Drive the table by replaying `events` with CSS transitions (card flips, chip
  movement); sound effects behind a muted-by-default toggle.
- API base from `NEXT_PUBLIC_POKER_API` (default `https://poker.zfdupont.com`).
- Tailwind + the site's `<html data-theme>` tokens; add a nav link.
- Graceful failure: if the service is unreachable, show a "table closed" state
  rather than a broken table.

## Heads-Up Checkpoint Training

- New `sixmax/configs/hu.toml`: a copy of `default.toml` with
  `[train.blueprint] num_players = 2`, `checkpoint =
  sixmax/checkpoints/hu_blueprint.bin`, and chunked `checkpoint_interval`.
- Train on Hetzner via `scripts/cloud_train.sh`. That script currently hardcodes
  the default config; it gains a `--config` passthrough (small, backwards
  compatible).
- Select the best checkpoint by **duplicate-deal heads-up eval**, using
  `scripts/eval_hu_sanity.py` against the frozen tabular and neural bots, and a
  duplicate-deal **HU blueprint vs 6-max blueprint** comparison (to confirm the
  dedicated retrain actually wins). Iterate iterations until it does.
- Torch-free throughout — `//sixmax:sixmax` links only `//common:evaluator` +
  pybind11.

## Deployment

### Image

New `web/Dockerfile` on `linux/amd64`, modeled on `docker/Dockerfile`:

1. builder stage — Buck2-build `//sixmax:sixmax`, copy `sixmax.so` to
   `/opt/sixmax.so`;
2. runtime stage — python3.12, `uv pip install --system` FastAPI + uvicorn +
   numpy + tomli, `COPY web/ game/ agents/ models/ util/ sixmax/`, set
   `SIXMAX_SO_PATH=/opt/sixmax.so`, `CMD` = uvicorn.

New CI workflow → `ghcr.io/zfdupont/pokerbot-web:latest`, mirroring
`.github/workflows/build-trainer-image.yml` (native amd64 runner).

### Droplet

- `docker-compose` `web` service, bound to `127.0.0.1:8100`, restart
  `unless-stopped`, healthcheck on `/health`.
- Checkpoint: rsync `hu_blueprint.bin` to
  `/var/www/poker-service/checkpoints/` and mount it read-only
  (`-v .../checkpoints:/checkpoints:ro`,
  `POKERBOT_CHECKPOINT=/checkpoints/hu_blueprint.bin`). **Not** baked into the
  image, **not** committed.
- nginx: new server block for `poker.zfdupont.com` → `127.0.0.1:8100`; TLS via
  the existing certbot flow; Cloudflare DNS A record.
- Deploy: manual `docker compose pull && docker compose up -d web` (matches the
  `wnba` stack). A GH Actions SSH deploy like `blogfolio`'s is a later option.

## Testing

Backend (`pokerbot`, pytest under `tests/web/`):

- Session store: create, TTL expiry, lock behavior.
- `game_runner`: event log ordering; legal-action computation; **bot hole cards
  absent from pre-showdown state**; hand completes → result present; bust →
  rebuy resets stack and preserves `start_stack`.
- HU ordering: `n == 2` blind posting, first-to-act pre/post, positions
  `[SB, BB]`; `n > 2` unchanged. (These live with the engine tests.)
- A real-extension integration test gated the same way as `tests/sixmax`
  (skipped when `sixmax.so` cannot be built), exercising one full hand through
  the API with a real `SixmaxAgent`.
- Unit tests use a **stub strategy** (deterministic/random) so the fast suite
  needs no compiled extension.

Frontend (`blogfolio`): no test infra exists today. Add Vitest component tests
for the pure logic (event/state reducer, action availability). Playwright e2e is
optional, pending the user's call — the design allows it but does not require it.

## Out of Scope (v1)

Accounts, leaderboard, persistent bankroll across visits, multiplayer tables,
WebSocket transport, difficulty selector, real money, multi-worker scaling.

## Risks & Open Items

- **Engine heads-up correctness** — the fix above is a prerequisite; verify with
  `eval_hu_sanity.py` and engine tests before trusting the served bot.
- **Checkpoint distribution** — manual rsync; retraining requires re-uploading.
- **Single-worker session store** — acceptable for a demo; documented.
- **Subdomain setup** — Cloudflare DNS + certbot + a new nginx block on the
  droplet; depends on droplet access.
- **Droplet resources** — one shared `BlueprintStrategy` load; each session holds
  a `PokerGame`. Games are CPU-cheap; monitor if traffic grows.
- **Cross-repo work** — implementation spans `pokerbot`, `blogfolio`, and droplet
  config; the implementation plan must sequence them.

## Implementation Sequencing (rough)

1. Fix `game/poker.py` heads-up ordering + tests.
2. Add `sixmax/configs/hu.toml` + `cloud_train.sh --config`; train + select the
   heads-up checkpoint; validate with `eval_hu_sanity.py`.
3. Build `web/` service (stub-strategy tests first), then the `web/Dockerfile` +
   CI workflow.
4. Build the `blogfolio` `/poker` page against a local service.
5. Deploy: image, droplet compose + checkpoint mount, nginx/DNS, then the
   `blogfolio` deploy.
