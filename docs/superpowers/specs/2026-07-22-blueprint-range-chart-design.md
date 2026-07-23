# Blueprint Preflop Range Chart — Design

**Date:** 2026-07-22
**Status:** Approved (brainstorming), pending implementation plan
**Author:** zfdupont + Claude

## Purpose

Render the six-max blueprint's **BTN-open** and **SB-open** preflop RFI ranges as
colored 13×13 range charts in the terminal, each with a range-width headline, so a
human can eyeball how the bot's opening ranges compare to human-professional opening
frequencies (e.g. a pro opens the button ~45–50%).

This is a **read-only diagnostic**. It renders only; it bakes in no reference-chart
data and computes no automated diff against a "correct" range.

## Scope

**In scope**
- A new standalone script `scripts/blueprint_range_chart.py`.
- Backend: the six-max blueprint (`sixmax/checkpoints/blueprint*.bin`), the only trained
  bot with a real BTN. Defaults to the newest snapshot.
- Two spots: **BTN open (RFI)** and **SB open (RFI)** — both unopened preflop pots.
- Terminal ANSI output: per-position 13×13 grid + range-width % + per-role summary.

**Out of scope (YAGNI)**
- BB defend, CO/UTG RFI, or any 3-bet/postflop spot.
- The tabular heads-up bot (`cfr/`, `.pkl`) — `scripts/range_chart.py` already covers it
  and has no true BTN. The new script only borrows that file's display *ideas*.
- Any baked-in reference/GTO chart, side-by-side comparison, or diff map. Human eyeballs
  the bot chart against their own knowledge.
- Non-terminal export (HTML/PNG/CSV).

## Background / why this is feasible

The blueprint stores **abstract** infosets keyed by
`(card, street, raises[4], pot, live, after)` — there is **no seat/position field**.
Position is reconstructed from the abstraction geometry:

- Preflop = `street == 0`; unopened pot = all `raises == 0`.
- In an unopened pot, `(live, after)` encodes strategic position: `live` = **live
  opponents** (players not folded, *excluding hero*), `after` = players still to act
  behind hero. This is the `(live_opps, after)` table-size-invariant encoding pinned by
  `tests/sixmax/test_abstract_key.py` (a 6-handed and 5-handed spot with the same
  geometry are literally the same infoset).
- `dump_infosets()` confirms each unopened-preflop `(live, after)` cell holds a full
  169-hand grid, and `preflop_class([c1,c2])` maps concrete hole cards to the card id at
  **full 169-hand resolution** (no bucket collapse preflop).

Empirical validation on the 1.5M snapshot (seat order UTG,HJ,CO,BTN,SB,BB):
- **BTN open** = `(live=2, after=2)` → width 34.6%; AA .84, AKs .91, JTs .52, 54s .77,
  72o .00 — a real button range.
- **SB open** = `(live=1, after=1)` → width 53.0%; wide and aggressive, correct for the
  strategically-heads-up SB-vs-BB spot (same infoset as the HU SB the abstract-key test
  pins).

Shape is poker-sane and the two widths discriminate by position. vs a human pro the bot's
button (~35%) is a touch tight (pros ~45%) but reasonable; SB (~53%) is squarely human.
Surfacing that gap is the point of the tool.

## Data flow

```
newest blueprint_*.bin
   → sixmax.dump_infosets(path)                # [(key, probs, mass, regret), ...]
   → filter street==0 AND all(raises)==0       # unopened preflop
   → index[(live, after, card_id)] = probs
   → for each of 169 grid cells (i, j):
         card_id  = sixmax.preflop_class(sample_cards(i, j))
         probs    = index[(POS.live, POS.after, card_id)]
         rolemass = {fold, passive(limp/call), aggressive(raise)}  # resolve_action_roles(vocab)
   → render grid + width%
```

Reuses, by import (no duplication, no new bridge — stays a `scripts/` tool):
`diagnose_blueprint._sixmax`, `decode_key`, `load_vocab_for`, `resolve_action_roles`,
`_card_int`, and `sixmax.preflop_class`.

## Position mapping + validating test

```
POSITIONS = {
    "BTN": (live=2, after=2),   # folded to button: SB, BB live opponents behind
    "SB":  (live=1, after=1),   # folded to SB: BB the lone live opponent behind
}
```

`live` = live *opponents* (hero excluded), per the `(live_opps, after)` encoding.
Derivation (num_players=6, seat order UTG,HJ,CO,BTN,SB,BB): BTN open = 3 prior folds →
2 live opponents (SB,BB), 2 to act after → `(2,2)`; SB open = 4 prior folds → 1 live
opponent (BB), 1 to act after → `(1,1)`.

Because raise-shape alone can't distinguish positions (every unopened spot raises
premiums), a pytest asserts the *widths discriminate as expected*:
- both mapped cells hold all 169 hands;
- premiums (AA, AKs) aggressive-mass > trash (72o) at each cell;
- SB `(1,1)` width strictly greater than BTN `(2,2)` width (SB-vs-BB opens wider than the
  3-handed button) — the property that would have caught the original mis-mapping.

## Rendering

Reuse `range_chart.py`'s scheme: green (raise >70%), yellow (raise 35–70%), blue
(limp/call >40%), dim (fold). Upper-right triangle = suited, diagonal = pairs,
lower-left = offsuit.

Per position, print:
1. Headline: `Range width: X% raised` (mean aggressive-mass over 169 hands) — the number
   for the pro comparison.
2. The 13×13 colored grid.
3. A compact per-role summary line.

Two charts back-to-back: BTN, then SB.

## Human contribution (`TODO(human)`)

The **cell-classification function** that maps a hand's `{fold, limp, raise}` role mass to
a display color/category. This is the one spot with genuine poker judgment (raise/mixed
thresholds; whether limp earns its own color when the bot barely limps). ~5–8 lines.
Everything else — indexing, position mapping, grid layout, width math — is built before
the handoff.

## Testing

- **Position-mapping test** (above): premiums > trash at each mapped cell; cells present.
- **Smoke test**: the script renders both BTN and SB charts against the newest snapshot
  with a non-zero exit only on real failure.

## Non-goals / invariants respected

- No import of `game/poker.py`; no cross-module bridging beyond the sanctioned
  `scripts/` + `agents/sixmax_agent` helpers already used by `diagnose_blueprint`.
- Read-only: opens no checkpoint for writing, commits no checkpoint files.
```
