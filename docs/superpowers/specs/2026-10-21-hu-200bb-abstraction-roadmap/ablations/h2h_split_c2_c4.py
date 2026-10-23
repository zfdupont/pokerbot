"""Split C2-vs-C4 head-to-head winnings by the preflop opening sequence."""
import collections, sys
sys.path.insert(0, "/Users/zfdupont/pokerbot")
from agents.sixmax_agent import SixmaxAgent
from scripts.eval_hu_sanity import _play_seeded
BB = 2
game = []
class Rec:
    def __init__(self, name, agent): self.name, self.agent = name, agent
    @property
    def _rng(self): return self.agent._rng   # _play_seeded reseeds it per hand
    def get_action(self, player, gs):
        a, amt = self.agent.get_action(player, gs)
        if gs.betting_round == 0: game.append((self.name, a.name, None if amt is None else amt / BB))
        return a, amt
c2 = {s: Rec("C2", SixmaxAgent("sixmax/checkpoints/hu200_c2_38477898.bin", config_toml="sixmax/configs/hu200_c2.toml")) for s in (0, 1)}
c4 = {s: Rec("C4", SixmaxAgent("sixmax/checkpoints/hu200_c4_38477898.bin", config_toml="sixmax/configs/hu200_c4.toml")) for s in (0, 1)}
res = collections.defaultdict(lambda: [0, 0.0]); total = 0.0; n = 0
for seed in range(1, 9):
    for h in range(20000):
        s = seed * 1_000_003 + h
        for hero_seat in (0, 1):
            game.clear()
            d = _play_seeded(c2[hero_seat], c4[hero_seat], hero_seat, s, BB, 400) / BB   # C2's result in BB
            total += d; n += 1
            opener, a0, amt0 = game[0]
            if a0 in ("FOLD",): spot = f"{opener} folds SB"
            elif a0 in ("CALL", "CHECK"): spot = f"{opener} limps"
            else:
                resp = game[1][1] if len(game) > 1 else "-"
                resp = {"FOLD": "fold", "CALL": "call", "CHECK": "call"}.get(resp, "raise")
                size = "jam" if amt0 and amt0 >= 150 else f"{amt0:g}"
                spot = f"{opener} opens {size} -> {'C4' if opener == 'C2' else 'C2'} {resp}"
            res[spot][0] += 1; res[spot][1] += d
print(f"games {n}; C2 total {100*total/n:+.2f} BB/100 (gate: +7.40)")
print(f"{'spot':<34}{'games':>8}{'share':>7}{'C2 BB/game':>12}{'contrib BB/100':>16}")
for spot, (k, s) in sorted(res.items(), key=lambda kv: -abs(kv[1][1])):
    print(f"{spot:<34}{k:>8}{100*k/n:>6.1f}%{s/k:>12.3f}{100*s/n:>+16.2f}")
