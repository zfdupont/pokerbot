"""Replay every logged Slumbot decision of ours through the agent's key
builder: was the infoset in the table, and what did the hand win?"""
import collections, json, random, sys
sys.path.insert(0, "/Users/zfdupont/pokerbot")
import agents.sixmax_agent as sa
from scripts import eval_slumbot as ev

log, ckpt, cfg = sys.argv[1:4]
agent = sa.SixmaxAgent(ckpt, config_toml=cfg)
seen = {}
orig = sa.SixmaxDeployStrategy.decide
def decide(self, **kw):
    ab = self._strategy.abstraction()
    card = self._card_bucket(kw["hole"], kw["board"], kw["street"])
    eff = kw.get("eff_stack_bb", kw["stack_bb"])
    key = ab.pack_key(card, kw["street"], list(kw["raises_per_street"]), kw["pot_bb"], eff, kw["live"], kw["after"])
    seen["miss"] = not self._strategy.probs(key)
    seen["ctx"] = (kw["street"], tuple(kw["raises_per_street"]), round(kw["pot_bb"], 2), round(kw["to_call_bb"], 2))
    return orig(self, **kw)
sa.SixmaxDeployStrategy.decide = decide

def decisions(r):
    """(action prefix, board so far) at each of our decision points."""
    me = r["pos"]; segs = r["action"].split("/"); out = []
    for st, seg in enumerate(segs):
        actor = 1 if st == 0 else 0
        pos = 0
        for tok in ev.TOKEN_RE.findall(seg):
            if actor == me:
                prefix = "/".join(segs[:st] + [seg[:pos]])
                board = r["board"][: {0: 0, 1: 3, 2: 4, 3: 5}[st]]
                out.append((prefix, board, tok))
            pos += len(tok); actor ^= 1
    return out

per_hand = []; miss_by_ctx = collections.Counter(); all_by_street = collections.Counter(); miss_by_street = collections.Counter()
for line in open(log):
    r = json.loads(line); misses = 0; first_miss = None
    for prefix, board, tok in decisions(r):
        resp = {"action": prefix, "board": board, "client_pos": r["pos"], "hole_cards": r["hole"]}
        gs, hero, _ = ev.build_state(resp)
        agent.get_action(hero, gs)
        st = seen["ctx"][0]; all_by_street[st] += 1
        if seen["miss"]:
            misses += 1; miss_by_street[st] += 1; miss_by_ctx[seen["ctx"][:3]] += 1
            if first_miss is None: first_miss = (prefix, tok, seen["ctx"])
    per_hand.append({"hand": r["hand"], "misses": misses, "first_miss": first_miss,
                     "winnings": r["winnings"], "baseline": r["baseline"], "action": r["action"]})
json.dump(per_hand, open(sys.argv[4], "w"))
print("decisions by street:", dict(all_by_street))
print("misses by street:   ", dict(miss_by_street), {s: f"{100*miss_by_street[s]/all_by_street[s]:.1f}%" for s in all_by_street})
print("top missed contexts (street, raises, pot_bb):")
for ctx, n in miss_by_ctx.most_common(12): print("  ", ctx, n)
