"""Deployed pot-fraction raises must match the trainer's sizes. The engine
sizes a raise from the pot net of the amount the actor owes
(EngineGameState::bet_context: pot - owe); the deploy bridge holds the gross
live pot, so it must net the call out before calling vocab.target_bb."""
import pytest

import agents.sixmax_agent as sa

sixmax = pytest.importorskip("sixmax")

_C4 = sa.os.path.join(sa._ROOT, "sixmax", "configs", "hu200_c4.toml")
_VOCAB = sa._load_vocab(_C4, "blueprint")


def _facing_state(*actions):
    deck = list(range(52))
    cfg = sixmax.EngineConfig(num_players=2, starting_stack=200.0)
    st = sixmax.EngineGameState(cfg, 0, deck, _VOCAB, [200.0, 200.0])
    for a in actions:
        st.apply(a)
    return st


def _gross(ctx):
    return ctx.pot + ctx.to_call


def test_engine_bet_context_nets_the_call():
    ctx = sa.engine_bet_context(pot_bb=10.0, current_bet_bb=4.0,
                                to_call_bb=3.0, stack_bb=190.0)
    assert (ctx.pot, ctx.current_bet, ctx.to_call, ctx.stack) == (7.0, 4.0, 3.0, 190.0)


@pytest.mark.parametrize("line", [
    (4,),            # BB facing a 3.5 BB open
    (5, 7),          # SB facing a 0.75-pot 3-bet over a 5 BB open
    (3, 2, 1, 6),    # flop: BB checked, SB bet 0.33 pot; BB to act facing it
])
def test_deployed_raise_matches_engine_target(line):
    st = _facing_state(*line)
    ctx_e = st.bet_context()
    assert ctx_e.to_call > 0
    deploy = sa.SixmaxDeployStrategy(object(), _VOCAB)
    for idx in (6, 7, 8):                      # the pot-fraction sizes
        _, raise_to = deploy._finish(idx, _gross(ctx_e), ctx_e.current_bet,
                                     ctx_e.to_call, ctx_e.stack)
        assert raise_to == pytest.approx(_VOCAB.target_bb(idx, ctx_e))


def test_unfaced_bet_is_unchanged():
    # Nothing to call: net and gross pot are equal, sizes as before.
    st = _facing_state(3, 2)                  # flop, BB first to act
    ctx_e = st.bet_context()
    assert ctx_e.to_call == 0
    deploy = sa.SixmaxDeployStrategy(object(), _VOCAB)
    for idx in (6, 7, 8):
        assert deploy._finish(idx, ctx_e.pot, ctx_e.current_bet, 0.0,
                              ctx_e.stack)[1] == pytest.approx(
            _VOCAB.target_bb(idx, ctx_e))
