from web.session import SessionStore, new_deck


class _StubBot:
    def get_action(self, player, state):  # pragma: no cover - not called here
        raise AssertionError


def test_new_deck_is_52_unique_and_deterministic():
    a = new_deck(123)
    b = new_deck(123)
    assert len(a) == 52
    assert len({c.key for c in a}) == 52
    assert [c.key for c in a] == [c.key for c in b]
    assert [c.key for c in a] != [c.key for c in new_deck(124)]


def test_create_get_delete():
    store = SessionStore(ttl_s=100, bot_factory=_StubBot)
    s = store.create()
    assert store.get(s.token) is s
    store.delete(s.token)
    assert store.get(s.token) is None


def test_ttl_expiry():
    now = [0.0]
    store = SessionStore(ttl_s=100, bot_factory=_StubBot, clock=lambda: now[0])
    s = store.create()
    now[0] = 100.0
    assert store.get(s.token) is not None   # at the boundary: not > ttl
    now[0] = 201.0                          # last access was at 100
    assert store.get(s.token) is None
    assert store.sweep() == 0               # already dropped on get


def test_sweep_removes_idle():
    now = [0.0]
    store = SessionStore(ttl_s=10, bot_factory=_StubBot, clock=lambda: now[0])
    s = store.create()
    now[0] = 20.0
    assert store.sweep() == 1
    assert store.get(s.token) is None


def test_start_hand_resets_history_and_sets_active():
    store = SessionStore(ttl_s=100, bot_factory=_StubBot)
    s = store.create()
    s.history.append(object())
    s.start_hand(seed=7)
    assert s.history == []
    assert s.hand_active is True
    assert len(s.deck) == 52
    assert s.last_state is None
