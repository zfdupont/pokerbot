from cfr.info_set import InfoSet, stack_bucket


class TestInfoSet:
    def _make(self, **kwargs):
        defaults = dict(
            player=0, hand_bucket=2, street=0, board_bucket=0,
            betting_history=(), stack_bucket=3,
        )
        defaults.update(kwargs)
        return InfoSet(**defaults)

    def test_is_hashable(self):
        i = self._make()
        d = {i: 1}
        assert d[i] == 1

    def test_equality_same_fields(self):
        a = self._make()
        b = self._make()
        assert a == b

    def test_inequality_different_player(self):
        assert self._make(player=0) != self._make(player=1)

    def test_inequality_different_history(self):
        a = self._make(betting_history=("b0.5", "c"))
        b = self._make(betting_history=("check",))
        assert a != b

    def test_immutable(self):
        i = self._make()
        try:
            i.player = 1
            assert False, "should have raised"
        except Exception:
            pass


class TestStackBucket:
    def test_very_short_stack(self):
        assert stack_bucket(5.0, big_blind=1.0) == 0   # < 10BB

    def test_short_stack(self):
        assert stack_bucket(15.0, big_blind=1.0) == 1  # 10-25BB

    def test_medium_stack(self):
        assert stack_bucket(40.0, big_blind=1.0) == 2  # 25-50BB

    def test_deep_stack(self):
        assert stack_bucket(75.0, big_blind=1.0) == 3  # 50-100BB

    def test_very_deep_stack(self):
        assert stack_bucket(200.0, big_blind=1.0) == 4  # 100BB+
