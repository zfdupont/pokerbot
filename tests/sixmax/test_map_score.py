"""Map scoring: the within-bucket spread kernel on hand-computed inputs, its
refusals, and score_bucket_maps' refusals. Scoring real maps needs the
multi-hundred-MB equity store, so it is exercised by scripts/score_bucket_map.py
on real tables, not here."""
import random

import pytest

import sixmax


def test_spread_scalar_unweighted():
    # Buckets {0.1, 0.3} and {0.7, 0.9}: every hand is 0.1 from its bucket
    # mean. One bucket: mean 0.5, deviations 0.4/0.2/0.2/0.4.
    err, base = sixmax._bucket_spread(
        [[0.1], [0.3], [0.7], [0.9]], [1, 1, 1, 1], [0, 0, 1, 1], 2)
    assert err == pytest.approx(0.1)
    assert base == pytest.approx(0.3)


def test_spread_scalar_weighted():
    # Bucket 0 mean (3*0.1 + 0.3)/4 = 0.15: deviations 3*0.05 + 0.15 = 0.3.
    # Bucket 1: 0.1 + 0.1 = 0.2. Total 0.5 over weight 6.
    err, _ = sixmax._bucket_spread(
        [[0.1], [0.3], [0.7], [0.9]], [3, 1, 1, 1], [0, 0, 1, 1], 2)
    assert err == pytest.approx(0.5 / 6)


def test_spread_cdf_is_emd_per_dim():
    # CDFs of a point mass in bin 0 and in bin 1 share a bucket: the mean CDF
    # is [0.5, 1], each hand is 0.5 away in one dimension, divided by 2 dims.
    err, base = sixmax._bucket_spread([[1.0, 1.0], [0.0, 1.0]], [1, 1],
                                      [0, 0], 1)
    assert err == pytest.approx(0.25)
    assert base == pytest.approx(0.25)


def test_spread_singleton_buckets_are_exact():
    feats = [[0.2, 0.5], [0.4, 0.9], [0.1, 1.0]]
    err, base = sixmax._bucket_spread(feats, [2, 1, 5], [0, 1, 2], 3)
    assert err == 0.0
    assert base > 0.0


def test_spread_thread_count_independent():
    rng = random.Random(7)
    n = 5000
    feats = [[rng.random() for _ in range(4)] for _ in range(n)]
    weight = [rng.randint(1, 24) for _ in range(n)]
    buckets = [rng.randrange(9) for _ in range(n)]
    one = sixmax._bucket_spread(feats, weight, buckets, 9, num_threads=1)
    many = sixmax._bucket_spread(feats, weight, buckets, 9, num_threads=7)
    assert one == many


def test_spread_finer_split_never_worse():
    # Splitting a bucket can only lower the spread around the bucket means
    # for scalars ordered within each split.
    vals = [[v / 20] for v in range(20)]
    w = [1] * 20
    coarse, _ = sixmax._bucket_spread(vals, w, [i // 10 for i in range(20)], 2)
    fine, _ = sixmax._bucket_spread(vals, w, [i // 5 for i in range(20)], 4)
    assert fine < coarse


@pytest.mark.parametrize("args, match", [
    (([], [], [], 1), "no features"),
    (([[0.1]], [1, 1], [0], 1), "size"),
    (([[0.1], [0.2, 0.3]], [1, 1], [0, 0], 1), "same length"),
    (([[0.1]], [1], [1], 1), "bucket id"),
    (([[0.1]], [1], [0], 0), "k must be"),
    (([[0.1]], [1], [0], 256), "k must be"),
    (([[0.1]], [0], [0], 1), "weight is zero"),
])
def test_spread_refuses_bad_input(args, match):
    with pytest.raises(ValueError, match=match):
        sixmax._bucket_spread(*args)


def test_score_refuses_no_maps(tmp_path):
    with pytest.raises(ValueError, match="no maps"):
        sixmax.score_bucket_maps(str(tmp_path / "equity.bin"), [])


def test_score_refuses_missing_map(tmp_path):
    with pytest.raises(ValueError):
        sixmax.score_bucket_maps(str(tmp_path / "equity.bin"),
                                 [str(tmp_path / "missing.bin")])
