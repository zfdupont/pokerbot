"""Config resolution + end-to-end smoke for scripts/train_sixmax.py."""
import importlib.util
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "train_sixmax.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("train_sixmax", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Args:
    """argparse.Namespace stand-in; unset flags are None."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __getattr__(self, _):
        return None


def test_resolve_config_precedence(tmp_path):
    mod = _load_script()
    toml = tmp_path / "cfg.toml"
    toml.write_text(
        "[abstraction]\nflop_buckets = 8\n"
        "[train.blueprint]\niterations = 500\nnum_players = 3\n")
    cfg = mod.resolve_config(_Args(config=str(toml), iterations=250), _ROOT)
    assert cfg["iterations"] == 250          # CLI beats TOML
    assert cfg["num_players"] == 3           # TOML beats builtin
    assert cfg["flop_buckets"] == 8          # [abstraction] section merged
    assert cfg["turn_buckets"] == 50         # builtin default survives


def test_hu_config_sets_two_players():
    mod = _load_script()
    cfg = mod.resolve_config(
        _Args(config=os.path.join(_ROOT, "sixmax", "configs", "hu.toml")), _ROOT)
    assert cfg["num_players"] == 2
    assert cfg["checkpoint"].endswith("hu_blueprint.bin")
    assert cfg["checkpoint_interval"] > 0
    # The bucket-map example is commented out: hu.toml keeps the legacy path.
    assert mod.BUILTIN_DEFAULTS["bucket_map"] == ""
    assert cfg["bucket_map"] == ""


def test_resolve_config_accepts_bucket_map(tmp_path):
    mod = _load_script()
    toml = tmp_path / "map.toml"
    toml.write_text(
        '[abstraction]\nbucket_map = "sixmax/tables/map_pct_50_50_20.bin"\n')
    cfg = mod.resolve_config(_Args(config=str(toml)), _ROOT)
    assert cfg["bucket_map"] == "sixmax/tables/map_pct_50_50_20.bin"


def test_hu_map_config():
    mod = _load_script()
    cfg = mod.resolve_config(
        _Args(config=os.path.join(_ROOT, "sixmax", "configs",
                                  "hu_map.toml")), _ROOT)
    assert cfg["bucket_map"] == "sixmax/tables/map_pct_50_50_20.bin"
    assert cfg["iterations"] == 30_000_000
    assert cfg["checkpoint_interval"] == 5_000_000
    assert cfg["checkpoint"].endswith("hu_map_pct.bin")
    assert cfg["snapshots"] is True
    assert cfg["num_players"] == 2


def test_resolve_config_rejects_unknown_keys(tmp_path):
    mod = _load_script()
    toml = tmp_path / "bad.toml"
    toml.write_text("[train.blueprint]\nnot_a_key = 1\n")
    try:
        mod.resolve_config(_Args(config=str(toml)), _ROOT)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_end_to_end_smoke(tmp_path):
    """Tiny full run: builds abstraction, trains, saves a loadable artifact."""
    toml = tmp_path / "smoke.toml"
    toml.write_text(
        "[abstraction]\n"
        "flop_buckets = 6\nturn_buckets = 6\nriver_buckets = 4\n"
        "equity_rollouts = 20\nquantile_samples = 150\nseed = 5\n"
        "[actions.blueprint]\n"
        "preflop_opens = [{size = 2.5, unit = \"bb\"}]\n"
        "bet_sizes = [{size = 0.33, unit = \"pot\"}]\n"
        "include_allin = true\n"
        "[train.blueprint]\n"
        "num_players = 2\niterations = 40\nnum_threads = 1\nseed = 3\n")
    ckpt = tmp_path / "bp.bin"
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", str(toml),
         "--checkpoint", str(ckpt)],
        capture_output=True, text=True, cwd=_ROOT, timeout=600)
    assert result.returncode == 0, result.stderr
    assert ckpt.exists()
    assert (tmp_path / "bp.bin.config.toml").exists()   # effective-config snapshot
    import sixmax  # conftest already force-loaded the extension
    abs_ = sixmax.load_abstraction(str(ckpt))
    assert abs_.num_buckets(1) == 6


def test_train_with_map_writes_sixbp002(tmp_path, synthetic_bucket_map,
                                        monkeypatch):
    """A [abstraction] bucket_map config trains, saves SIXBP002, and resumes."""
    import sixmax  # conftest already force-loaded the extension

    mod = _load_script()
    monkeypatch.setenv("SIXMAX_BUCKET_MAP", synthetic_bucket_map)
    expected = sixmax.Abstraction.from_bucket_map(synthetic_bucket_map)
    toml = tmp_path / "map.toml"
    toml.write_text(
        "[abstraction]\n"
        f"bucket_map = {json.dumps(synthetic_bucket_map)}\n"
        "[actions.blueprint]\n"
        'preflop_opens = [{size = 2.5, unit = "bb"}]\n'
        'bet_sizes = [{size = 0.33, unit = "pot"}]\n'
        "include_allin = true\n"
        "[train.blueprint]\n"
        "num_players = 2\niterations = 200\nnum_threads = 1\nseed = 3\n")
    env = {**os.environ, "SIXMAX_BUCKET_MAP": synthetic_bucket_map}
    ckpt = tmp_path / "map_bp.bin"
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", str(toml),
         "--checkpoint", str(ckpt)],
        capture_output=True, text=True, cwd=_ROOT, env=env, timeout=600)
    assert result.returncode == 0, result.stderr
    with open(ckpt, "rb") as f:
        assert f.read(8) == b"SIXBP002"
    assert "bucket map" in result.stderr.lower()
    assert f"{expected.hash():016x}" in result.stderr

    vocab = mod._load_vocab(_ROOT, str(toml))
    strategy = sixmax.BlueprintStrategy.load(str(ckpt), vocab)
    assert strategy.iterations() == 200
    assert strategy.abstraction().is_map()
    assert strategy.abstraction().hash() == expected.hash()

    resumed = tmp_path / "map_bp_resumed.bin"
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", str(toml),
         "--resume", str(ckpt), "--checkpoint", str(resumed),
         "--iterations", "100"],
        capture_output=True, text=True, cwd=_ROOT, env=env, timeout=600)
    assert result.returncode == 0, result.stderr
    resumed_strategy = sixmax.BlueprintStrategy.load(str(resumed), vocab)
    assert resumed_strategy.iterations() == 300
    assert resumed_strategy.abstraction().hash() == expected.hash()


# ── Progress observability (heartbeat) ────────────────────────────────────────

def test_format_duration():
    mod = _load_script()
    assert mod._format_duration(0) == "0s"
    assert mod._format_duration(73) == "1m 13s"
    assert mod._format_duration(63 * 3600 + 5 * 60) == "63h 05m"
    assert mod._format_duration(float("inf")) == "?"
    assert mod._format_duration(-5) == "0s"


def test_format_progress():
    mod = _load_script()
    assert mod.format_progress(3000, 10000, 96.0, 143102) == \
        "[3,000/10,000]  96 it/s  ETA 1m 13s  143,102 infosets"
    assert mod.format_progress(10000, 10000, 50.0) == \
        "[10,000/10,000]  50 it/s  ETA 0s"
    assert mod.format_progress(0, 100, 0.0) == "[0/100]  0 it/s  ETA ?"


def test_progress_reporter_reports_once():
    mod = _load_script()

    class _FakeTrainer:
        def __init__(self):
            self.n = 1000
        def iterations(self):
            return self.n
        def num_infosets(self):
            return 143102

    lines = []
    tr = _FakeTrainer()
    rep = mod._ProgressReporter(tr, 30_000_000, 60, log=lines.append)
    rep.start()                      # base = 1000
    tr.n = 1_001_000
    rep.report_once()
    rep.stop()
    assert len(lines) == 1, lines
    assert lines[0].startswith("[1,000,000/30,000,000]"), lines[0]
    assert lines[0].endswith("143,102 infosets"), lines[0]


def test_resolve_config_report_interval_precedence(tmp_path):
    mod = _load_script()
    assert mod.BUILTIN_DEFAULTS["report_interval"] == 60
    toml = tmp_path / "cfg.toml"
    toml.write_text("[train.blueprint]\nreport_interval = 5\n")
    cfg = mod.resolve_config(_Args(config=str(toml)), _ROOT)
    assert cfg["report_interval"] == 5
    cfg = mod.resolve_config(_Args(config=str(toml), report_interval=9), _ROOT)
    assert cfg["report_interval"] == 9


def test_bucket_map_cli_flag_overrides_toml(tmp_path):
    mod = _load_script()
    toml = tmp_path / "map.toml"
    toml.write_text('[abstraction]\nbucket_map = "from_toml.bin"\n')
    cfg = mod.resolve_config(
        _Args(config=str(toml), bucket_map="from_cli.bin"), _ROOT)
    assert cfg["bucket_map"] == "from_cli.bin"


def test_resume_uses_bucket_map_candidate(tmp_path, synthetic_bucket_map):
    """A renamed map is found through --bucket-map on resume (the content-hash
    check still guards it, and the TOML key is overridden)."""
    original = tmp_path / "original_map.bin"
    with open(synthetic_bucket_map, "rb") as src, open(original, "wb") as dst:
        dst.write(src.read())
    toml = tmp_path / "map.toml"
    toml.write_text(
        "[actions.blueprint]\n"
        'preflop_opens = [{size = 2.5, unit = "bb"}]\n'
        'bet_sizes = [{size = 0.33, unit = "pot"}]\n'
        "include_allin = true\n"
        "[train.blueprint]\n"
        "num_players = 2\niterations = 40\nnum_threads = 1\nseed = 3\n")
    ckpt = tmp_path / "map_bp.bin"
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", str(toml),
         "--bucket-map", str(original), "--checkpoint", str(ckpt)],
        capture_output=True, text=True, cwd=_ROOT, timeout=600)
    assert result.returncode == 0, result.stderr
    with open(ckpt, "rb") as f:
        assert f.read(8) == b"SIXBP002"

    moved = tmp_path / "moved" / "renamed.bin"
    moved.parent.mkdir()
    os.replace(original, moved)

    resumed = tmp_path / "map_bp_resumed.bin"
    result = subprocess.run(
        [sys.executable, _SCRIPT, "--config", str(toml),
         "--bucket-map", str(moved), "--resume", str(ckpt),
         "--checkpoint", str(resumed), "--iterations", "20"],
        capture_output=True, text=True, cwd=_ROOT, timeout=600)
    assert result.returncode == 0, result.stderr
    assert "warning" in result.stderr.lower()
    import sixmax
    assert sixmax.load_abstraction(
        str(resumed), bucket_map=str(moved)).hash() > 0
