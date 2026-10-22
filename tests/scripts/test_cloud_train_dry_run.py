"""cloud_train.sh --dry-run / --bucket-map pre-flight (no credentials, no VM).

The script provisions real cloud VMs, so these tests only exercise the
pre-flight path: every run uses a credential-free environment (no
HCLOUD_TOKEN), and --dry-run must print exactly what it would send and exit 0
without sourcing .env, touching hcloud/ssh/rsync/docker or requiring the bucket
map file to exist locally.
"""
import os
import shutil
import subprocess

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "cloud_train.sh")
_HU200 = "sixmax/configs/hu200.toml"
_MAP = "sixmax/tables/map_pct_50_50_20.bin"


# PATH only — no HCLOUD_TOKEN / HETZNER_SSH_KEY_NAME / GHCR_PAT.
_CLEAN_PATH = os.environ.get("PATH", "/usr/bin:/bin")


def _run(*args, path=None):
    return subprocess.run(
        ["bash", _SCRIPT, *args],
        capture_output=True, text=True, cwd=_ROOT,
        env={"PATH": path or _CLEAN_PATH},
        timeout=120)


def test_dry_run_with_map():
    result = _run("--dry-run", "--config", _HU200, "--bucket-map", _MAP,
                  "--checkpoint-name", "hu200_c0.bin")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # The map is shipped separately and mounted read-only for the container.
    assert "SIXMAX_BUCKET_MAP=/pokerbot/tables/map_pct_50_50_20.bin" in out
    assert "-v /tmp/pokerbot/tables:/pokerbot/tables:ro" in out
    # sixmax/tables/ is never part of the source sync.
    assert "--exclude='sixmax/tables/'" in out
    # The trainer is told which map to use, and the upload target is printed.
    assert "--bucket-map /pokerbot/tables/map_pct_50_50_20.bin" in out
    assert "bucket map upload target: root@<server>:/tmp/pokerbot/tables/" \
           "map_pct_50_50_20.bin" in out
    assert "--checkpoint /pokerbot/sixmax/checkpoints/hu200_c0.bin" in out
    # A dry run never needs (or asks for) credentials.
    assert "HCLOUD_TOKEN" not in out + result.stderr


def test_dry_run_excludes_local_envs_and_worktrees():
    # Sibling worktrees and virtualenvs (torch alone is GBs) never ship.
    result = _run("--dry-run", "--config", _HU200, "--bucket-map", _MAP)
    assert result.returncode == 0, result.stderr
    assert "--exclude='.worktrees/'" in result.stdout
    assert "--exclude='.venv/'" in result.stdout


def test_dry_run_relative_map_path():
    # Review Focus 4: a relative path with ../ segments resolves to its basename
    # on the box, and the dry run must not require the file to exist.
    result = _run("--dry-run", "--config", _HU200,
                  "--bucket-map", "./sixmax/tables/../tables/map_x.bin")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "SIXMAX_BUCKET_MAP=/pokerbot/tables/map_x.bin" in out
    assert "--bucket-map /pokerbot/tables/map_x.bin" in out
    assert "/tmp/pokerbot/tables/map_x.bin" in out
    assert "tables/../tables" not in out


def test_dry_run_without_map_has_no_map_plumbing(tmp_path):
    toml = tmp_path / "legacy.toml"
    toml.write_text("[abstraction]\nflop_buckets = 8\n"
                    "[train.blueprint]\nnum_players = 2\n")
    result = _run("--dry-run", "--config", str(toml))
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--exclude='sixmax/tables/'" in out      # excluded unconditionally
    assert "SIXMAX_BUCKET_MAP" not in out
    assert "--bucket-map" not in out
    assert "/pokerbot/tables" not in out


def test_missing_map_refused():
    # hu200.toml names a bucket map in its [abstraction] block, but the map is
    # no longer shipped with the source sync: refuse instead of training blind.
    result = _run("--config", _HU200,
                  "--checkpoint-name", "hu200_c0.bin")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "pass --bucket-map" in result.stdout + result.stderr


def test_commented_bucket_map_is_not_a_requirement(tmp_path):
    toml = tmp_path / "commented.toml"
    toml.write_text("[abstraction]\n"
                    '# bucket_map = "sixmax/tables/map.bin"\n'
                    "[train.blueprint]\nnum_players = 2\n")
    result = _run("--dry-run", "--config", str(toml))
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("python3.11") is None,
                    reason="python3.11 (tomllib) not installed")
def test_config_check_tomllib_branch(tmp_path):
    """The operator's Mac may run python3 >= 3.11 (tomllib) instead of the
    grep fallback exercised above; both must detect the same TOML."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").symlink_to(shutil.which("python3.11"))
    toml = tmp_path / "map.toml"
    toml.write_text('[abstraction]\nbucket_map = "sixmax/tables/map.bin"\n'
                    "# bucket_map = \"other.bin\"\n")
    result = _run("--config", str(toml),
                  path=f"{bindir}{os.pathsep}{_CLEAN_PATH}")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "pass --bucket-map" in result.stdout + result.stderr


def test_unreadable_config_refused(tmp_path):
    # Never guess: an unreadable config might hide a bucket_map requirement.
    result = _run("--dry-run", "--config", str(tmp_path / "missing.toml"))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not read" in result.stdout + result.stderr
