"""cloud_train.sh gains --config / --checkpoint-name passthrough.

Static checks (the script provisions cloud VMs, so it is not executed here):
syntax is valid and the flags reach the trainer command.
"""
import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "cloud_train.sh")


def test_cloud_train_accepts_config_flag():
    assert subprocess.run(["bash", "-n", _SCRIPT]).returncode == 0
    text = open(_SCRIPT).read()
    assert "--config)" in text
    assert "--checkpoint-name)" in text
    assert "--config /pokerbot/$CONFIG" in text
    assert "--checkpoint $CKPT_PATH" in text
