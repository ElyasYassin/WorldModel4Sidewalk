"""Sensor mount overrides (mount studies): in-process and via GO2WM_SENSOR_MOUNTS in a fresh process."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from go2wm.sim import sensors


def test_apply_mount_overrides_and_restore():
    old = {k: dict(v) for k, v in sensors.SENSOR_SPECS.items()}
    try:
        sensors.apply_mount_overrides('{"depth_d405": {"pitch_deg": 45.0, "mount_pos": [0.33, 0.0, 0.08]}}')
        assert sensors.SENSOR_SPECS["depth_d405"]["pitch_deg"] == 45.0
        assert sensors.SENSOR_SPECS["depth_d405"]["mount_pos"] == [0.33, 0.0, 0.08]
        assert sensors.SENSOR_SPECS["depth_d555"] == old["depth_d555"]
        with pytest.raises(KeyError):
            sensors.apply_mount_overrides({"depth_d405": {"hfov_deg": 10}})
        with pytest.raises(KeyError):
            sensors.apply_mount_overrides({"nope": {"pitch_deg": 1}})
    finally:
        sensors.SENSOR_SPECS.clear()
        sensors.SENSOR_SPECS.update(old)


def test_env_override_reaches_new_process():
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src), "GO2WM_SENSOR_MOUNTS": json.dumps({"depth_d555": {"pitch_deg": 22.0}})}
    out = subprocess.run([sys.executable, "-c", "from go2wm.sim.sensors import SENSOR_SPECS as S; "
                          "print(S['depth_d555']['pitch_deg'])"], env=env, capture_output=True, text=True, check=True)
    assert float(out.stdout.strip()) == 22.0
