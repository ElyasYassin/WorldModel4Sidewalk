import pytest

from go2wm.data.recorder import collect_episodes
from go2wm.sim.mock import MockGo2Sim

LIMITS = {"vx": [-0.4, 1.0], "vy": [-0.4, 0.4], "yaw_rate": [-1.2, 1.2]}


@pytest.fixture(scope="session")
def mock_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("mockdata")
    collect_episodes(MockGo2Sim(), root, num_episodes=12, base_seed=100, max_steps=120, limits=LIMITS, log=lambda s: None)
    return root
