import numpy as np

from go2wm.common.frames import yaw_to_quat
from go2wm.common.schema import STATE_DIM, Observation, VelocityCommand, state_pose2d


def test_state_vector_layout_and_pose():
    o = Observation(
        timestamp_ns=1,
        pos_w=np.array([1.0, 2.0, 0.3]),
        quat_w=yaw_to_quat(0.7),
        lin_vel_b=np.array([0.5, 0.0, 0.0]),
        ang_vel_b=np.zeros(3),
        gravity_b=np.array([0, 0, -1.0]),
    )
    s = o.state_vector()
    assert s.shape == (STATE_DIM,)
    assert np.allclose(state_pose2d(s), [1.0, 2.0, 0.7])
    assert np.allclose(o.pose2d, [1.0, 2.0, 0.7])


def test_velocity_command_roundtrip():
    c = VelocityCommand.from_action([0.3, -0.1, 0.5], 0.1)
    assert np.allclose(c.as_array(), [0.3, -0.1, 0.5, 0.1])
    assert np.allclose(c.action, [0.3, -0.1, 0.5])
