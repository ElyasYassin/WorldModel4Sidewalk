

def test_project_deadband():
    import numpy as np
    from go2wm.planning.local_mpc import project_deadband

    A = np.array([[[0.05, 0.0, 0.8], [0.2, 0.0, 0.3], [0.0, -0.3, 0.0], [0.7, 0.1, -0.5]]])
    P = project_deadband(A, 0.1, 0.4)
    assert np.allclose(P[0, 0], 0.0)  # dead -> stop, yaw too
    assert np.allclose(P[0, 1], [0.4, 0.0, 0.3])  # scaled up, yaw kept
    assert np.allclose(P[0, 2], [0.0, -0.4, 0.0])
    assert np.allclose(P[0, 3], A[0, 3])  # untouched
