import numpy as np

from go2wm.sim.synthetic_camera import CAMERAS, ground_points, measure_crack, project


def test_measure_crack_width_and_length_any_angle():
    res = 0.001
    Y, X = np.mgrid[0:400, 0:400]
    for W in (2, 5, 13):
        for a in (0, 35, 60):
            ang = np.radians(a)
            d = np.abs((X - 200) * np.sin(ang) - (Y - 200) * np.cos(ang))
            t = (X - 200) * np.cos(ang) + (Y - 200) * np.sin(ang)
            m = measure_crack(np.where((d < W / 2) & (np.abs(t) < 150), 0.25, 0.7), res)
            assert abs(m["width_m"] / res - W) <= 1.5 and abs(m["length_m"] / res - 300) <= 30


def test_camera_projection_roundtrip():
    cam = CAMERAS["d555_rgb"]
    base, R = np.array([0.0, 0.0, 0.33]), np.eye(3)
    u = np.array([100.5, 640.5, 1100.5])
    v = np.array([700.5, 500.5, 650.5])
    g, _ = ground_points(cam, u, v, base, R, {"width": 3.0})
    u2, v2 = project(cam, g, base, R)
    assert np.allclose(u, u2, atol=1e-6) and np.allclose(v, v2, atol=1e-6) and np.allclose(g[:, 2], 0.0)
