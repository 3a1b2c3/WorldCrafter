import unittest

import numpy as np

from demo.camera import Camera, ControlBuffer


class DemoControlsTest(unittest.TestCase):
    def test_defaults_and_slider_bounds(self):
        controls = ControlBuffer()
        self.assertEqual((controls.speed, controls.vertical_speed, controls.rotation_angle), (2, 2, 30))
        self.assertEqual(controls.orbit_radius, 1)
        for setting, low, high in (
            ("speed", 1, 5), ("vertical_speed", 1, 5),
            ("rotation_angle", 10, 45), ("orbit_radius", 1, 5),
        ):
            for value, expected in ((-1, low), (100, high)):
                controls.update(dict(type=setting, value=value))
                self.assertEqual(getattr(controls, setting), expected)

    def test_modes_separate_translation_and_preserve_orbit_pivot(self):
        for radius in (1, 5):
            controls = ControlBuffer()
            controls.update(dict(type="orbit_radius", value=radius))
            camera = Camera()
            for mode in ("look", "orbit", "look"):
                controls.update(dict(type="rotation_mode", value=mode))
                center = camera.world[:3, 3] + radius * camera.world[:3, 2]
                for key in ("arrowright", "arrowup", "arrowleft", "arrowdown"):
                    controls.clear()
                    controls.update(dict(type="key", key=key, down=True))
                    start = camera.world.copy()
                    action = controls.consume()
                    self.assertEqual(action.orbit, mode == "orbit")
                    self.assertEqual(action.orbit_radius, radius if mode == "orbit" else 0)
                    _, world = camera.append(action)
                    poses = world[-33:].astype(np.float64)
                    np.testing.assert_allclose(poses[0], start[:3], atol=1e-6)
                    if mode == "look":
                        np.testing.assert_allclose(poses[:, :, 3], np.broadcast_to(start[:3, 3], (33, 3)), atol=1e-6)
                    else:
                        np.testing.assert_allclose(poses[:, :, 3] + radius * poses[:, :, 2], np.broadcast_to(center, (33, 3)), atol=1e-6)
                        np.testing.assert_allclose(np.linalg.norm(poses[:, :, 3] - center, axis=1), radius, atol=1e-6)
                        self.assertGreater(np.linalg.norm(camera.world[:3, 3] - start[:3, 3]), 0.1)

    def test_45_degree_rotation_reaches_camera(self):
        for mode in ("look", "orbit"):
            for key in ("arrowright", "arrowup"):
                controls = ControlBuffer()
                controls.update(dict(type="rotation_mode", value=mode))
                controls.update(dict(type="rotation_angle", value=45))
                controls.update(dict(type="key", key=key, down=True))
                action = controls.consume()
                self.assertEqual(action.yaw or action.pitch, 45)
                camera = Camera()
                local, global_poses = camera.append(action)
                self.assertEqual(global_poses.shape, (33, 3, 4))
                self.assertTrue(np.isfinite(local).all())
                # The endpoint advances by 45 degrees, despite exclusive sampling.
                angle = np.degrees(np.arccos(np.clip((np.trace(camera.world[:3, :3]) - 1) / 2, -1, 1)))
                self.assertAlmostEqual(angle, 45)


if __name__ == "__main__":
    unittest.main()
