import unittest

import numpy as np

from demo.camera import Camera, ControlBuffer


class DemoControlsTest(unittest.TestCase):
    def test_defaults_and_slider_bounds(self):
        controls = ControlBuffer()
        self.assertEqual((controls.speed, controls.vertical_speed, controls.rotation_angle), (2, 2, 30))
        for setting, low, high in (
            ("speed", 1, 5), ("vertical_speed", 1, 5),
            ("rotation_angle", 10, 45), ("orbit_radius", 0, 5),
        ):
            for value, expected in ((-1, low), (100, high)):
                controls.update(dict(type=setting, value=value))
                self.assertEqual(getattr(controls, setting), expected)

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
