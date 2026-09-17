"""Exercise capture-to-verification transitions without opening a real camera."""
from argparse import Namespace
from contextlib import ExitStack
from itertools import count
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import app
from core import Store


class WebcamTests(unittest.TestCase):
    def run_session(self, keys, store, camera_open=True):
        args = Namespace(command='enroll', camera=0, name='Demo', bio='Participant',
                         consent=True, threshold=.55, margin=.08)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = camera_open
        camera.read.return_value = (True, frame)
        engine = Mock()
        engine.faces.return_value = [np.array([100, 100, 100, 100])]
        engine.feature.return_value = np.array([1., 0.], dtype=np.float32)
        with ExitStack() as stack:
            stack.enter_context(patch('app.Engine', return_value=engine))
            stack.enter_context(patch('app.cv2.VideoCapture', return_value=camera))
            stack.enter_context(patch('app.cv2.resize', return_value=frame))
            stack.enter_context(patch('app.cv2.waitKey', side_effect=keys))
            stack.enter_context(patch('app.cv2.getWindowProperty', return_value=1))
            stack.enter_context(patch('app.time.monotonic', side_effect=count(2, 2)))
            stack.enter_context(patch('builtins.print'))
            renderer = stack.enter_context(patch('app.cv2.putText'))
            cleanup = stack.enter_context(patch('app.cv2.destroyAllWindows'))
            stack.enter_context(patch('app.cv2.rectangle'))
            stack.enter_context(patch('app.cv2.imshow'))
            lookup = stack.enter_context(patch('app.identify', wraps=app.identify))
            try:
                app.webcam(args, store)
            finally:
                camera.release.assert_called_once()
                cleanup.assert_called_once()
        return camera, renderer, lookup

    def test_fifth_capture_saves_once_then_verifies_until_quit(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'profiles.db')
            try:
                old_id = store.add('Old demo', '', [[1, 0]] * 5, True)
                camera, renderer, lookup = self.run_session([32] * 6 + [ord('q')], store)
                self.assertEqual(camera.read.call_count, 7)
                self.assertEqual(len(store.profiles()), 2)
                self.assertEqual(lookup.call_count, 2)
                for call in lookup.call_args_list:
                    selected = call.args[1]
                    self.assertEqual(len(selected), 1)
                    self.assertNotEqual(selected[0]['id'], old_id)
                labels = [call.args[1] for call in renderer.call_args_list]
                self.assertTrue(any('Profile saved' in label for label in labels))
                self.assertTrue(any('Demo | similarity' in label for label in labels))
            finally:
                store.close()

    def test_quit_after_first_capture_discards_incomplete_enrollment(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'profiles.db')
            try:
                camera, _, lookup = self.run_session([32, ord('q')], store)
                self.assertEqual(camera.read.call_count, 2)
                self.assertEqual(store.profiles(), [])
                lookup.assert_not_called()
            finally:
                store.close()

    def test_unavailable_camera_still_releases_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'profiles.db')
            try:
                with self.assertRaisesRegex(ValueError, 'Cannot open webcam'):
                    self.run_session([], store, camera_open=False)
                self.assertEqual(store.profiles(), [])
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
