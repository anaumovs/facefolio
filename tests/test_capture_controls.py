"""Real mouse/key events, including a camera refresh between press and release."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import Qt, QEvent, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLineEdit
from core import Store
from desktop import WebcamWindow
from directory import DirectoryPage
from enrollment import EnrollmentDialog
from ui import capture_face


class CaptureControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / 'profiles.db')
        self.frame = np.full((480, 640, 3), 120, np.uint8)
        self.face = np.array([100, 100, 160, 180], np.float32)
        self.vector = np.array([1., 0.], np.float32)
        self.camera, self.engine = Mock(), Mock()
        self.camera.isOpened.return_value = True
        self.camera.read.return_value = (True, self.frame)
        self.engine.faces.return_value = [self.face]
        self.engine.feature.return_value = self.vector
        self.args = Namespace(command='recognize', camera=0, threshold=.55, margin=.08)
        self.widgets = []

    def tearDown(self):
        for widget in self.widgets:
            widget.close()
        self.store.close()
        self.temp.cleanup()

    def show(self, widget):
        self.widgets.append(widget)
        widget.show()
        widget.activateWindow()
        self.qt.processEvents()
        if hasattr(widget, 'timer'):
            widget.timer.stop()
        return widget

    def dialog(self):
        return self.show(EnrollmentDialog(self.engine, self.camera))

    def add_student(self):
        photo = capture_face(self.frame, self.face, self.vector)
        return self.store.add_student('Ana', 'Smith', '0123456', self.vector, photo.photo_jpeg, True)

    def test_mouse_capture_survives_camera_refresh_while_pressed(self):
        dialog = self.dialog()
        dialog.tick()
        QTest.mousePress(dialog.capture_button, Qt.MouseButton.LeftButton)
        self.assertTrue(dialog.capture_button.isDown())
        dialog.tick()
        self.assertTrue(dialog.capture_button.isDown())
        QTest.mouseRelease(dialog.capture_button, Qt.MouseButton.LeftButton)
        self.assertEqual(len(dialog.samples), 1)
        self.assertIn('Photo 1 of 5 captured', dialog.message.text())

    def test_live_camera_capture_survives_refresh_while_pressed(self):
        window = self.show(WebcamWindow(self.args, self.store, self.engine, self.camera, autostart=False))
        window.tick()
        QTest.mousePress(window.capture_button, Qt.MouseButton.LeftButton)
        window.tick()
        self.assertTrue(window.capture_button.isDown())
        with patch('desktop.EnrollmentDialog') as enrollment:
            enrollment.return_value.exec.return_value = QDialog.DialogCode.Rejected
            QTest.mouseRelease(window.capture_button, Qt.MouseButton.LeftButton)
            enrollment.assert_called_once()
        self.assertEqual(self.store.profiles(), [])

    def test_c_captures_five_photos_and_retakes_last_without_extra_sample(self):
        dialog = self.dialog()
        for i in range(5):
            self.camera.read.return_value = (True, np.full((480, 640, 3), 40 + 20 * i, np.uint8))
            with patch('enrollment.time.monotonic', return_value=10 + 2 * i):
                dialog.tick()
                QTest.keyClick(dialog, Qt.Key.Key_C)
            self.assertEqual(len(dialog.samples), i + 1)
        original = [s.photo_jpeg for s in dialog.samples]
        self.assertTrue(dialog.continue_button.isEnabled())
        self.camera.read.return_value = (True, np.full((480, 640, 3), 200, np.uint8))
        with patch('enrollment.time.monotonic', return_value=30):
            dialog.tick()
            self.assertTrue(dialog.capture_button.isEnabled())
            QTest.keyClick(dialog, Qt.Key.Key_C)
        self.assertEqual(len(dialog.samples), 5)
        self.assertEqual([s.photo_jpeg for s in dialog.samples[:4]], original[:4])
        self.assertNotEqual(dialog.samples[-1].photo_jpeg, original[-1])
        self.assertIn('Photo 5 retaken', dialog.message.text())
        self.assertEqual(self.store.profiles(), [])

    def test_c_does_not_bypass_invalid_frame_or_cooldown_or_repeat(self):
        dialog = self.dialog()
        self.engine.faces.return_value = []
        dialog.tick()
        QTest.keyClick(dialog, Qt.Key.Key_C)
        self.assertEqual(len(dialog.samples), 0)
        self.engine.faces.return_value = [self.face]
        with patch('enrollment.time.monotonic', return_value=10):
            dialog.tick()
            QTest.keyClick(dialog, Qt.Key.Key_C)
            QTest.keyClick(dialog, Qt.Key.Key_C)
        self.assertEqual(len(dialog.samples), 1)
        self.camera.read.return_value = (True, self.frame + 1)
        with patch('enrollment.time.monotonic', return_value=10.5):
            dialog.tick()
            QTest.keyClick(dialog, Qt.Key.Key_C)
        self.assertEqual(len(dialog.samples), 1)
        with patch('enrollment.time.monotonic', return_value=20):
            dialog.tick()
            event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier, 'c', True)
            self.qt.sendEvent(dialog, event)
        self.assertEqual(len(dialog.samples), 1)
        self.assertTrue(dialog.keyboard_filter_installed)

    def test_directory_modal_capture_works_with_child_focus_and_shift_c(self):
        identity = self.add_student()
        window = self.show(WebcamWindow(self.args, self.store, self.engine, self.camera, autostart=False))
        window.tabs.setCurrentIndex(1)
        window.directory.table.selectRow(0)
        window.directory.table.setFocus()
        reference = next(p for p in self.store.profiles() if p['id'] == identity)
        dialog = EnrollmentDialog(self.engine, self.camera, parent=window, reference=reference)
        self.widgets.append(dialog)
        failures = []
        def exercise():
            try:
                dialog.timer.stop()
                cancel = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Cancel)
                for i, target in enumerate((cancel, dialog.capture_button, dialog.retake_button, cancel, dialog.capture_button)):
                    self.camera.read.return_value = (True, np.full((480, 640, 3), 40 + 20 * i, np.uint8))
                    with patch('enrollment.time.monotonic', return_value=100 + 2 * i):
                        dialog.tick()
                        target.setFocus()
                        QTest.keyClick(target, Qt.Key.Key_C, Qt.KeyboardModifier.ShiftModifier if i % 2 else Qt.KeyboardModifier.NoModifier)
                    self.assertEqual(len(dialog.samples), i + 1)
                self.assertTrue(dialog.continue_button.isEnabled())
            except BaseException as error:
                failures.append(error)
            finally:
                dialog.reject()
        QTimer.singleShot(0, exercise)
        dialog.exec()
        self.assertFalse(dialog.keyboard_filter_installed)
        if failures:
            raise failures[0]
        self.assertEqual(len(dialog.samples), 5)
        self.assertEqual(len(self.store.photos(identity)), 1)

    def test_capture_filter_does_not_swallow_other_window_text(self):
        dialog = self.dialog()
        field = self.show(QLineEdit())
        QTest.keyClicks(field, 'Cameron')
        self.assertEqual(field.text(), 'Cameron')
        self.assertEqual(len(dialog.samples), 0)

    def test_only_camera_and_directory_tabs_remain(self):
        window = self.show(WebcamWindow(self.args, self.store, self.engine, self.camera, autostart=False))
        self.assertEqual([window.tabs.tabText(i) for i in range(window.tabs.count())], ['Live camera', 'Student directory'])

    def test_r_removes_last_and_c_captures_replacement(self):
        dialog = self.dialog()
        dialog.tick()
        QTest.keyClick(dialog, Qt.Key.Key_C)
        QTest.keyClick(dialog, Qt.Key.Key_R)
        self.assertEqual(len(dialog.samples), 0)
        self.camera.read.return_value = (True, self.frame + 20)
        dialog.last_capture = 0
        dialog.tick()
        QTest.keyClick(dialog, Qt.Key.Key_C)
        self.assertEqual(len(dialog.samples), 1)

    def test_directory_c_opens_selected_profile_but_search_keeps_letters(self):
        identity = self.add_student()
        directory = self.show(DirectoryPage(self.store))
        requested = Mock()
        directory.recapture.connect(requested)
        directory.table.selectRow(0)
        directory.table.setFocus()
        QTest.keyClick(directory.table, Qt.Key.Key_C)
        requested.assert_called_once_with(identity)
        directory.search.setFocus()
        QTest.keyClicks(directory.search, 'c')
        self.assertEqual(directory.search.text(), 'c')
        self.assertEqual(requested.call_count, 1)

    def test_c_on_recognized_student_opens_photo_update(self):
        identity = self.add_student()
        window = self.show(WebcamWindow(self.args, self.store, self.engine, self.camera, autostart=False))
        window.tick()
        self.assertEqual(window.capture_target_id, identity)
        with patch('desktop.EnrollmentDialog') as enrollment:
            enrollment.return_value.exec.return_value = QDialog.DialogCode.Rejected
            QTest.keyClick(window, Qt.Key.Key_C)
            self.assertEqual(enrollment.call_args.kwargs['reference']['id'], identity)
        self.assertTrue(window.capture_shortcut.isEnabled())
        self.assertEqual(len(self.store.profiles()), 1)

    def test_parent_capture_shortcut_is_disabled_inside_modal(self):
        window = self.show(WebcamWindow(self.args, self.store, self.engine, self.camera, autostart=False))
        window.tick()
        def cancel_dialog():
            self.assertFalse(window.capture_shortcut.isEnabled())
            return QDialog.DialogCode.Rejected
        with patch('desktop.EnrollmentDialog') as enrollment:
            enrollment.return_value.exec.side_effect = cancel_dialog
            window.capture_profile()
        self.assertTrue(window.capture_shortcut.isEnabled())


if __name__ == '__main__':
    unittest.main()
