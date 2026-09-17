"""Offscreen desktop workflow tests with a simulated camera; no biometric test data."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from core import Store
from desktop import Capture, ProfileDialog, WebcamWindow, capture_face


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.folder.name) / 'profiles.db')
        self.frame = np.full((480, 640, 3), [80, 100, 160], dtype=np.uint8)
        self.face = np.array([160, 100, 160, 180], dtype=np.float32)
        self.vector = np.array([1., 0.], dtype=np.float32)
        self.capture = capture_face(self.frame, self.face, self.vector)
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.camera.read.return_value = (True, self.frame)
        self.engine = Mock()
        self.engine.faces.return_value = [self.face]
        self.engine.feature.return_value = self.vector
        self.args = Namespace(command='recognize', camera=0, threshold=.55, margin=.08)
        self.window = None

    def tearDown(self):
        if self.window:
            self.window.close()
        self.store.close()
        self.folder.cleanup()

    def make_window(self):
        self.window = WebcamWindow(self.args, self.store, self.engine, self.camera, autostart=False)
        return self.window

    def test_photo_is_real_jpeg_and_immutable_snapshot(self):
        original = self.capture.photo_jpeg
        decoded = cv2.imdecode(np.frombuffer(original, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(decoded.shape[0], 0)
        self.frame[:] = 0
        self.vector[:] = 0
        self.assertEqual(self.capture.photo_jpeg, original)
        np.testing.assert_array_equal(self.capture.embedding, [1., 0.])

    def test_dialog_validation_cancel_and_save(self):
        dialog = ProfileDialog(self.capture, self.store)
        dialog.save_profile()
        self.assertIn('first name', dialog.error.text())
        self.assertEqual(self.store.profiles(), [])
        dialog.first_name.setText('Ana')
        dialog.last_name.setText('García')
        dialog.student_id.setText('123')
        dialog.save_profile()
        self.assertIn('exactly 7', dialog.error.text())
        self.assertEqual(self.store.profiles(), [])
        dialog.student_id.setText('0123456')
        dialog.save_profile()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(self.store.profiles()[0]['student_id'], '0123456')
        duplicate = ProfileDialog(self.capture, self.store)
        duplicate.first_name.setText('Ana')
        duplicate.last_name.setText('Other')
        duplicate.student_id.setText('0123456')
        duplicate.save_profile()
        self.assertIn('already has a profile', duplicate.error.text())
        self.assertIsNone(duplicate.profile_id)
        duplicate.reject()
        self.assertEqual(len(self.store.profiles()), 1)

    def test_empty_gallery_capture_save_resume_and_recognize(self):
        window = self.make_window()
        window.tick()
        self.assertTrue(window.capture_button.isEnabled())
        self.assertIn('Unknown', window.status.text())

        def save_dialog(dialog):
            self.assertFalse(window.timer.isActive())
            dialog.first_name.setText('Ana')
            dialog.last_name.setText('Smith')
            dialog.student_id.setText('0000123')
            dialog.save_profile()
            return dialog.result()

        with patch('desktop.EnrollmentDialog') as enrollment, patch.object(ProfileDialog, 'exec', save_dialog):
            enrollment.return_value.exec.return_value = QDialog.DialogCode.Accepted
            enrollment.return_value.samples = [self.capture] * 5
            window.capture_profile()
        self.assertTrue(window.timer.isActive())
        self.assertEqual(window.first_name.text(), 'Ana')
        self.assertEqual(window.last_name.text(), 'Smith')
        self.assertEqual(window.student_id.text(), '0000123')
        self.assertFalse(window.photo.pixmap().isNull())
        window.tick()
        self.assertIn('Profile found', window.status.text())
        self.assertTrue(window.capture_button.isEnabled())
        self.assertIn('Update 5', window.capture_button.text())
        self.assertEqual(len(self.store.profiles()), 1)

    def test_invalid_id_is_not_filtered_and_requires_reentry_before_save(self):
        dialog = ProfileDialog(self.capture, self.store)
        dialog.first_name.setText('Ana')
        dialog.last_name.setText('Smith')
        dialog.show()
        self.qt.processEvents()
        save = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save)
        try:
            for invalid in ('', '123456', '12345678', '123a4567', '1234567 '):
                with self.subTest(invalid=invalid):
                    dialog.student_id.clear()
                    QTest.keyClicks(dialog.student_id, invalid)
                    self.assertEqual(dialog.student_id.text(), invalid)
                    QTest.mouseClick(save, Qt.MouseButton.LeftButton)
                    self.assertTrue(dialog.isVisible())
                    self.assertIsNone(dialog.profile_id)
                    self.assertEqual(self.store.profiles(), [])
                    self.assertIn('Please re-enter a valid student ID', dialog.error.text())
                    self.assertIs(dialog.focusWidget(), dialog.student_id)
                    self.assertEqual(dialog.student_id.selectedText(), invalid)
                    self.assertTrue(dialog.student_id.property('invalid'))
                    self.assertEqual(dialog.first_name.text(), 'Ana')
                    self.assertEqual(dialog.last_name.text(), 'Smith')
                    self.assertEqual(dialog.capture.photo_jpeg, self.capture.photo_jpeg)
            # Typing replaces the selected invalid entry; only a new Save accepts it.
            QTest.keyClicks(dialog.student_id, '0123456')
            self.assertEqual(self.store.profiles(), [])
            QTest.mouseClick(save, Qt.MouseButton.LeftButton)
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertFalse(dialog.student_id.property('invalid'))
            self.assertEqual(self.store.profiles()[0]['student_id'], '0123456')
        finally:
            dialog.close()

    def test_duplicate_id_keeps_form_open_and_refocuses_id(self):
        self.store.add_student('Existing', 'Student', '0123456', self.vector, self.capture.photo_jpeg, True)
        dialog = ProfileDialog(self.capture, self.store)
        dialog.first_name.setText('New')
        dialog.last_name.setText('Student')
        dialog.student_id.setText('0123456')
        dialog.show()
        self.qt.processEvents()
        try:
            dialog.save_profile()
            self.assertTrue(dialog.isVisible())
            self.assertIn('Please re-enter', dialog.error.text())
            self.assertIs(dialog.focusWidget(), dialog.student_id)
            self.assertEqual(dialog.student_id.selectedText(), '0123456')
            self.assertIsNone(dialog.profile_id)
            self.assertEqual(len(self.store.profiles()), 1)
        finally:
            dialog.close()

    def test_cancel_keeps_camera_open_without_saving(self):
        window = self.make_window()
        window.tick()
        with patch('desktop.EnrollmentDialog') as enrollment:
            enrollment.return_value.exec.return_value = QDialog.DialogCode.Rejected
            window.capture_profile()
        self.assertTrue(window.timer.isActive())
        self.assertEqual(self.store.profiles(), [])
        self.assertIsNone(window.pending)

    def test_no_face_multiple_faces_and_poor_quality_clear_identity(self):
        self.store.add_student('Ana', 'Smith', '0000123', self.vector, self.capture.photo_jpeg, True)
        window = self.make_window()
        window.tick()
        self.assertEqual(window.first_name.text(), 'Ana')
        for faces in ([], [self.face, self.face]):
            self.engine.faces.return_value = faces
            window.tick()
            self.assertFalse(window.capture_button.isEnabled())
            self.assertEqual(window.first_name.text(), '—')
            self.assertIsNone(window.pending)
        self.engine.faces.return_value = [self.face]
        self.engine.feature.side_effect = ValueError('Move closer')
        window.tick()
        self.assertFalse(window.capture_button.isEnabled())
        self.assertIsNone(window.pending)
        self.assertIn('Move closer', window.status.text())

    def test_camera_loss_clears_profile_and_stops_timer(self):
        self.store.add_student('Ana', 'Smith', '0000123', self.vector, self.capture.photo_jpeg, True)
        window = self.make_window()
        window.tick()
        self.camera.read.return_value = (False, None)
        window.tick()
        self.assertFalse(window.timer.isActive())
        self.assertFalse(window.capture_button.isEnabled())
        self.assertEqual(window.first_name.text(), '—')
        self.camera.release.assert_called()

    def test_legacy_duplicates_do_not_block_new_student_match(self):
        for name in ('Old demo 1', 'Old demo 2'):
            self.store.add(name, '', [self.vector] * 5, True)
        window = self.make_window()
        window.tick()
        self.assertTrue(window.capture_button.isEnabled())
        self.store.add_student('Ana', 'Smith', '0000123', self.vector, self.capture.photo_jpeg, True)
        window.profiles = self.store.profiles()
        window.tick()
        self.assertEqual(window.first_name.text(), 'Ana')
        self.assertEqual(len(self.store.profiles()), 3)


if __name__ == '__main__':
    unittest.main()
