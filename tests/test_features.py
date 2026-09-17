"""Persistence, measured evaluation, enrollment and directory regression coverage."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
from core import Store, StudentIDError
from ui import Capture, capture_face
from enrollment import EnrollmentDialog
from directory import DirectoryPage
from profile_form import ProfileDialog
from desktop import WebcamWindow
from evaluation import TestPhoto, assess_photo, build_report, enrollment_hashes, load_manifest


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / 'profiles.db')
        self.vector = np.array([1., 0.], dtype=np.float32)
        self.face = np.array([100, 100, 150, 170], dtype=np.float32)
        self.samples = []
        for color in range(5):
            frame = np.full((480, 640, 3), 50 + color * 20, np.uint8)
            self.samples.append(capture_face(frame, self.face, self.vector))
        self.frame = np.full((480, 640, 3), 200, np.uint8)
        self.engine = Mock()
        self.engine.faces.return_value = [self.face]
        self.engine.feature.return_value = self.vector
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.camera.read.return_value = (True, self.frame)
        self.args = Namespace(command='recognize', camera=0, threshold=.55, margin=.08)
        self.widgets = []

    def tearDown(self):
        for widget in self.widgets:
            widget.close()
        self.store.close()
        self.temp.cleanup()

    def add_student(self, first='Ana', student_id='0123456'):
        return self.store.add_student_samples(first, 'Smith', student_id, [(s.embedding, s.photo_jpeg) for s in self.samples], True)

    def photo(self, name, color=210):
        path = self.root / name
        cv2.imwrite(str(path), np.full((480, 640, 3), color, np.uint8))
        return path

    def test_five_samples_persist_and_delete_cascades(self):
        identity = self.add_student()
        self.assertEqual(len(json.loads(self.store.profiles()[0]['embeddings'])), 5)
        self.assertEqual(self.store.photos(identity), [s.photo_jpeg for s in self.samples])
        self.store.close()
        self.store = Store(self.root / 'profiles.db')
        self.assertEqual(len(self.store.photos(identity)), 5)
        self.store.delete(identity)
        self.assertEqual(self.store.photos(identity), [])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM profile_photos').fetchone()[0], 0)

    def test_replace_samples_and_edit_are_atomic(self):
        identity = self.add_student()
        second = self.add_student('Other', '1234567')
        with self.assertRaises(StudentIDError):
            self.store.update_student(identity, 'Changed', 'Name', '1234567')
        profile = next(p for p in self.store.profiles() if p['id'] == identity)
        self.assertEqual(profile['name'], 'Ana Smith')
        with self.assertRaises(ValueError):
            self.store.replace_student_samples(identity, [(self.vector, b'photo')])
        self.assertEqual(self.store.photos(identity), [s.photo_jpeg for s in self.samples])
        reversed_samples = list(reversed(self.samples))
        self.store.replace_student_samples(identity, [(s.embedding, s.photo_jpeg) for s in reversed_samples])
        self.assertEqual(self.store.photos(identity)[0], self.samples[-1].photo_jpeg)
        self.store.update_student(identity, 'Ana', 'García', '0000123')
        self.assertEqual(next(p for p in self.store.profiles() if p['id'] == identity)['student_id'], '0000123')
        self.assertEqual(len(self.store.photos(second)), 5)

    def test_directory_search_edit_and_delete_with_cancel(self):
        identity = self.add_student()
        self.add_student('Ben', '7654321')
        directory = DirectoryPage(self.store)
        self.widgets.append(directory)
        directory.search.setText('0123')
        self.assertEqual(directory.table.rowCount(), 1)
        directory.table.selectRow(0)
        self.assertEqual(directory.identity, identity)
        self.assertFalse(directory.thumbnails[4].pixmap().isNull())
        def edit(dialog):
            dialog.last_name.setText('García')
            dialog.save_profile()
            return dialog.result()
        with patch.object(ProfileDialog, 'exec', edit):
            directory.edit_selected()
        self.assertIn('García', directory.details.text())
        with patch.object(QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Cancel):
            directory.delete_selected()
        self.assertEqual(len(self.store.profiles()), 2)
        with patch.object(QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Yes):
            directory.delete_selected()
        self.assertEqual(len(self.store.profiles()), 1)
        self.assertEqual(self.store.photos(identity), [])
        self.assertFalse(directory.delete_button.isEnabled())

    def test_enrollment_requires_five_and_can_retake_or_cancel(self):
        dialog = EnrollmentDialog(self.engine, self.camera)
        self.widgets.append(dialog)
        dialog.timer.stop()
        dialog.finish()
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
        for index in range(5):
            self.camera.read.return_value = (True, np.full((480, 640, 3), 50 + index * 20, np.uint8))
            with patch('enrollment.time.monotonic', return_value=10 + index * 2):
                dialog.tick()
                self.assertTrue(dialog.capture_button.isEnabled())
                dialog.capture_next()
        self.assertEqual(len(dialog.samples), 5)
        self.assertTrue(dialog.continue_button.isEnabled())
        dialog.retake()
        self.assertEqual(len(dialog.samples), 4)
        self.assertFalse(dialog.continue_button.isEnabled())
        dialog.reject()
        self.assertEqual(self.store.profiles(), [])
        self.camera.release.assert_not_called()

    def test_enrollment_rejects_switched_person_and_duplicate_frame(self):
        dialog = EnrollmentDialog(self.engine, self.camera, initial=self.samples[0])
        self.widgets.append(dialog)
        dialog.timer.stop()
        dialog.last_capture = 0
        self.engine.feature.return_value = np.array([0., 1.], dtype=np.float32)
        dialog.tick()
        self.assertFalse(dialog.capture_button.isEnabled())
        self.assertIn('original person', dialog.message.text())
        self.engine.feature.return_value = self.vector
        self.camera.read.return_value = (True, np.full((480, 640, 3), 50, np.uint8))
        dialog.tick()
        dialog.capture_next()
        self.assertEqual(len(dialog.samples), 1)
        self.assertIn('repeats', dialog.message.text())

    def test_evaluation_counts_all_outcomes_without_inflating_accuracy(self):
        identity = self.add_student()
        results = [dict(known=known, outcome=outcome, latency_ms=10 + i) for i, (known, outcome) in enumerate([
            (True, 'correct_identity'), (True, 'wrong_identity'), (True, 'missed_known'),
            (True, 'rejected_capture'), (False, 'correct_rejection'), (False, 'false_accept'), (False, 'rejected_capture')])]
        report = build_report(results, self.store.profiles(), .55, .08)
        self.assertAlmostEqual(report['correct_decision_rate'], 2 / 7)
        self.assertEqual(report['known_identification_rate'], .25)
        self.assertEqual(report['known_miss_rate'], .5)
        self.assertEqual(report['unknown_false_match_rate'], .5)
        self.assertEqual(report['valid_unknown'], 2)
        self.assertEqual(report['median_latency_ms'], 13)
        self.assertNotIn(identity, json.dumps(report))
        no_unknown = build_report(results[:1], self.store.profiles(), .55, .08)
        self.assertIsNone(no_unknown['unknown_false_match_rate'])

    def test_real_image_evaluation_rejects_enrollment_leak_and_invalid_capture(self):
        identity = self.add_student()
        profiles = self.store.profiles()
        path = self.photo('held-out.png')
        result = assess_photo(TestPhoto(path, identity), self.engine, profiles, .55, .08)
        self.assertEqual(result['outcome'], 'correct_identity')
        self.engine.faces.return_value = []
        result = assess_photo(TestPhoto(path, ''), self.engine, profiles, .55, .08)
        self.assertEqual(result['outcome'], 'rejected_capture')
        leak = self.root / 'enrollment.jpg'
        leak.write_bytes(self.samples[0].photo_jpeg)
        with self.assertRaisesRegex(ValueError, 'enrollment photo'):
            assess_photo(TestPhoto(leak, identity), self.engine, profiles, .55, .08, enrollment_hashes(self.store, profiles))

    def test_tabs_pause_camera_and_directory_changes_refresh_matching(self):
        identity = self.add_student()
        window = WebcamWindow(self.args, self.store, self.engine, self.camera, autostart=False)
        self.widgets.append(window)
        window.tick()
        self.assertEqual(window.first_name.text(), 'Ana')
        window.tabs.setCurrentIndex(1)
        self.assertFalse(window.timer.isActive())
        self.assertFalse(window.quit_shortcut.isEnabled())
        self.store.delete(identity)
        window.directory.changed.emit()
        self.assertEqual(window.profiles, [])
        self.assertEqual(window.first_name.text(), '—')
        window.tabs.setCurrentIndex(0)
        self.assertTrue(window.timer.isActive())

    def test_manifest_checks_header_and_profile_labels(self):
        identity = self.add_student()
        csv = self.root / 'test.csv'
        csv.write_text(f'path,expected_id\nknown.jpg,{identity}\nunknown.jpg,\n')
        self.assertEqual(len(load_manifest(csv, self.store.profiles())), 2)
        csv.write_text('path,expected_id\nphoto.jpg,bogus\n')
        with self.assertRaises(ValueError):
            load_manifest(csv, self.store.profiles())


if __name__ == '__main__':
    unittest.main()
