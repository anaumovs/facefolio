import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from core import Store, identify, validate_student


class StudentStoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / 'profiles.db'
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.folder.cleanup()

    def test_names_and_ascii_seven_digit_id(self):
        self.assertEqual(validate_student(' Ana ', ' García ', '0123456'), ('Ana', 'García', '0123456'))
        for bad_id in ('123456', '12345678', '123a456', '１２３４５６７', ' 1234567', '', '1234567\n', 1234567):
            with self.subTest(bad_id=bad_id), self.assertRaises(ValueError):
                validate_student('Ana', 'García', bad_id)
        for first, last in (('', 'Smith'), ('Ana', ' '), ('a' * 81, 'Smith'), ('Ana\n', '\tSmith')):
            # Leading and trailing whitespace is intentionally trimmed.
            if first == 'Ana\n':
                first = 'An\na'
            with self.subTest(first=first, last=last), self.assertRaises(ValueError):
                validate_student(first, last, '1234567')

    def test_capture_roundtrip_unique_id_and_delete(self):
        photo = b'captured-photo-test-bytes'
        identity = self.store.add_student('Ana', 'García', '0123456', [1, 0], photo, True)
        self.store.close()
        self.store = Store(self.path)
        profile = self.store.profiles()[0]
        self.assertEqual(profile['photo_jpeg'], photo)
        self.assertEqual(profile['student_id'], '0123456')
        self.assertEqual(profile['name'], 'Ana García')
        self.assertEqual(len(json.loads(profile['embeddings'])), 1)
        self.assertEqual(identify([1, 0], [profile])[0]['id'], identity)
        with self.assertRaisesRegex(ValueError, 'already has a profile'):
            self.store.add_student('Another', 'Student', '0123456', [0, 1], b'other-photo', True)
        self.assertEqual(len(self.store.profiles()), 1)
        self.store.delete(identity)
        self.assertEqual(self.store.profiles(), [])
        self.assertEqual(self.store.db.execute('SELECT count(photo_jpeg) FROM profiles').fetchone()[0], 0)

    def test_invalid_capture_and_no_consent_do_not_save(self):
        for embedding, photo, consent in (([1, 0], b'', True), ([0, 0], b'photo', True), ([1, 0], b'photo', False)):
            with self.assertRaises(ValueError):
                self.store.add_student('Ana', 'Smith', '1234567', embedding, photo, consent)
        self.assertEqual(self.store.profiles(), [])

    def test_migration_preserves_old_profiles(self):
        old_path = Path(self.folder.name) / 'old.db'
        with sqlite3.connect(old_path) as db:
            db.execute('CREATE TABLE profiles(id TEXT PRIMARY KEY,name TEXT NOT NULL,bio TEXT NOT NULL,consent_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,embeddings TEXT NOT NULL)')
            db.execute('INSERT INTO profiles(id,name,bio,embeddings) VALUES(?,?,?,?)', ('old-id', 'Existing Person', 'Original bio', '[[1,0]]'))
        db.close()
        migrated = Store(old_path)
        try:
            old = migrated.profiles()[0]
            self.assertEqual(old['id'], 'old-id')
            self.assertEqual(old['bio'], 'Original bio')
            self.assertEqual(old['embeddings'], '[[1,0]]')
            self.assertIsNone(old['student_id'])
            self.assertIsNone(old['photo_jpeg'])
            migrated.add_student('New', 'Student', '0000123', [0, 1], b'photo', True)
            self.assertEqual(len(migrated.profiles()), 2)
        finally:
            migrated.close()


if __name__ == '__main__':
    unittest.main()
