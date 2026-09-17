import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from core import Store, identify, normalize


def profile(identity, vector):
    return dict(id=identity, embeddings=json.dumps([vector] * 5))


class MatchingTests(unittest.TestCase):
    def test_unknown_and_empty(self):
        self.assertEqual(identify([1, 0], []), (None, None))
        self.assertIsNone(identify([0, 1], [profile('a', [1, 0])])[0])

    def test_match(self):
        self.assertEqual(identify([1, 0], [profile('a', [1, 0]), profile('b', [0, 1])])[0]['id'], 'a')

    def test_ambiguous(self):
        self.assertIsNone(identify([1, 0], [profile('a', [1, 0]), profile('b', [.999, .0447])])[0])

    def test_invalid_vectors(self):
        for vector in ([0, 0], [float('nan'), 1], [float('inf'), 0]):
            with self.assertRaises(ValueError):
                normalize(vector)

    def test_database_consent_roundtrip_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'profiles.db')
            try:
                with self.assertRaises(ValueError):
                    store.add('Test', '', [[1, 0]] * 5, False)
                with self.assertRaises(ValueError):
                    store.add('Test', '', [[1, 0]], True)
                identity = store.add('Test', 'Demo participant', [[2, 0]] * 5, True)
                self.assertEqual(identify([1, 0], store.profiles())[0]['id'], identity)
                self.assertTrue(store.profiles()[0]['consent_at'])
                self.assertEqual(store.delete(identity), 1)
                self.assertEqual(store.profiles(), [])
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
