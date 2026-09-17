"""Local profile persistence and open-set matching, independent of the webcam."""
import json
import re
import sqlite3
import uuid
from pathlib import Path
import numpy as np


class StudentIDError(ValueError):
    """An ID the user must correct before a profile can be saved."""


def validate_student(first_name, last_name, student_id):
    first_name, last_name = first_name.strip(), last_name.strip()
    if not first_name or not last_name:
        raise ValueError('Enter both a first name and a last name.')
    if any(len(name) > 80 or any(ord(c) < 32 for c in name) for name in (first_name, last_name)):
        raise ValueError('Names must be at most 80 characters and contain no control characters.')
    if not isinstance(student_id, str) or not re.fullmatch(r'[0-9]{7}', student_id):
        raise StudentIDError('Invalid student ID. Please re-enter a valid student ID containing exactly 7 digits (0-9).')
    return first_name, last_name, student_id


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).flatten()
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError('Invalid face embedding')
    return vector / norm


class Store:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA foreign_keys=ON')
        path.chmod(0o600)
        self.db.execute('PRAGMA secure_delete=ON')
        self.db.execute('CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, name TEXT NOT NULL, bio TEXT NOT NULL, consent_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, embeddings TEXT NOT NULL)')
        # Add fields without rewriting or discarding profiles from older versions.
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(profiles)')}
        with self.db:
            for name, kind in (('first_name', 'TEXT'), ('last_name', 'TEXT'),
                               ('student_id', 'TEXT'), ('photo_jpeg', 'BLOB')):
                if name not in columns:
                    self.db.execute(f'ALTER TABLE profiles ADD COLUMN {name} {kind}')
            self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS profiles_student_id_unique ON profiles(student_id) WHERE student_id IS NOT NULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS profile_photos (profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE, position INTEGER NOT NULL, photo_jpeg BLOB NOT NULL, PRIMARY KEY(profile_id,position))')

    def add(self, name, bio, embeddings, consent):
        if not consent or not name.strip() or len(embeddings) < 5:
            raise ValueError('Consent, a name, and at least five samples are required')
        identity = str(uuid.uuid4())
        vectors = [normalize(e).tolist() for e in embeddings]
        with self.db:
            self.db.execute('INSERT INTO profiles(id,name,bio,embeddings) VALUES(?,?,?,?)', (identity, name.strip(), bio, json.dumps(vectors)))
        return identity

    def add_student(self, first_name, last_name, student_id, embedding, photo_jpeg, consent):
        return self._create_student(first_name, last_name, student_id, [(embedding, photo_jpeg)], consent)

    @staticmethod
    def validate_samples(samples, minimum=5):
        if not minimum <= len(samples) <= 10:
            raise ValueError(f'Capture at least {minimum} enrollment photos (maximum 10).')
        vectors, photos = [], []
        for embedding, photo in samples:
            if not isinstance(photo, bytes) or not photo or len(photo) > 5_000_000:
                raise ValueError('A captured face photo is required (maximum 5 MB).')
            vectors.append(normalize(embedding).tolist())
            photos.append(photo)
        if len({len(v) for v in vectors}) != 1:
            raise ValueError('Enrollment samples must use the same face model.')
        return vectors, photos

    def add_student_samples(self, first_name, last_name, student_id, samples, consent):
        self.validate_samples(samples)
        return self._create_student(first_name, last_name, student_id, samples, consent)

    def _create_student(self, first_name, last_name, student_id, samples, consent):
        first_name, last_name, student_id = validate_student(first_name, last_name, student_id)
        if not consent:
            raise ValueError('The participant must agree to local enrollment.')
        vectors, photos = self.validate_samples(samples, minimum=1)
        identity = str(uuid.uuid4())
        try:
            with self.db:
                self.db.execute(
                    'INSERT INTO profiles(id,name,bio,embeddings,first_name,last_name,student_id,photo_jpeg) VALUES(?,?,?,?,?,?,?,?)',
                    (identity, f'{first_name} {last_name}', '', json.dumps(vectors),
                     first_name, last_name, student_id, photos[0]))
                self.db.executemany('INSERT INTO profile_photos VALUES(?,?,?)',
                                    [(identity, i, photo) for i, photo in enumerate(photos[1:], 1)])
        except sqlite3.IntegrityError as error:
            if self.db.execute('SELECT 1 FROM profiles WHERE student_id=?', (student_id,)).fetchone():
                raise StudentIDError('That student ID already has a profile. Please re-enter this student’s correct 7-digit ID or use their existing profile.') from error
            raise
        return identity

    def update_student(self, identity, first_name, last_name, student_id):
        first_name, last_name, student_id = validate_student(first_name, last_name, student_id)
        try:
            with self.db:
                count = self.db.execute('UPDATE profiles SET name=?,first_name=?,last_name=?,student_id=? WHERE id=? AND student_id IS NOT NULL',
                                       (f'{first_name} {last_name}', first_name, last_name, student_id, identity)).rowcount
                if not count:
                    raise ValueError('Student profile no longer exists.')
        except sqlite3.IntegrityError as error:
            raise StudentIDError('That student ID already has a profile. Please re-enter this student’s correct 7-digit ID.') from error

    def replace_student_samples(self, identity, samples):
        vectors, photos = self.validate_samples(samples)
        with self.db:
            count = self.db.execute('UPDATE profiles SET embeddings=?,photo_jpeg=? WHERE id=? AND student_id IS NOT NULL',
                                   (json.dumps(vectors), photos[0], identity)).rowcount
            if not count:
                raise ValueError('Student profile no longer exists.')
            self.db.execute('DELETE FROM profile_photos WHERE profile_id=?', (identity,))
            self.db.executemany('INSERT INTO profile_photos VALUES(?,?,?)',
                                [(identity, i, photo) for i, photo in enumerate(photos[1:], 1)])

    def photos(self, identity):
        row = self.db.execute('SELECT photo_jpeg FROM profiles WHERE id=?', (identity,)).fetchone()
        if not row or not row[0]:
            return []
        return [row[0]] + [p[0] for p in self.db.execute('SELECT photo_jpeg FROM profile_photos WHERE profile_id=? ORDER BY position', (identity,))]

    def profiles(self):
        fields = ('id', 'name', 'bio', 'consent_at', 'embeddings', 'first_name', 'last_name', 'student_id', 'photo_jpeg')
        return [dict(zip(fields, row)) for row in self.db.execute('SELECT ' + ','.join(fields) + ' FROM profiles')]

    def delete(self, identity):
        with self.db:
            count = self.db.execute('DELETE FROM profiles WHERE id=?', (identity,)).rowcount
        return count

    def close(self):
        self.db.close()


def identify(vector, profiles, threshold=0.55, margin=0.08):
    """Mean of best three sample similarities; reject close competing identities.

    Scores are cosine similarities, NOT probabilities. Defaults need calibration.
    """
    vector = normalize(vector)
    ranked = []
    for profile in profiles:
        samples = np.array(json.loads(profile['embeddings']), dtype=np.float32)
        scores = np.sort(samples @ vector)
        ranked.append((float(np.mean(scores[-3:])), profile))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return None, None
    score, profile = ranked[0]
    if score < threshold or (len(ranked) > 1 and score - ranked[1][0] < margin):
        return None, score
    return profile, score
