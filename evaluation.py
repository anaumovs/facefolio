"""Shared held-out evaluation for the dashboard and CLI. No invented results."""
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from core import identify


@dataclass(frozen=True)
class TestPhoto:
    path: Path
    expected_id: str  # Empty means a consenting person outside the gallery.


def load_manifest(path, profiles):
    path = Path(path)
    ids = {p['id'] for p in profiles}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'path', 'expected_id'} <= set(reader.fieldnames or []):
            raise ValueError('CSV must have path and expected_id columns.')
        tests = []
        for row in reader:
            expected = (row.get('expected_id') or '').strip()
            if expected and expected not in ids:
                raise ValueError('A CSV expected_id does not match a profile UUID in this gallery.')
            value = (row.get('path') or '').strip()
            if not value:
                raise ValueError('Every test row needs an image path.')
            tests.append(TestPhoto((path.parent / value).resolve(), expected))
    if not tests:
        raise ValueError('The CSV contains no test photos.')
    if len({t.path for t in tests}) != len(tests):
        raise ValueError('The CSV repeats an image path. Use each test photo once.')
    return tests


def enrollment_hashes(store, profiles):
    return {hashlib.sha256(photo).hexdigest() for p in profiles for photo in store.photos(p['id'])}


def assess_photo(test, engine, profiles, threshold, margin, forbidden_hashes=frozenset()):
    if test.expected_id and not any(p['id'] == test.expected_id for p in profiles):
        raise ValueError('A test refers to a student who is no longer in the gallery.')
    path = Path(test.path)
    if path.stat().st_size > 25_000_000:
        raise ValueError(f'{path.name}: test photos must be under 25 MB.')
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest in forbidden_hashes:
        raise ValueError(f'{path.name}: this is an enrollment photo. Choose a separate test capture.')
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f'{path.name}: cannot read this image. Choose a supported photo.')
    scale = min(1., 960 / frame.shape[1])
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    start = time.perf_counter()
    prediction, score, rejected = None, None, ''
    faces = engine.faces(frame)
    try:
        if len(faces) != 1:
            raise ValueError('Exactly one face required')
        feature = engine.feature(frame, faces[0])
        prediction, score = identify(feature, profiles, threshold, margin)
    except ValueError as error:
        rejected = str(error)
    latency = (time.perf_counter() - start) * 1000
    predicted = prediction['id'] if prediction else ''
    if rejected:
        outcome = 'rejected_capture'
    elif test.expected_id:
        outcome = 'correct_identity' if predicted == test.expected_id else 'wrong_identity' if predicted else 'missed_known'
    else:
        outcome = 'false_accept' if predicted else 'correct_rejection'
    return dict(known=bool(test.expected_id), outcome=outcome, latency_ms=latency, score=score,
                rejection_reason=rejected, image_hash=digest)


def build_report(results, profiles, threshold, margin):
    if not results:
        raise ValueError('Add test photos before running an evaluation.')
    outcomes = ('correct_identity', 'correct_rejection', 'wrong_identity', 'missed_known', 'false_accept', 'rejected_capture')
    counts = {key: sum(r['outcome'] == key for r in results) for key in outcomes}
    known = sum(r['known'] for r in results)
    unknown = len(results) - known
    rejected_known = sum(r['known'] and r['outcome'] == 'rejected_capture' for r in results)
    valid_unknown = sum(not r['known'] and r['outcome'] != 'rejected_capture' for r in results)
    latencies = [r['latency_ms'] for r in results]
    report = dict(
        created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        total=len(results), known=known, unknown=unknown, valid_unknown=valid_unknown,
        counts=counts, gallery_size=len(profiles), threshold=threshold, margin=margin,
        correct_decision_rate=(counts['correct_identity'] + counts['correct_rejection']) / len(results),
        known_identification_rate=counts['correct_identity'] / known if known else None,
        unknown_false_match_rate=counts['false_accept'] / valid_unknown if valid_unknown else None,
        known_miss_rate=(counts['missed_known'] + rejected_known) / known if known else None,
        median_latency_ms=float(np.median(latencies)), p95_latency_ms=float(np.percentile(latencies, 95)),
        gallery_fingerprint=hashlib.sha256(json.dumps(sorted((p['id'], p['embeddings']) for p in profiles)).encode()).hexdigest(),
        models={'detector': 'YuNet 2023mar', 'recognizer': 'SFace 2021dec', 'opencv': cv2.__version__},
        measurement='Operator-labeled, held-out gallery identification; latency excludes image loading. Capture failures are incorrect decisions and remain in known-photo denominators; false-match rate uses valid unknown captures only.',
    )
    return report
