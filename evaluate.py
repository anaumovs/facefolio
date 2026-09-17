"""Evaluate held-out local photos against the enrolled student gallery."""
import argparse
import json
from pathlib import Path
from app import Engine, ROOT
from core import Store
from evaluation import assess_photo, build_report, enrollment_hashes, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, help='CSV with path,expected_id; blank ID means unknown')
    parser.add_argument('--threshold', type=float, default=.55)
    parser.add_argument('--margin', type=float, default=.08)
    parser.add_argument('--gallery', choices=('students', 'all'), default='students')
    args = parser.parse_args()
    if not -1 <= args.threshold <= 1 or not 0 <= args.margin <= 2:
        parser.error('Threshold must be -1..1; margin must be 0..2')
    engine = Engine()
    store = Store(ROOT / 'private' / 'profiles.sqlite3')
    try:
        profiles = store.profiles()
        if args.gallery == 'students':
            profiles = [p for p in profiles if p.get('student_id')]
        tests = load_manifest(args.manifest, profiles)
        forbidden = enrollment_hashes(store, profiles)
        results = [assess_photo(t, engine, profiles, args.threshold, args.margin, forbidden) for t in tests]
        if len({r['image_hash'] for r in results}) != len(results):
            raise ValueError('Repeated image content found. Use each test photo once.')
        report = build_report(results, profiles, args.threshold, args.margin)
        report['gallery'] = args.gallery
        print(json.dumps(report, indent=2))
    finally:
        store.close()


if __name__ == '__main__':
    main()
