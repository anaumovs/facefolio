"""FaceFolio: consent-based local webcam enrollment and profile lookup."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path
import time
import urllib.request
import cv2
from core import Store, identify, normalize

ROOT = Path(__file__).resolve().parent
MODELS = {
    'face_detection_yunet_2023mar.onnx': 'face_detection_yunet',
    'face_recognition_sface_2021dec.onnx': 'face_recognition_sface',
}


def download():
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    for filename, directory in MODELS.items():
        target = folder / filename
        if target.exists():
            continue
        url = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{directory}/{filename}'
        print(f'Downloading {filename}')
        temporary = target.with_suffix('.part')
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


class Engine:
    def __init__(self):
        paths = [ROOT / 'models' / name for name in MODELS]
        if not all(p.exists() for p in paths):
            raise ValueError('Download models first: python app.py download-models')
        self.detector = cv2.FaceDetectorYN.create(str(paths[0]), '', (320, 320), 0.9)
        self.recognizer = cv2.FaceRecognizerSF.create(str(paths[1]), '')

    def faces(self, frame):
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else faces

    def feature(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        if min(face[2:4]) < 90 or cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 45:
            raise ValueError('Move closer and hold still in good light')
        return normalize(self.recognizer.feature(aligned))


def webcam(args, store):
    engine = Engine()
    mode = args.command
    enrolled = False
    profiles = store.profiles()
    if args.command == 'recognize' and not profiles:
        raise ValueError('Enroll a profile first')
    if args.command == 'verify':
        profiles = [p for p in profiles if p['id'] == args.id]
        if not profiles:
            raise ValueError('Profile ID not found')
    camera = cv2.VideoCapture(args.camera)
    samples, last_capture = [], 0
    try:
        if not camera.isOpened():
            raise ValueError('Cannot open webcam. Check camera permissions or --camera index.')
        while True:
            ok, frame = camera.read()
            if not ok:
                raise ValueError('Camera stopped delivering frames')
            frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            faces = engine.faces(frame)
            feature = None
            label = 'Show exactly one face'
            profile = None
            if len(faces) == 1:
                face = faces[0]
                try:
                    feature = engine.feature(frame, face)
                    if mode == 'enroll':
                        label = f'SPACE: capture ({len(samples)}/5). Change angle slightly.'
                    else:
                        profile, score = identify(feature, profiles, args.threshold, args.margin)
                        label = profile['name'] if profile else 'Unknown / no confident match'
                        if score is not None:
                            label += f' | similarity {score:.3f}'
                except ValueError as error:
                    label = str(error)
                x, y, w, h = map(int, face[:4])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (70, 210, 120) if profile else (0, 180, 255), 2)
            cv2.rectangle(frame, (0, 0), (960, 105), (27, 24, 22), -1)
            heading = 'FACEFOLIO | Profile saved - verifying your profile | Q: quit' if enrolled else f'FACEFOLIO | {mode} | Q: quit'
            cv2.putText(frame, heading, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (235, 235, 235), 1)
            cv2.putText(frame, label[:100], (18, 57), cv2.FONT_HERSHEY_SIMPLEX, .55, (100, 220, 190), 1)
            if profile:
                cv2.putText(frame, profile['bio'][:100], (18, 85), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1)
            cv2.imshow('FaceFolio', frame)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty('FaceFolio', cv2.WND_PROP_VISIBLE) < 1:
                break
            if mode == 'enroll' and key == 32 and feature is not None and time.monotonic() - last_capture > 1:
                samples.append(feature.copy())
                last_capture = time.monotonic()
                print(f'Captured {len(samples)}/5', flush=True)
                if len(samples) == 5:
                    identity = store.add(args.name, args.bio, samples, args.consent)
                    profiles = [p for p in store.profiles() if p['id'] == identity]
                    mode = 'verify'
                    enrolled = True
                    print(f'Enrolled {args.name}. Profile ID: {identity}. Camera stays open to verify this profile; Q quits.', flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('download-models')
    sub.add_parser('profiles')
    deletion = sub.add_parser('delete')
    deletion.add_argument('id')
    for command in ('enroll', 'recognize', 'verify'):
        p = sub.add_parser(command)
        p.add_argument('--camera', type=int, default=0)
        p.add_argument('--threshold', type=float, default=.55)
        p.add_argument('--margin', type=float, default=.08)
        if command == 'enroll':
            p.add_argument('--name', required=True)
            p.add_argument('--bio', default='')
            p.add_argument('--consent', action='store_true', required=True, help='Confirm the participant agreed to local face enrollment')
        if command == 'verify':
            p.add_argument('--id', required=True)
    args = parser.parse_args(sys.argv[1:] or ['recognize'])
    store = None
    try:
        if args.command == 'download-models':
            download()
            return
        store = Store(ROOT / 'private' / 'profiles.sqlite3')
        if args.command == 'profiles':
            for p in store.profiles():
                print(json.dumps({k: v for k, v in p.items() if k not in ('embeddings', 'photo_jpeg')}))
        elif args.command == 'delete':
            print(f'Deleted {store.delete(args.id)} profile(s)')
        else:
            if not -1 <= args.threshold <= 1 or not 0 <= args.margin <= 2:
                raise ValueError('Threshold must be -1..1; margin must be 0..2')
            if args.command == 'enroll':
                webcam(args, store)
            else:
                from desktop import run
                run(args, store)
    except (ValueError, cv2.error, OSError, sqlite3.Error) as error:
        parser.exit(1, f'FaceFolio: {error}\n')
    finally:
        if store:
            store.close()


if __name__ == '__main__':
    main()
