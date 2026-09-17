"""FaceFolio desktop application: camera and student directory."""
import sqlite3
import sys
import cv2
from PySide6.QtCore import Slot, Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QApplication, QDialog, QFrame, QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTabWidget
from core import identify
from ui import STYLE, label, frame_pixmap, fit_photo, Capture, capture_face
from profile_form import ProfileDialog
from enrollment import EnrollmentDialog
from directory import DirectoryPage

class WebcamWindow(QMainWindow):
    def __init__(self, args, store, engine=None, camera=None, autostart=True):
        super().__init__()
        if engine is None:
            from app import Engine
            engine = Engine()
        self.engine, self.store, self.args = engine, store, args
        self.profiles = store.profiles()
        self.selected_id = args.id if args.command == 'verify' else None
        if self.selected_id and not any(p['id'] == self.selected_id for p in self.profiles):
            raise ValueError('Profile ID not found')
        self.camera = camera if camera is not None else cv2.VideoCapture(args.camera)
        self.pending = None
        self.capture_target_id = None
        self.displayed_id = None
        self.closed = False
        self.camera_announced = False
        self.timer = QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self.tick)
        self.setWindowTitle('FaceFolio — Student profiles')
        self.setStyleSheet(STYLE)
        self.resize(1180, 760)
        self.setMinimumSize(1050, 680)
        root = QWidget()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(root, 'Live camera')
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        header = QHBoxLayout()
        header.addWidget(label('FaceFolio', 'title'))
        header.addStretch()
        self.gallery_label = label('', 'muted')
        header.addWidget(self.gallery_label)
        layout.addLayout(header)
        layout.addWidget(label('Recognize a student. Capture a new face. Build their profile.', 'muted'))
        content = QHBoxLayout()
        content.setSpacing(22)
        self.video = label('Starting camera…', 'photo')
        self.video.setMinimumSize(560, 380)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(self.video, 3)
        card = QFrame()
        card.setObjectName('card')
        card.setMinimumWidth(390)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 22, 20, 22)
        card_layout.setSpacing(16)
        self.card_title = label('No person selected', 'section', True)
        card_layout.addWidget(self.card_title)
        row = QHBoxLayout()
        row.setSpacing(16)
        self.photo = label('Captured photo\nappears here', 'photo')
        self.photo.setFixedSize(144, 180)
        self.photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.photo)
        details = QVBoxLayout()
        details.addWidget(label('FIRST NAME', 'muted'))
        self.first_name = label('—', wrap=True)
        details.addWidget(self.first_name)
        details.addWidget(label('LAST NAME', 'muted'))
        self.last_name = label('—', wrap=True)
        details.addWidget(self.last_name)
        details.addWidget(label('STUDENT ID', 'muted'))
        self.student_id = label('—')
        details.addWidget(self.student_id)
        details.addStretch()
        row.addLayout(details, 1)
        card_layout.addLayout(row)
        self.score = label('', 'muted', True)
        card_layout.addWidget(self.score)
        self.hint = label('Show one face in good light to begin.', 'muted', True)
        card_layout.addWidget(self.hint)
        card_layout.addStretch()
        self.capture_button = QPushButton('Capture and add profile (C)')
        self.capture_button.setObjectName('primary')
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_profile)
        card_layout.addWidget(self.capture_button)
        content.addWidget(card, 2)
        layout.addLayout(content, 1)
        self.status = label('Starting camera…', 'muted', True)
        layout.addWidget(self.status)
        self.statusBar().showMessage('Photos are saved only when you choose Save profile. Close the window or press Q to quit.')
        self.quit_shortcut = QShortcut(QKeySequence('Q'), self)
        self.quit_shortcut.activated.connect(self.close)
        self.capture_shortcut = QShortcut(QKeySequence('C'), self)
        self.capture_shortcut.setAutoRepeat(False)
        self.capture_shortcut.activated.connect(self.capture_profile)
        self.directory = DirectoryPage(store)
        self.tabs.addTab(self.directory, 'Student directory')
        self.directory.changed.connect(self.refresh_profiles)
        self.directory.recapture.connect(self.replace_enrollment)
        self.tabs.currentChanged.connect(self.change_tab)
        self.refresh_gallery_label()
        if not self.camera.isOpened():
            self.status.setText('Cannot open the webcam. Close other camera apps and check Terminal camera access, then restart FaceFolio.')
            self.video.setText('Camera unavailable')
            print(self.status.text(), flush=True)
        elif autostart:
            self.timer.start()

    def refresh_gallery_label(self):
        mode = 'Selected profile verification' if self.selected_id else 'Recognizing saved profiles'
        count = sum(bool(p.get('student_id')) for p in self.profiles)
        noun = 'student' if count == 1 else 'students'
        self.gallery_label.setText(f'{mode}  ·  {count} {noun}')

    @Slot()
    def refresh_profiles(self):
        self.profiles = self.store.profiles()
        if self.selected_id and not any(p['id'] == self.selected_id for p in self.profiles):
            self.selected_id = None
        self.displayed_id = None
        self.show_profile(None)
        self.pending = None
        self.capture_target_id = None
        self.capture_button.setEnabled(False)
        self.refresh_gallery_label()
        self.directory.refresh()

    @Slot(int)
    def change_tab(self, index):
        if index < 0 or getattr(self, 'closed', True):
            return
        self.timer.stop()
        self.pending = None
        self.capture_target_id = None
        self.capture_button.setEnabled(False)
        self.capture_shortcut.setEnabled(index == 0)
        self.quit_shortcut.setEnabled(index == 0)
        if index == 0 and not self.closed and self.camera.isOpened():
            self.timer.start()
        elif index == 1:
            self.directory.refresh()

    @Slot(str)
    def replace_enrollment(self, identity):
        self.timer.stop()
        self.capture_shortcut.setEnabled(False)
        dialog = None
        try:
            if not self.camera.isOpened():
                raise ValueError('Camera unavailable. Reconnect it and restart FaceFolio.')
            reference = next((p for p in self.profiles if p['id'] == identity), None)
            if reference is None:
                raise ValueError('Student profile no longer exists.')
            dialog = EnrollmentDialog(self.engine, self.camera, self.args.threshold, parent=self, reference=reference)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.store.replace_student_samples(identity, [(s.embedding, s.photo_jpeg) for s in dialog.samples])
                self.refresh_profiles()
                self.directory.message.setText('Five enrollment photos saved. The student’s details are unchanged.')
        except (ValueError, sqlite3.Error, cv2.error) as error:
            self.directory.message.setText(str(error))
        finally:
            if dialog is not None:
                dialog.deleteLater()
            if not self.closed and self.tabs.currentIndex() == 0 and self.camera.isOpened():
                self.capture_shortcut.setEnabled(True)
                self.timer.start()

    def show_profile(self, profile, score=None):
        identity = profile['id'] if profile else None
        if identity != self.displayed_id or identity is None:
            self.displayed_id = identity
            self.photo.clear()
            if profile:
                self.card_title.setText(profile['name'])
                self.first_name.setText(profile.get('first_name') or profile['name'])
                self.last_name.setText(profile.get('last_name') or 'Not recorded')
                self.student_id.setText(profile.get('student_id') or 'Not recorded')
                fit_photo(self.photo, profile.get('photo_jpeg'))
            else:
                self.card_title.setText('No confident match')
                self.first_name.setText('—')
                self.last_name.setText('—')
                self.student_id.setText('—')
                self.photo.setText('Captured photo\nappears here')
        self.score.setText(f'Similarity: {score:.3f}' if profile and score is not None else '')

    @Slot()
    def tick(self):
        # Never leave another person's details or a stale capture active.
        self.pending = None
        self.capture_target_id = None
        if self.closed:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.capture_button.setEnabled(False)
            self.timer.stop()
            self.camera.release()
            self.video.clear()
            self.video.setText('Camera disconnected')
            self.show_profile(None)
            self.status.setText('Camera stopped delivering frames. Reconnect it and restart FaceFolio.')
            print(self.status.text(), flush=True)
            return
        if not self.camera_announced:
            self.camera_announced = True
            print('Webcam connected. Unknown faces can now be captured to create student profiles.', flush=True)
        try:
            scale = min(1., 960 / frame.shape[1])
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            faces = self.engine.faces(frame)
            profile, score, color = None, None, (0, 180, 255)
            message = 'Show exactly one face in good light.'
            if len(faces) == 1:
                face = faces[0]
                feature = self.engine.feature(frame, face)
                # Earlier CLI demo enrollments have no student ID. Preserve them,
                # but keep them outside the student directory's matching gallery.
                students = [p for p in self.profiles if p.get('student_id')]
                gallery_match, gallery_score = identify(feature, students, self.args.threshold, self.args.margin)
                candidates = [p for p in self.profiles if p['id'] == self.selected_id] if self.selected_id else students
                profile, score = identify(feature, candidates, self.args.threshold, self.args.margin) if self.selected_id else (gallery_match, gallery_score)
                if profile:
                    color = (80, 210, 130)
                    message = 'Profile found.'
                    if profile.get('student_id'):
                        self.capture_target_id = profile['id']
                        message += ' Press C to update their five enrollment photos.'
                elif gallery_match:
                    message = 'This person has a saved profile but does not match the selected profile.'
                else:
                    # Encode only when the user chooses capture; keep a clean copy in memory.
                    self.pending = (frame.copy(), face.copy(), feature.copy())
                    message = 'Unknown person — choose Capture and add profile, or press C.'
                x, y, width, height = map(int, face[:4])
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            elif len(faces) > 1:
                message = 'More than one face detected. Capture one person at a time.'
            self.show_profile(profile, score)
            self.status.setText(message)
            self.hint.setText('Saved photo and details for the matched profile.' if profile else message)
        except (ValueError, cv2.error) as error:
            self.pending = None
            self.capture_target_id = None
            self.capture_button.setEnabled(False)
            self.show_profile(None)
            self.status.setText(str(error))
            self.hint.setText('Move closer, hold still, and use good lighting.')
        self.capture_button.setText('Update 5 profile photos (C)' if self.capture_target_id else 'Capture and add profile (C)')
        self.capture_button.setEnabled(self.pending is not None or self.capture_target_id is not None)
        self.video.setPixmap(frame_pixmap(frame).scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    @Slot()
    def capture_profile(self):
        if not self.capture_button.isEnabled():
            return
        if self.capture_target_id:
            self.replace_enrollment(self.capture_target_id)
            return
        if self.pending is None:
            return
        self.timer.stop()
        self.capture_shortcut.setEnabled(False)
        frame, face, feature = self.pending
        self.pending = None
        self.capture_button.setEnabled(False)
        enrollment, dialog = None, None
        try:
            capture = capture_face(frame, face, feature)
            enrollment = EnrollmentDialog(self.engine, self.camera, self.args.threshold, initial=capture, parent=self)
            if enrollment.exec() != QDialog.DialogCode.Accepted:
                return
            dialog = ProfileDialog(enrollment.samples[0], self.store, self, captures=enrollment.samples)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.selected_id = None
                self.refresh_profiles()
                profile = next(p for p in self.profiles if p['id'] == dialog.profile_id)
                self.show_profile(profile)
                self.statusBar().showMessage('Profile saved. The camera is now recognizing all saved student profiles.', 8000)
        except (ValueError, cv2.error, sqlite3.Error) as error:
            self.status.setText(str(error))
        finally:
            for temporary_dialog in (enrollment, dialog):
                if temporary_dialog is not None:
                    temporary_dialog.deleteLater()
            if not self.closed and self.tabs.currentIndex() == 0 and self.camera.isOpened():
                self.capture_shortcut.setEnabled(True)
                self.timer.start()

    def closeEvent(self, event):
        self.closed = True
        # Qt can emit currentChanged while destroying its child tabs. Prevent
        # callbacks into a Python window whose attributes are already cleared.
        self.tabs.blockSignals(True)
        self.pending = None
        self.timer.stop()
        self.camera.release()
        event.accept()


def run(args, store):
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName('FaceFolio')
    window = WebcamWindow(args, store)
    window.show()
    window.raise_()
    window.activateWindow()
    return application.exec()
