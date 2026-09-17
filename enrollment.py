"""Guided multi-photo enrollment using the existing camera connection."""
import time
import cv2
from PySide6.QtCore import Slot, Qt, QTimer, QEvent
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QDialogButtonBox
from core import normalize, identify
from ui import STYLE, label, fit_photo, frame_pixmap, capture_face


class EnrollmentDialog(QDialog):
    PROMPTS = ('Look straight at the camera.', 'Turn your face slightly to the left.',
               'Turn your face slightly to the right.', 'Try a small change in expression.',
               'Face forward at a slightly different distance.')

    def __init__(self, engine, camera, threshold=.55, initial=None, parent=None, reference=None):
        super().__init__(parent)
        self.engine, self.camera, self.threshold = engine, camera, threshold
        self.reference = reference
        self.samples = [initial] if initial else []
        self.pending = None
        self.feedback = ''
        self.feedback_until = 0
        self.blocked_reason = 'Wait for one clear face before capturing.'
        self.last_capture = time.monotonic() if initial else -float('inf')
        self.setWindowTitle('FaceFolio — Capture five enrollment photos')
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(660)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)
        layout.addWidget(label('Build a stronger face profile', 'title'))
        self.guide = label('', 'section', True)
        layout.addWidget(self.guide)
        self.video = label('Starting camera…', 'photo')
        self.video.setFixedSize(616, 330)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video)
        self.message = label('Keep the same consenting person in view throughout.', 'muted', True)
        layout.addWidget(self.message)
        thumbs = QHBoxLayout()
        self.thumbnails = []
        for i in range(5):
            item = label(str(i + 1), 'photo')
            item.setFixedSize(100, 80)
            item.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.thumbnails.append(item)
            thumbs.addWidget(item)
        layout.addLayout(thumbs)
        actions = QHBoxLayout()
        self.capture_button = QPushButton('Capture photo (C)')
        self.capture_button.setObjectName('primary')
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_next)
        self.retake_button = QPushButton('Remove last photo (R)')
        self.retake_button.clicked.connect(self.retake)
        actions.addWidget(self.capture_button)
        actions.addWidget(self.retake_button)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.continue_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.continue_button.setText('Save 5 photos' if reference else 'Continue')
        self.continue_button.setEnabled(False)
        buttons.accepted.connect(self.finish)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.keyboard_filter_installed = False
        self.timer = QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self.tick)
        self.finished.connect(self.timer.stop)
        self.update_progress()
        self.timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if not self.keyboard_filter_installed:
            QApplication.instance().installEventFilter(self)
            self.keyboard_filter_installed = True
        self.activateWindow()
        self.setFocus()

    def hideEvent(self, event):
        if self.keyboard_filter_installed:
            QApplication.instance().removeEventFilter(self)
            self.keyboard_filter_installed = False
        super().hideEvent(event)

    def eventFilter(self, watched, event):
        # Handle keys in this capture popup directly, including when a child
        # button has focus. Claim ShortcutOverride before parent shortcuts can
        # consume the key; never intercept keys in another window or text form.
        if (self.isVisible() and isinstance(watched, QWidget)
                and watched.window() is self
                and event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
                and event.key() in (Qt.Key.Key_C, Qt.Key.Key_R)
                and not event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)):
            event.accept()
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                if event.key() == Qt.Key.Key_C:
                    self.capture_next()
                else:
                    self.retake()
            return True
        return super().eventFilter(watched, event)

    def update_progress(self):
        count = len(self.samples)
        self.guide.setText(f'{count}/5 photos · ' + (self.PROMPTS[count] if count < 5 else 'C retakes photo 5, or finish below.'))
        self.capture_button.setText('Retake last photo (C)' if count == 5 else 'Capture photo (C)')
        self.continue_button.setEnabled(count == 5)
        self.retake_button.setEnabled(count > 0)
        for i, thumbnail in enumerate(self.thumbnails):
            thumbnail.clear()
            if i < count:
                fit_photo(thumbnail, self.samples[i].photo_jpeg)
            else:
                thumbnail.setText(str(i + 1))

    @Slot()
    def tick(self):
        self.pending = None
        ok, frame = self.camera.read()
        if not ok:
            self.capture_button.setEnabled(False)
            self.timer.stop()
            self.message.setText('Camera disconnected. Cancel, reconnect the camera, and try again.')
            self.blocked_reason = self.message.text()
            return
        try:
            scale = min(1., 960 / frame.shape[1])
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            faces = self.engine.faces(frame)
            if len(faces) != 1:
                self.message.setText('Keep exactly one person in the frame.')
            else:
                feature = self.engine.feature(frame, faces[0])
                if self.reference and identify(feature, [self.reference], self.threshold)[0] is None:
                    self.message.setText('Bring the selected student into view to replace their photos.')
                elif self.samples and float(normalize(feature) @ normalize(self.samples[0].embedding)) < self.threshold:
                    self.message.setText('Face changed or angle is too strong. Bring the original person back into view.')
                elif time.monotonic() - self.last_capture < 1:
                    self.message.setText('Pause for one second and change your angle slightly.')
                else:
                    self.pending = (frame.copy(), faces[0].copy(), feature.copy())
                    self.message.setText('Five photos ready. Press C to retake the last, or finish below.' if len(self.samples) == 5 else 'Ready. Click Capture photo or press C.')
        except (ValueError, cv2.error) as error:
            self.message.setText(str(error))
        # Do not disable/re-enable between valid frames: that cancels a mouse
        # press if a timer refresh falls between the press and release events.
        self.capture_button.setEnabled(self.pending is not None)
        self.blocked_reason = self.message.text() if self.pending is None else ''
        if time.monotonic() < self.feedback_until and (self.pending is not None or time.monotonic() - self.last_capture < 1):
            self.message.setText(self.feedback)
        self.video.setPixmap(frame_pixmap(frame).scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    @Slot()
    def capture_next(self):
        if self.pending is None:
            self.notify(self.blocked_reason or 'Wait for one clear face before capturing.')
            return
        if time.monotonic() - self.last_capture < 1:
            self.notify('Wait one second before capturing another photo.')
            return
        frame, face, feature = self.pending
        try:
            sample = capture_face(frame, face, feature)
            replacing = len(self.samples) == 5
            if any(s.photo_jpeg == sample.photo_jpeg for s in self.samples):
                self.notify('This repeats a captured frame. Change your angle slightly and capture again.')
                return
            if replacing:
                self.samples[-1] = sample
            else:
                self.samples.append(sample)
            self.last_capture = time.monotonic()
            self.pending = None
            self.capture_button.setEnabled(False)
            self.update_progress()
            self.notify('Photo 5 retaken.' if replacing else f'Photo {len(self.samples)} of 5 captured.')
        except (ValueError, cv2.error) as error:
            self.notify(str(error))

    def notify(self, message):
        self.feedback = message
        self.feedback_until = time.monotonic() + 1.5
        self.message.setText(message)

    @Slot()
    def retake(self):
        if not self.samples:
            self.notify('No photos yet. Press C to capture the first one.')
            return
        self.samples.pop()
        self.pending = None
        self.capture_button.setEnabled(False)
        self.update_progress()
        self.notify('Last photo removed. Press C to capture its replacement.')

    @Slot()
    def finish(self):
        if len(self.samples) == 5:
            self.accept()
