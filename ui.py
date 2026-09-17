"""Shared desktop styling, image helpers, and immutable face captures."""
from dataclasses import dataclass
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

STYLE = '''
QWidget { background: #111827; color: #e5e7eb; font-family: ".AppleSystemUIFont", "Segoe UI", sans-serif; font-size: 14px; }
QLabel { background: transparent; }
QLabel#title { font-size: 27px; font-weight: 700; color: #f9fafb; }
QLabel#section { font-size: 19px; font-weight: 600; }
QLabel#metric { font-size: 26px; font-weight: 700; color: #5eead4; }
QTabBar::tab { padding: 13px 24px; background: #1c2537; }
QTabBar::tab:selected { color: #5eead4; border-bottom: 2px solid #5eead4; }
QTabWidget::pane { border: none; }
QTableWidget { background: #1c2537; gridline-color: #334155; selection-background-color: #28574f; }
QHeaderView::section { background: #263449; color: #cbd5e1; padding: 9px; border: none; }
QComboBox { background: #1c2537; padding: 9px; border: 1px solid #536078; border-radius: 5px; }
QProgressBar { border: 1px solid #334155; border-radius: 5px; text-align: center; }
QProgressBar::chunk { background: #167463; }
QLabel#muted { color: #a3adc2; }
QFrame#card { background: #1c2537; border: 1px solid #334155; border-radius: 12px; }
QLabel#photo { background: #0b1220; border-radius: 8px; color: #94a3b8; }
QLabel#error { color: #fca5a5; }
QLineEdit { background: #0b1220; border: 1px solid #536078; border-radius: 6px; padding: 10px; selection-background-color: #166b5a; }
QLineEdit:focus { border: 2px solid #5eead4; }
QLineEdit[invalid="true"] { border: 2px solid #fca5a5; }
QPushButton { background: #263449; border: 1px solid #475569; border-radius: 7px; padding: 11px 15px; font-weight: 600; }
QPushButton#primary { background: #5eead4; color: #102923; border: none; }
QPushButton#primary:hover { background: #99f6e4; }
QPushButton:disabled, QPushButton#primary:disabled { background: #273246; color: #7b8aa3; border-color: #334155; }
'''


def label(text='', object_name='', wrap=False):
    result = QLabel(text)
    # Names and student IDs are data, never HTML markup.
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setObjectName(object_name)
    result.setWordWrap(wrap)
    return result


def frame_pixmap(frame):
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def fit_photo(widget, jpeg):
    pixmap = QPixmap()
    if jpeg and pixmap.loadFromData(jpeg):
        widget.setPixmap(pixmap.scaled(widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    else:
        widget.setText('No saved photo')


@dataclass
class Capture:
    embedding: np.ndarray
    photo_jpeg: bytes


def capture_face(frame, face, embedding):
    """Use the same unannotated frame for the embedding and portrait."""
    x, y, width, height = [float(value) for value in face[:4]]
    if not np.isfinite([x, y, width, height]).all() or width <= 0 or height <= 0:
        raise ValueError('Could not capture the face. Please try again.')
    left, top = max(0, int(x - width * .18)), max(0, int(y - height * .18))
    right = min(frame.shape[1], int(x + width * 1.18))
    bottom = min(frame.shape[0], int(y + height * 1.18))
    if right <= left or bottom <= top:
        raise ValueError('Move your face fully into the frame.')
    portrait = frame[top:bottom, left:right].copy()
    ok, encoded = cv2.imencode('.jpg', portrait, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError('Could not capture the photo. Please try again.')
    return Capture(embedding.copy(), encoded.tobytes())
