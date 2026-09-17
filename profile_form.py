"""Validated student profile creation and editing."""
import sqlite3
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox
from PySide6.QtCore import Slot, Qt
from core import StudentIDError, validate_student
from ui import STYLE, label, fit_photo

class ProfileDialog(QDialog):
    def __init__(self, capture, store, parent=None, captures=None, profile=None):
        super().__init__(parent)
        self.capture, self.store = capture, store
        self.captures, self.profile = captures, profile
        self.profile_id = None
        self.setWindowTitle('FaceFolio — New student profile')
        self.setMinimumWidth(600)
        self.setStyleSheet(STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)
        layout.addWidget(label('Edit student details' if profile else 'Create a student profile', 'title'))
        layout.addWidget(label('Check the captured photo, then enter their details.', 'muted', True))
        content = QHBoxLayout()
        self.photo = label('', 'photo')
        self.photo.setFixedSize(180, 220)
        self.photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fit_photo(self.photo, profile.get('photo_jpeg') if profile else capture.photo_jpeg)
        content.addWidget(self.photo)
        form = QFormLayout()
        form.setSpacing(14)
        self.first_name = QLineEdit()
        self.first_name.setAccessibleName('First name')
        self.first_name.setMaxLength(80)
        self.last_name = QLineEdit()
        self.last_name.setAccessibleName('Last name')
        self.last_name.setMaxLength(80)
        self.student_id = QLineEdit()
        self.student_id.setAccessibleName('Student ID, exactly seven digits')
        self.student_id.setPlaceholderText('e.g. 0123456')
        # Validate the whole submitted value. Filtering keystrokes could silently
        # turn an eight-digit ID or one containing letters into a different ID.
        form.addRow('First name', self.first_name)
        form.addRow('Last name', self.last_name)
        form.addRow('Student ID', self.student_id)
        content.addLayout(form, 1)
        layout.addLayout(content)
        self.error = label('', 'error', True)
        layout.addWidget(self.error)
        layout.addWidget(label('Save this consenting participant’s photo and profile on this computer.', 'muted', True))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText('Save profile')
        buttons.button(QDialogButtonBox.StandardButton.Save).setObjectName('primary')
        buttons.accepted.connect(self.save_profile)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.first_name.setFocus()
        if captures:
            layout.insertWidget(2, label(f'{len(captures)} enrollment photos ready to save.', 'muted'))
        if profile:
            self.first_name.setText(profile['first_name'])
            self.last_name.setText(profile['last_name'])
            self.student_id.setText(profile['student_id'])

    @Slot()
    def save_profile(self):
        self.error.clear()
        self.mark_student_id_invalid(False)
        try:
            fields = validate_student(self.first_name.text(), self.last_name.text(), self.student_id.text())
            if self.profile:
                self.store.update_student(self.profile['id'], *fields)
                self.profile_id = self.profile['id']
            elif self.captures is not None:
                samples = [(capture.embedding, capture.photo_jpeg) for capture in self.captures]
                self.profile_id = self.store.add_student_samples(*fields, samples, consent=True)
            else:
                self.profile_id = self.store.add_student(*fields, self.capture.embedding, self.capture.photo_jpeg, consent=True)
        except StudentIDError as error:
            self.error.setText(str(error))
            self.mark_student_id_invalid(True)
            self.student_id.setFocus()
            self.student_id.selectAll()
            return
        except (ValueError, sqlite3.Error, OSError) as error:
            self.error.setText(str(error))
            return
        self.accept()

    def mark_student_id_invalid(self, invalid):
        self.student_id.setProperty('invalid', invalid)
        self.student_id.style().unpolish(self.student_id)
        self.student_id.style().polish(self.student_id)
        self.student_id.update()
