"""Search, review, edit, and delete student profiles."""
import json
import sqlite3
from PySide6.QtCore import Slot, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QPushButton, QDialog, QMessageBox,
)
from ui import label, fit_photo
from profile_form import ProfileDialog


class DirectoryPage(QWidget):
    changed = Signal()
    recapture = Signal(str)

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.students = []
        self.identity = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.addWidget(label('Student directory', 'title'))
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search first name, last name, or student ID')
        self.search.setAccessibleName('Search students')
        self.search.textChanged.connect(self.filter_students)
        layout.addWidget(self.search)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['First name', 'Last name', 'Student ID', 'Photos'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.itemSelectionChanged.connect(self.select_student)
        layout.addWidget(self.table, 1)
        self.details = label('Select a student to view enrollment photos.', 'section', True)
        layout.addWidget(self.details)
        photos = QHBoxLayout()
        self.thumbnails = []
        for _ in range(5):
            item = label('', 'photo')
            item.setFixedSize(105, 100)
            item.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.thumbnails.append(item)
            photos.addWidget(item)
        photos.addStretch()
        layout.addLayout(photos)
        buttons = QHBoxLayout()
        self.edit_button = QPushButton('Edit details')
        self.edit_button.clicked.connect(self.edit_selected)
        self.photos_button = QPushButton('Replace with 5 photos (C)')
        self.photos_button.clicked.connect(self.request_recapture)
        # Table-scoped: typing a C into the name/ID search must remain typing.
        self.capture_shortcut = QShortcut(QKeySequence('C'), self.table)
        self.capture_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.capture_shortcut.setAutoRepeat(False)
        self.capture_shortcut.activated.connect(self.request_recapture)
        self.delete_button = QPushButton('Delete profile')
        self.delete_button.clicked.connect(self.delete_selected)
        for button in (self.edit_button, self.photos_button, self.delete_button):
            button.setEnabled(False)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.message = label('', 'muted', True)
        layout.addWidget(self.message)
        self.refresh()

    def refresh(self):
        self.students = sorted((p for p in self.store.profiles() if p.get('student_id')),
                               key=lambda p: (p['last_name'].casefold(), p['first_name'].casefold()))
        self.filter_students()

    @Slot()
    def filter_students(self):
        query = self.search.text().strip().casefold()
        rows = [p for p in self.students if query in f"{p['first_name']} {p['last_name']} {p['student_id']}".casefold()]
        previous = self.identity
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        selected_row = None
        for row, profile in enumerate(rows):
            values = [profile['first_name'], profile['last_name'], profile['student_id'], str(len(self.store.photos(profile['id'])))]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, profile['id'])
                self.table.setItem(row, column, item)
            if profile['id'] == previous:
                selected_row = row
        self.table.clearSelection()
        if selected_row is not None:
            self.table.selectRow(selected_row)
        self.table.blockSignals(False)
        self.select_student()
        self.message.setText(f'{len(rows)} of {len(self.students)} student profiles shown.')

    def selected_profile(self):
        return next((p for p in self.students if p['id'] == self.identity), None)

    @Slot()
    def select_student(self):
        items = self.table.selectedItems()
        self.identity = items[0].data(Qt.ItemDataRole.UserRole) if items else None
        profile = self.selected_profile()
        for button in (self.edit_button, self.photos_button, self.delete_button):
            button.setEnabled(profile is not None)
        photos = self.store.photos(self.identity) if profile else []
        for i, item in enumerate(self.thumbnails):
            item.clear()
            if i < len(photos):
                fit_photo(item, photos[i])
        if profile:
            count = len(json.loads(profile['embeddings']))
            self.details.setText(f"{profile['name']} · {profile['student_id']} · {count} enrollment sample(s)")
        else:
            self.details.setText('Select a student to view enrollment photos.')

    @Slot()
    def edit_selected(self):
        profile = self.selected_profile()
        if not profile:
            return
        dialog = ProfileDialog(None, self.store, self, profile=profile)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.refresh()
                self.changed.emit()
        finally:
            dialog.deleteLater()

    @Slot()
    def request_recapture(self):
        if self.identity:
            self.recapture.emit(self.identity)

    @Slot()
    def delete_selected(self):
        profile = self.selected_profile()
        if not profile:
            return
        prompt = QMessageBox(self)
        prompt.setWindowTitle('Delete student profile')
        prompt.setTextFormat(Qt.TextFormat.PlainText)
        prompt.setText(f"Delete {profile['name']} ({profile['student_id']}) and all their enrollment photos?")
        prompt.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        prompt.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if prompt.exec() != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete(profile['id'])
            self.refresh()
            self.changed.emit()
        except sqlite3.Error as error:
            self.message.setText(f'Could not delete the profile: {error}')
