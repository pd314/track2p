from qtpy.QtWidgets import (
    QWidget, QPushButton, QFileDialog, QLineEdit, QLabel,
    QFormLayout, QComboBox, QFrame, QVBoxLayout
)
from qtpy.QtCore import Qt


class ImportWindow(QWidget):

    def __init__(self, main_wd):
        super().__init__()
        self.main_window = main_wd
        self.path_to_t2p = None

        root = QVBoxLayout()
        root.setSpacing(12)

        # ── Section: folder picker ──────────────────────────────────────────
        folder_section = QFormLayout()
        folder_section.setRowWrapPolicy(QFormLayout.WrapAllRows)

        folder_header = QLabel("📂  Select track2p output folder")
        folder_header.setStyleSheet("font-weight: bold; font-size: 11pt;")

        hint = QLabel(
            "Select the <b>suite2p</b> folder directly — the one that <b>contains</b> "
            "<code>plane0/</code>, <code>plane1/</code>, etc.<br>"
            "<small>Example:&nbsp; <code>D:/experiment/day2/suite2p/</code></small>"
        )
        
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.RichText)

        self.import_button = QPushButton("Browse…")
        self.import_button.setFixedWidth(90)
        self.import_button.clicked.connect(self._browse)

        self.path_display = QLabel("<i>No folder selected</i>")
        self.path_display.setTextFormat(Qt.RichText)
        self.path_display.setWordWrap(True)
        self.path_display.setStyleSheet("color: grey;")

        folder_section.addRow(folder_header)
        folder_section.addRow(hint)
        folder_section.addRow("Folder:", self.import_button)
        folder_section.addRow("Selected:", self.path_display)

        root.addLayout(folder_section)
        root.addWidget(_hline())

        # ── Section: analysis options ───────────────────────────────────────
        options_header = QLabel("⚙️  Analysis options")
        options_header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        root.addWidget(options_header)

        options = QFormLayout()
        options.setRowWrapPolicy(QFormLayout.WrapAllRows)

        self.plane_box = QLineEdit("0")
        self.plane_box.setFixedWidth(50)
        self.plane_box.setToolTip("Zero-indexed plane number (e.g. 0 for plane0)")

        self.trace_choice = QComboBox()
        self.trace_choice.addItems(["F", "dF/F0", "spks"])
        self.trace_choice.setToolTip("Fluorescence trace to display in plots")

        self.channel_choice = QComboBox()
        self.channel_choice.addItems(["0", "1", "Vcorr", "max_proj"])
        self.channel_choice.setToolTip("Image channel shown in mean-image panels")

        options.addRow("Plane index:", self.plane_box)
        options.addRow("Trace type:", self.trace_choice)
        options.addRow("Mean image channel:", self.channel_choice)

        root.addLayout(options)
        root.addWidget(_hline())

        # ── Run ─────────────────────────────────────────────────────────────
        self.run_button = QPushButton("▶  Load && Run")
        self.run_button.setEnabled(False)          # disabled until a folder is chosen
        self.run_button.setFixedHeight(32)
        self.run_button.clicked.connect(self._run)
        root.addWidget(self.run_button, alignment=Qt.AlignRight)

        root.addStretch()
        self.setLayout(root)

    # ── slots ────────────────────────────────────────────────────────────────

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select the folder containing the track2p/ subfolder",
        )
        if path:
            self.path_to_t2p = path
            self.path_display.setText(f"<code>{path}</code>")
            self.path_display.setStyleSheet("")
            self.run_button.setEnabled(True)

    def _run(self):
        self.main_window.central_widget.data_management.import_files(
            self.path_to_t2p,
            plane=int(self.plane_box.text()),
            trace_type=self.trace_choice.currentText(),
            channel=self.channel_choice.currentText(),
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line