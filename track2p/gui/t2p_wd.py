import logging
from pathlib import Path
from enum import IntEnum

from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QLabel,
    QFormLayout,
    QListWidget,
    QMessageBox,
    QListWidgetItem,
    QCheckBox,
    QSizePolicy,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
)
from qtpy.QtCore import Qt

from track2p.t2p import run_t2p
from track2p.ops.default import DefaultTrackOps
from track2p.gui.custom_wd import CustomDialog

from ..logs import get_logger

logger = get_logger(__name__)


class RegChannel(IntEnum):
    FUNCTIONAL = 0
    ANATOMICAL = 1


class Track2pWindow(QWidget):
    """Set parameters for and launch the track2p algorithm."""

    def __init__(self, main_wd):
        super().__init__()
        self.main_window = main_wd
        self.track_ops = DefaultTrackOps()
        self.saved_directory: Path | None = None

        layout = QFormLayout()
        self.setLayout(layout)

        # ── Session folder picker ────────────────────────────────────────────
        session_hint = QLabel(
            "Select the <b>parent folder</b> whose subfolders are individual sessions "
            "(one per recording day) — each subfolder must be a suite2p output folder "
            "containing <code>plane0/</code>, <code>plane1/</code>, etc.<br>"
            "<small>Example:&nbsp;<code>D:/subject01/</code>&nbsp; containing "
            "<code>day1/</code>, <code>day2/</code>, …</small>"
        )
        session_hint.setWordWrap(True)
        session_hint.setTextFormat(Qt.RichText)

        self.import_recording_button = QPushButton("Browse…")
        self.import_recording_button.clicked.connect(self._browse_sessions)
        layout.addRow(session_hint, self.import_recording_button)

        self.path_recording = QLabel("<i>No folder selected</i>")
        self.path_recording.setTextFormat(Qt.RichText)
        self.path_recording.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addRow("Selected parent folder:", self.path_recording)

        # ── Session list shuttle ─────────────────────────────────────────────
        shuttle_hint = QLabel(
            "Select sessions in the left list and press <b>→</b> to add them to the "
            "track2p run (right list), in chronological order. Use <b>←</b> to remove."
        )
        shuttle_hint.setWordWrap(True)

        self.computer_file_list = QListWidget()
        self.computer_file_list.setFixedHeight(200)
        self.paths_list = QListWidget()
        self.paths_list.setFixedHeight(200)

        btn_right = QPushButton("→")
        btn_right.clicked.connect(self._move_to_run_list)
        btn_left = QPushButton("←")
        btn_left.clicked.connect(self._move_to_available_list)

        arrow_col = QVBoxLayout()
        arrow_col.addStretch()
        arrow_col.addWidget(btn_right)
        arrow_col.addWidget(btn_left)
        arrow_col.addStretch()

        shuttle = QHBoxLayout()
        shuttle.addWidget(self.computer_file_list)
        shuttle.addLayout(arrow_col)
        shuttle.addWidget(self.paths_list)

        layout.addRow(shuttle_hint)
        layout.addRow(shuttle)

        # ── Input format ─────────────────────────────────────────────────────
        self.format_box = QComboBox()
        self.format_box.addItems(["suite2p", "npy"])
        self.format_box.currentIndexChanged.connect(self._on_format_changed)
        layout.addRow("Input format:", self.format_box)

        # ── suite2p ROI selection ────────────────────────────────────────────
        self.checkbox_manual = QCheckBox("Manually curated (iscell == 1)")
        self.checkbox_thr = QCheckBox("Probability threshold (iscell[:,1] > thr)")
        self.checkbox_thr.stateChanged.connect(self._on_thr_toggled)

        self.iscell_thr_box = QLineEdit("0.5")
        self.iscell_thr_box.setFixedWidth(50)
        self.iscell_thr_box.setToolTip(
            "Minimum iscell probability to accept an ROI (0–1)"
        )
        self.iscell_thr_box.setVisible(False)

        roi_col = QVBoxLayout()
        roi_col.addWidget(self.checkbox_manual)
        roi_col.addWidget(self.checkbox_thr)
        layout.addRow("ROI selection method:", roi_col)
        layout.addRow("iscell probability threshold:", self.iscell_thr_box)

        # ── Registration options ─────────────────────────────────────────────
        self.reg_chan_box = QComboBox()
        self.reg_chan_box.addItem("Functional (recommended)", RegChannel.FUNCTIONAL)
        self.reg_chan_box.addItem("Anatomical", RegChannel.ANATOMICAL)
        self.reg_chan_box.setToolTip("Select the channel used for registration")

        layout.addRow("Registration channel:", self.reg_chan_box)

        self.transform_box = QComboBox()
        self.transform_box.addItems(["affine", "rigid"])
        layout.addRow("Registration transform:", self.transform_box)

        self.thr_method_box = QComboBox()
        self.thr_method_box.addItems(["min", "otsu"])
        self.thr_method_box.setCurrentText("otsu")
        self.thr_method_box.setToolTip("Method used to threshold the IoU histogram")
        layout.addRow("IoU threshold method:", self.thr_method_box)

        # ── Output folder ────────────────────────────────────────────────────
        output_hint = QLabel(
            "Select the folder where results will be saved. "
            "A <code>track2p/</code> subfolder will be created inside it."
        )
        output_hint.setWordWrap(True)
        output_hint.setTextFormat(Qt.RichText)

        self.save_dir_button = QPushButton("Browse…")
        self.save_dir_button.clicked.connect(self._browse_output)
        layout.addRow(output_hint, self.save_dir_button)

        self.save_path_label = QLabel("<i>No folder selected</i>")
        self.save_path_label.setTextFormat(Qt.RichText)
        self.save_path_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addRow("Output folder:", self.save_path_label)

        # ── Misc options ─────────────────────────────────────────────────────
        self.checkbox_s2p_format = QCheckBox()
        layout.addRow(
            "Also save matched cells in suite2p format (one file per session):",
            self.checkbox_s2p_format,
        )

        # ── Run ──────────────────────────────────────────────────────────────
        self.run_button = QPushButton("▶  Run track2p")
        self.run_button.setFixedHeight(32)
        self.run_button.clicked.connect(self._run)
        layout.addRow(self.run_button)

        layout.addRow(
            QLabel(
                "<small><i>Progress is logged to the terminal where the GUI was launched.</i></small>"
            )
        )

    # ── slots ────────────────────────────────────────────────────────────────

    def _browse_sessions(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select the parent folder containing per-session suite2p subfolders",
        )
        if not directory:
            return

        root = Path(directory)
        self.saved_directory = root
        self.path_recording.setText(f"<code>{root}</code>")
        self.save_path_label.setText(f"<code>{root}</code>")

        self.computer_file_list.clear()
        for subfolder in sorted(root.iterdir()):
            if subfolder.is_dir():
                item = QListWidgetItem(str(subfolder))
                item.setData(Qt.UserRole, str(subfolder))
                self.computer_file_list.addItem(item)

    def _browse_output(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select the folder where the track2p/ output subfolder will be created",
        )
        if directory:
            self.saved_directory = Path(directory)
            self.save_path_label.setText(f"<code>{directory}</code>")

    def _on_thr_toggled(self, state):
        self.iscell_thr_box.setVisible(state == Qt.Checked)

    def _on_format_changed(self):
        is_s2p = self.format_box.currentText() == "suite2p"
        for w in (self.checkbox_manual, self.checkbox_thr, self.reg_chan_box):
            w.setVisible(is_s2p)
        self.iscell_thr_box.setVisible(is_s2p and self.checkbox_thr.isChecked())

    def _move_to_run_list(self):
        for item in self.computer_file_list.selectedItems():
            self.computer_file_list.takeItem(self.computer_file_list.row(item))
            self.paths_list.addItem(item)

    def _move_to_available_list(self):
        for item in self.paths_list.selectedItems():
            self.paths_list.takeItem(self.paths_list.row(item))
            self.computer_file_list.addItem(item)

    def _run(self):
        self.track_ops.all_ds_path = [
            self.paths_list.item(i).data(Qt.UserRole)
            for i in range(self.paths_list.count())
        ]
        self.track_ops.save_path = str(self.saved_directory)
        self.track_ops.input_format = self.format_box.currentText()

        self.track_ops.reg_chan = self.reg_chan_box.currentData()

        self.track_ops.transform_type = self.transform_box.currentText()
        self.track_ops.thr_method = self.thr_method_box.currentText()
        self.track_ops.save_in_s2p_format = self.checkbox_s2p_format.isChecked()

        if self.checkbox_thr.isChecked():
            self.track_ops.iscell_thr = float(self.iscell_thr_box.text())
        else:
            self.track_ops.iscell_thr = None

        logger.info(
            "Starting track2p — transform=%s, thr_method=%s, iscell_thr=%s",
            self.track_ops.transform_type,
            self.track_ops.thr_method,
            self.track_ops.iscell_thr,
        )

        run_t2p(self.track_ops)
        self._offer_open_gui()

    def _offer_open_gui(self):
        reply = QMessageBox.question(
            self,
            "Run complete",
            "Run completed successfully!\nOpen the results viewer?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.dialog = CustomDialog(
                self.main_window,
                str(self.saved_directory),
                self.reg_chan_box.currentText(),
            )
            self.dialog.exec_()
