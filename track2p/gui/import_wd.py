from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QLabel,
    QFormLayout,
    QComboBox,
    QFrame,
    QVBoxLayout,
)
from qtpy.QtCore import Qt

from track2p.gui.cell_plot import ImageMode
from ..logs import get_logger

logger = get_logger(__name__)


class ImportWindow(QWidget):
    def __init__(self, main_wd):
        super().__init__()
        self.main_window = main_wd
        self.path_to_t2p = None

        root = QVBoxLayout()
        root.setSpacing(12)

        # ── Folder section ───────────────────────────────────────────────
        folder_section = QFormLayout()

        folder_header = QLabel("📂 Select track2p output folder")
        folder_header.setStyleSheet("font-weight: bold; font-size: 11pt;")

        hint = QLabel("Select suite2p folder containing plane directories.")
        hint.setWordWrap(True)

        self.import_button = QPushButton("Browse…")
        self.import_button.setFixedWidth(90)
        self.import_button.clicked.connect(self._browse)

        self.path_display = QLabel("<i>No folder selected</i>")
        self.path_display.setStyleSheet("color: grey;")

        folder_section.addRow(folder_header)
        folder_section.addRow(hint)
        folder_section.addRow("Folder:", self.import_button)
        folder_section.addRow("Selected:", self.path_display)

        root.addLayout(folder_section)
        root.addWidget(_hline())

        # ── Options ──────────────────────────────────────────────────────
        options_header = QLabel("⚙️ Analysis options")
        options_header.setStyleSheet("font-weight: bold; font-size: 11pt;")
        root.addWidget(options_header)

        options = QFormLayout()

        self.plane_box = QLineEdit("0")
        self.plane_box.setFixedWidth(60)

        self.trace_choice = QComboBox()
        self.trace_choice.addItems(["F", "dF/F0", "spks"])

        self.channel_choice = QComboBox()

        # core enum modes (SAFE PATH)
        self.channel_choice.addItem("Functional (meanImg)", ImageMode.FUNC_MEAN)
        self.channel_choice.addItem(
            "Functional enhanced (meanImgE)", ImageMode.FUNC_MEAN_ENH
        )
        self.channel_choice.addItem("Anatomical (chan2)", ImageMode.ANAT_MEAN)
        self.channel_choice.addItem(
            "Anatomical enhanced (chan2E)", ImageMode.ANAT_MEAN_ENH
        )

        # optional non-image features (kept, but clearly tagged)
        self.channel_choice.addItem("Vcorr (no mean image)", "vcorr")
        self.channel_choice.addItem("Max projection (no mean image)", "max_proj")

        options.addRow("Plane index:", self.plane_box)
        options.addRow("Trace type:", self.trace_choice)
        options.addRow("Mean image channel:", self.channel_choice)

        root.addLayout(options)
        root.addWidget(_hline())

        # ── Run ──────────────────────────────────────────────────────────
        self.run_button = QPushButton("Load && Run")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run)

        root.addWidget(self.run_button, alignment=Qt.AlignRight)
        root.addStretch()

        self.setLayout(root)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select suite2p folder")
        if path:
            self.path_to_t2p = path
            self.path_display.setText(f"<code>{path}</code>")
            self.run_button.setEnabled(True)

    def _resolve_channel(self):
        value = self.channel_choice.currentData()

        # 1. real enum → perfect path
        if isinstance(value, ImageMode):
            return value

        # 2. unsupported feature → map to safe fallback
        if value in ("vcorr", "max_proj"):
            logger.warning(
                f"'{value}' has no mean image → falling back to FUNC_MEAN_ENH"
            )
            return ImageMode.FUNC_MEAN_ENH

        # 3. fallback safety
        return ImageMode.FUNC_MEAN_ENH

    def _run(self):
        self.main_window.central_widget.data_management.import_files(
            self.path_to_t2p,
            plane=int(self.plane_box.text()),
            trace_type=self.trace_choice.currentText(),
            channel=self._resolve_channel(),
        )


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line
