from pathlib import Path
import logging

from qtpy.QtWidgets import QApplication

from track2p.gui.main_wd import MainWindow
from track2p.logs import setup_logger


if __name__ == "__main__":
    setup_logger(
        name="track2p",
        level=logging.DEBUG,
        to_console=True,
    )

    app = QApplication([])

    base_dir = Path(__file__).resolve().parent
    icon_path = base_dir / "resources" / "logo.png"

    mainWindow = MainWindow()
    mainWindow.setWindowTitle("track2p")
    mainWindow.show()

    app.exec()
