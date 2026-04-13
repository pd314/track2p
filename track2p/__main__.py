from pathlib import Path
import logging

from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon

from track2p.gui.main_wd import MainWindow
from track2p.logs import setup_logger

if __name__ == '__main__':

    logger = setup_logger(
        name="track2p",
        level=logging.DEBUG,
        to_console=True
    )

    app = QApplication([])

    base_dir = Path(__file__).resolve().parent
    icon_path = base_dir / "resources" / "logo.png"

    logger.debug(f"Resolved icon path: {icon_path}")

    if not icon_path.exists():
        logger.warning(f"Icon file not found: {icon_path}")
    else:
        app.setWindowIcon(QIcon(str(icon_path)))

    mainWindow = MainWindow()
    mainWindow.setWindowTitle("track2p")
    mainWindow.show()

    logger.info("track2p GUI started")

    app.exec()