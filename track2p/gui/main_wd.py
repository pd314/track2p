from track2p.gui.window_manager import WindowManager
from track2p.gui.toolbar import Toolbar
from track2p.gui.statusbar import StatusBar
from track2p.gui.data_management import DataManagement
from track2p.gui.central_widget import CentralWidget

from qtpy.QtWidgets import QApplication, QMainWindow
from qtpy.QtCore import QObject, QThread, Signal

from ..logs import get_logger

logger = get_logger(__name__)


class InitWorker(QObject):
    finished = Signal(object)
    progress = Signal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        try:
            self.progress.emit("Initializing data manager...")

            # IMPORTANT:
            # No QWidget/QObject GUI stuff here
            data_management = DataManagement(None)

            self.progress.emit("Loading data...")
            if hasattr(data_management, "load_if_needed"):
                data_management.load_if_needed()

            self.progress.emit("Finalizing...")

            self.finished.emit(data_management)

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.finished.emit(None)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.window_manager = WindowManager(self)

        self.central_widget = CentralWidget(self)
        self.toolbar = Toolbar(self)
        self.status_bar = StatusBar(self)

        self.initUI()

        # placeholders
        self.data_management = None

        # start heavy init in background
        self.start_background_init()

    def initUI(self):

        self.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #666; }"
            "QTabWidget::tab-bar { alignment: center; }"
            "QTabBar::tab { background-color: #666; color: white; }"
            "QTabBar::tab:selected { background-color: #222; color: white; }"
            "QSplitter::handle { background: #888; }"
            "QFrame { background-color: black; color: black; border: 1px solid black; }"
            "QLabel { color: black; background-color: none; border: none; font-size: 13px }"
            "QPushButton { background-color: #666; color: white; border: 1px solid #888; }"
            "QPushButton:hover { background-color: #888; color: white; }"
            "QPushButton:pressed { background-color: #333; color: white; }"
            "QToolButton:pressed { background-color: #888; }"
            "QComboBox { background-color: black; color: white; }"
            "QComboBox QAbstractItemView { background-color: #666; color: white; }"
        )

        self.setWindowTitle("track2p GUI")

        self.setCentralWidget(self.central_widget)
        self.addToolBar(self.toolbar)
        self.setStatusBar(self.status_bar)

        QApplication.setStyle("Cleanlooks")

        self.showMaximized()

    def start_background_init(self):

        self.thread = QThread()
        self.worker = InitWorker()

        self.worker.moveToThread(self.thread)

        # start work
        self.thread.started.connect(self.worker.run)

        # progress updates (optional UI hook)
        self.worker.progress.connect(self.on_init_progress)

        # finish
        self.worker.finished.connect(self.on_background_done)

        # cleanup (VERY IMPORTANT)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_init_progress(self, msg: str):
        logger.debug(f"[Init] {msg}")
        self.statusBar().showMessage(msg)

    def on_background_done(self, data_management):

        logger.debug("[MainWindow] background init complete")

        if data_management is None:
            self.statusBar().showMessage("Initialization failed")
            return

        self.data_management = data_management

        if hasattr(self.central_widget, "set_data_management"):
            self.central_widget.set_data_management(data_management)

        if hasattr(self.toolbar, "set_data_management"):
            self.toolbar.set_data_management(data_management)

        self.statusBar().showMessage("Ready")
