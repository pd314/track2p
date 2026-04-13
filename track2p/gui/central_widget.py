from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QTabWidget, QVBoxLayout, QWidget, QSplitter,
    QHBoxLayout, QFrame
)

from track2p.gui.fluo_plot import FluorescencePlotWidget
from track2p.gui.roi_plot import ZoomPlotWidget
from track2p.gui.cell_plot import CellPlotWidget
from track2p.gui.data_management import DataManagement

from ..logs import get_logger
logger = get_logger(__name__)

class CentralWidget(QWidget):

    # =========================================================
    # ENUM (IMAGE MODE SELECTOR)
    # =========================================================
    class ImageMode:
        FUNC_MEAN = 0
        FUNC_MEAN_ENH = 1
        ANAT_MEAN = 2
        ANAT_MEAN_ENH = 3

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.fluorescences_plotting = None
        self.rois_plotting = None
        self.selected_roi = None
        self.cell_plot = None
        self.track_ops_dict = None

        self.data_management = DataManagement(self)

        # -----------------------------------------------------
        # DEFAULT IMAGE MODE (IMPORTANT)
        # -----------------------------------------------------
        self.image_mode = self.ImageMode.FUNC_MEAN_ENH

        # NEVER assume it is valid yet
        self.vector_curation_t2p = None

        self.init_central_widget()

    # =========================================================
    def init_central_widget(self):

        self.top = QFrame()
        self.top.setFrameShape(QFrame.StyledPanel)
        self.top_layout = QHBoxLayout(self.top)

        self.top_right = QFrame()
        self.top_right.setFrameShape(QFrame.StyledPanel)
        self.top_layout_right = QVBoxLayout(self.top_right)

        self.tabs = QTabWidget(self)

        self.splitter1 = QSplitter(Qt.Horizontal)
        self.splitter1.addWidget(self.tabs)
        self.splitter1.addWidget(self.top)
        self.splitter1.setSizes([100, 100])

        self.splitter2 = QSplitter(Qt.Horizontal)
        self.splitter2.addWidget(self.top_right)

        self.splitter3 = QSplitter(Qt.Vertical)
        self.splitter3.addWidget(self.splitter1)
        self.splitter3.addWidget(self.splitter2)
        self.splitter3.setSizes([100, 100])

        layout = QVBoxLayout()
        layout.addWidget(self.splitter3)
        self.setLayout(layout)

    # =========================================================
    def create_mean_img(self, channel):

        # allow override later (or ignore parameter)
        if channel is not None:
            self.image_mode = channel

        for i, (ops, stat_t2p) in enumerate(
            zip(self.data_management.all_ops, self.data_management.all_stat_t2p)
        ):
            tab = QWidget()

            self.cell_plot = CellPlotWidget(
                tab,
                ops=ops,
                stat_t2p=stat_t2p,
                f_t2p=self.data_management.all_f_t2p[i],
                colors=self.data_management.colors,
                update_selection_callback=self.update_selection,
                all_f_t2p=self.data_management.all_f_t2p,
                all_ops=self.data_management.all_ops,
                channel=self.image_mode
            )

            layout = QVBoxLayout(tab)
            layout.addWidget(self.cell_plot)
            tab.setLayout(layout)

            self.tabs.addTab(tab, f"Day {i + 1}")
            self.cell_plot.cell_selected.connect(self.update_selection)

    # =========================================================
    def create_mean_img_from_curation(self):
        import_window = self.main_window.window_manager.import_window
        t2p_window = self.main_window.window_manager.t2p_window

        if import_window is not None and import_window.plane is not None:
            self.data_management.import_files(
                import_window.path_to_t2p,
                import_window.plane,
                import_window.trace_type,
                import_window.channel
            )

        elif t2p_window is not None and t2p_window.saved_directory is not None:
            self.data_management.import_files(
                t2p_window.saved_directory,
                t2p_window.dialog.plane,
                t2p_window.dialog.trace_type,
                t2p_window.dialog.channel
            )

        else:
            logger.warning("Both import_window and t2p_window are not initialized. Cannot create mean image from curation.")

    # =========================================================
    def clear(self):
        self.data_management.reset_attributes()

        if self.fluorescences_plotting:
            self.top_layout_right.removeWidget(self.fluorescences_plotting)
            self.fluorescences_plotting.deleteLater()
            self.fluorescences_plotting = None

        if self.rois_plotting:
            self.top_layout.removeWidget(self.rois_plotting)
            self.rois_plotting.deleteLater()
            self.rois_plotting = None

        while self.tabs.count():
            self.tabs.removeTab(0)

    # =========================================================
    def update_selection(self, selected_cell_index):

        self.selected_roi = selected_cell_index
        self.main_window.status_bar.spin_box.setValue(selected_cell_index)

        vec = self.vector_curation_t2p
        if vec is None or len(vec) == 0 or selected_cell_index >= len(vec):
            state = 0
        else:
            state = vec[selected_cell_index]

        self.main_window.status_bar.roi_state_value.setText(f"{state}")

        for i in range(self.tabs.count()):
            tab_widget = self.tabs.widget(i)
            cell_object = tab_widget.findChild(CellPlotWidget)
            if cell_object:
                cell_object.remove_previous_underline()

        current_tab = self.tabs.currentWidget()
        if current_tab:
            cell_plot = current_tab.findChild(CellPlotWidget)
            if cell_plot:
                cell_plot.underline_cell(selected_cell_index)

        if self.fluorescences_plotting is None:
            self.fluorescences_plotting = FluorescencePlotWidget(
                all_f_t2p=self.data_management.all_f_t2p,
                all_ops=self.data_management.all_ops,
                colors=self.data_management.colors
            )
            self.top_layout_right.addWidget(self.fluorescences_plotting)

        if self.rois_plotting is None:
            self.rois_plotting = ZoomPlotWidget(
                all_ops=self.data_management.all_ops,
                all_stat_t2p=self.data_management.all_stat_t2p,
                colors=self.data_management.colors,
                all_iscell_t2p=self.data_management.all_iscell,
                t2p_match_mat_allday=self.data_management.t2p_match_mat_allday,
                track_ops=self.data_management.track_ops
            )
            self.top_layout.addWidget(self.rois_plotting)

        self.fluorescences_plotting.display_all_f_t2p(selected_cell_index)
        self.rois_plotting.display_zooms(selected_cell_index)

    # =========================================================
    def display_first_ROI(self, index):

        tab_widget = self.tabs.widget(0)
        cell_object = tab_widget.findChild(CellPlotWidget)

        if cell_object:
            cell_object.underline_cell(index)
            cell_object.draw()

        if self.fluorescences_plotting is None:
            self.fluorescences_plotting = FluorescencePlotWidget(
                all_f_t2p=self.data_management.all_f_t2p,
                all_ops=self.data_management.all_ops,
                colors=self.data_management.colors,
                all_stat_t2p=self.data_management.all_stat_t2p
            )
            self.top_layout_right.addWidget(self.fluorescences_plotting)

        if self.rois_plotting is None:
            self.rois_plotting = ZoomPlotWidget(
                all_ops=self.data_management.all_ops,
                all_stat_t2p=self.data_management.all_stat_t2p,
                colors=self.data_management.colors,
                all_iscell_t2p=self.data_management.all_iscell,
                t2p_match_mat_allday=self.data_management.t2p_match_mat_allday,
                track_ops=self.data_management.track_ops,
                imgs=self.cell_plot.all_img if self.cell_plot else None
            )
            self.top_layout.addWidget(self.rois_plotting)

        self.fluorescences_plotting.display_all_f_t2p(index)
        self.rois_plotting.display_zooms(index)

        vec = self.vector_curation_t2p
        if vec is None or len(vec) == 0 or index >= len(vec):
            state = 0
        else:
            state = vec[index]

        self.main_window.status_bar.roi_state_value.setText(f"{state}")