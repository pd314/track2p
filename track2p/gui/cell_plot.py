import time
from qtpy.QtCore import Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import skimage
from enum import IntEnum


# =========================================================
# ENUM
# =========================================================
class ImageMode(IntEnum):
    FUNC_MEAN = 0
    FUNC_MEAN_ENH = 1
    ANAT_MEAN = 2
    ANAT_MEAN_ENH = 3


# =========================================================
# SAFE PARSER
# =========================================================
def _parse_channel(channel):
    if channel is None:
        return ImageMode.FUNC_MEAN_ENH

    if isinstance(channel, ImageMode):
        return channel

    try:
        return ImageMode(int(channel))
    except Exception:
        raise ValueError(f"Invalid channel value: {channel}")


# =========================================================
# WIDGET
# =========================================================
class CellPlotWidget(FigureCanvas):
    '''This class is used to view and interact with the mean image of each recording (day)'''

    cell_selected = Signal(int)

    def __init__(self, tab=None, ops=None, stat_t2p=None, f_t2p=None,
                 colors=None, update_selection_callback=None,
                 all_f_t2p=None, all_stat_t2p=None, all_ops=None,
                 initial_colors=None, channel=None):

        self.fig, self.ax_image = plt.subplots(1, 1)
        self.fig.set_facecolor('black')
        super().__init__(self.fig)

        self.ops = ops
        self.stat_t2p = stat_t2p
        self.f_t2p = f_t2p
        self.all_fluorescence = all_f_t2p
        self.all_stat_t2p = all_stat_t2p
        self.all_ops = all_ops
        self.colors = colors
        self.initial_colors = initial_colors

        # -----------------------------
        # SAFE ENUM PARSING HERE
        # -----------------------------
        self.channel = _parse_channel(channel)

        self.all_img, self.img = self.load_all_imgs()

        self.selected_cell_index = None
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.update_selection_callback = update_selection_callback

        self.nb_cells = len(self.stat_t2p)

        self.plot_cells()
        self.initialize_interactions()

    # =========================================================
    def load_all_imgs(self):
        print('loading all images')
        print('channel img:', self.channel)

        all_img = []
        img = None

        # -------------------------
        # FUNCTIONAL MODES
        # -------------------------
        if self.channel in (ImageMode.FUNC_MEAN, ImageMode.FUNC_MEAN_ENH):

            key = 'meanImg' if self.channel == ImageMode.FUNC_MEAN else 'meanImgE'

            if key not in self.ops:
                print(f"[WARN] missing {key}")
                return [], None

            img = self.ops[key]

            for ops in self.all_ops:
                if key in ops:
                    all_img.append(ops[key])

        # -------------------------
        # ANATOMICAL MODES
        # -------------------------
        elif self.channel in (ImageMode.ANAT_MEAN, ImageMode.ANAT_MEAN_ENH):

            key = 'meanImg_chan2' if self.channel == ImageMode.ANAT_MEAN else 'meanImg_chan2E'

            if key not in self.ops:
                print(f"[WARN] missing {key}")
                return [], None

            img = self.ops[key]

            for ops in self.all_ops:
                if key in ops:
                    all_img.append(ops[key])

        else:
            raise ValueError(f"Unknown channel mode: {self.channel}")

        return all_img, img

    # =========================================================
    def plot_cells(self):
        self.ax_image.clear()

        start = time.time()

        match_mean_img = skimage.exposure.match_histograms(
            self.img,
            self.all_img[-1],
            channel_axis=None
        )

        self.ax_image.imshow(match_mean_img, cmap='gray')

        cell_count = 0

        for cell in range(self.nb_cells):
            bin_mask = np.zeros_like(self.img)
            bin_mask[self.stat_t2p[cell]['ypix'],
                     self.stat_t2p[cell]['xpix']] = 1

            color_cell = self.colors[cell]
            self.ax_image.contour(bin_mask, levels=[0.5],
                                  colors=[color_cell], linewidths=1)
            cell_count += 1

        self.ax_image.axis('off')

        print(f'time for plotting cells: {time.time()-start}')
        print(f'Total cells plotted: {cell_count}')

        self.draw()

    # =========================================================
    def plot_cells_remix(self, keys):
        self.ax_image.clear()

        start = time.time()

        match_mean_img = skimage.exposure.match_histograms(
            self.img,
            self.all_img[-1],
            channel_axis=None
        )

        self.ax_image.imshow(match_mean_img, cmap='gray')

        cell_count = 0

        for cell in range(self.nb_cells):
            if cell in keys:
                continue

            bin_mask = np.zeros_like(self.img)
            bin_mask[self.stat_t2p[cell]['ypix'],
                     self.stat_t2p[cell]['xpix']] = 1

            color_cell = self.colors[cell]
            self.ax_image.contour(bin_mask, levels=[0.5],
                                  colors=[color_cell], linewidths=1)
            cell_count += 1

        self.ax_image.axis('off')

        print(f'time for plotting cells: {time.time()-start}')
        print(f'Total cells plotted: {cell_count}')

        self.draw()

    # =========================================================
    def underline_cell_remix(self, colors):

        for cell in range(self.nb_cells):
            bin_mask = np.zeros_like(self.img)
            bin_mask[self.stat_t2p[cell]['ypix'],
                     self.stat_t2p[cell]['xpix']] = 1

            color_cell = colors[cell]
            self.ax_image.contour(bin_mask, levels=[0.5],
                                  colors=[color_cell], linewidths=1)

        self.draw()

    # =========================================================
    def underline_cell(self, selected_cell_index):
        for cell in range(self.nb_cells):
            if cell == selected_cell_index:
                bin_mask = np.zeros_like(self.img)
                bin_mask[self.stat_t2p[cell]['ypix'],
                         self.stat_t2p[cell]['xpix']] = 1

                color_cell = self.colors[cell]
                self.ax_image.contour(bin_mask, levels=[0.5],
                                      colors=[color_cell], linewidths=3)

        self.draw()

    # =========================================================
    def remove_previous_underline(self):
        for collection in self.ax_image.collections:
            collection.set_linewidth(1)

    # =========================================================
    def initialize_interactions(self):
        self.cid_scroll = self.fig.canvas.mpl_connect(
            'scroll_event', self.on_scroll
        )

        self.initial_xlim = self.ax_image.get_xlim()
        self.initial_ylim = self.ax_image.get_ylim()

    # =========================================================
    def on_scroll(self, event):
        if event.inaxes == self.ax_image:
            current_xlim = self.ax_image.get_xlim()
            current_ylim = self.ax_image.get_ylim()

            base_scale = 0.9
            scale_factor = base_scale if event.button == 'up' else 1/base_scale

            x_data, y_data = event.xdata, event.ydata

            new_xlim = [x_data - (x_data - x) * scale_factor for x in current_xlim]
            new_ylim = [y_data - (y_data - y) * scale_factor for y in current_ylim]

            new_xlim = [max(self.initial_xlim[0],
                            min(self.initial_xlim[1], x)) for x in new_xlim]
            new_ylim = [max(self.initial_ylim[1],
                            min(self.initial_ylim[0], y)) for y in new_ylim]

            self.ax_image.set_xlim(new_xlim)
            self.ax_image.set_ylim(new_ylim)

            self.fig.canvas.draw_idle()

    # =========================================================
    def on_mouse_press(self, event):
        start = time.time()

        if event.inaxes == self.ax_image:
            x, y = event.xdata, event.ydata

            for j, cell_info in enumerate(self.stat_t2p):
                ypix = cell_info['ypix']
                xpix = cell_info['xpix']

                if np.any((xpix == int(x)) & (ypix == int(y))):
                    self.selected_cell_index = j
                    self.update_selection_callback(j)
                    print(f"Cell selected: {j}", flush=True)
                    break

        print(f'time taken for update: {time.time()-start}')