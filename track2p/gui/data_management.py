from pathlib import Path
import numpy as np
import matplotlib.colors as mcolors
import random
from types import SimpleNamespace
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter


class DataManagement:
    def __init__(self, central_widget):
        self.central_widget = central_widget
        self.main_window = central_widget.main_window
        self.reset_attributes()

    # =========================================================
    # RESET (DO NOT CHANGE PUBLIC API)
    # =========================================================
    def reset_attributes(self):
        self.all_f_t2p = []
        self.all_ops = []
        self.all_stat_t2p = []
        self.all_iscell = []
        self.all_fneu = []

        self.colors = None
        self.colors_copy = None

        self.t2p_match_mat_allday = None
        self.track_ops = None
        self.vector_curation_t2p = np.array([])

        self.plane = None
        self.trace_type = None

    # =========================================================
    # PATH FIX (NO suite2p/suite2p BUG)
    # =========================================================
    @staticmethod
    def _suite2p_plane_path(ds_path, plane):
        ds_path = Path(ds_path)
        return ds_path / f"plane{plane}"

    # =========================================================
    # MAIN LOADER
    # =========================================================
    def import_files(self, t2p_folder_path, plane, trace_type, channel):

        t2p_folder_path = Path(t2p_folder_path)

        self.reset_attributes()
        self.plane = plane
        self.trace_type = trace_type

        if self.central_widget.fluorescences_plotting is not None:
            self.central_widget.clear()

        # -------------------------
        # MATCH MATRIX
        # -------------------------
        match_mat = np.load(
            t2p_folder_path / "track2p" / f"plane{plane}_match_mat.npy",
            allow_pickle=True
        )

        self.t2p_match_mat_allday = match_mat[~np.any(match_mat == None, axis=1)]

        track_ops_dict = np.load(
            t2p_folder_path / "track2p" / "track_ops.npy",
            allow_pickle=True
        ).item()

        track_ops = SimpleNamespace(**track_ops_dict)

        # -------------------------
        # LOAD DATASETS
        # -------------------------
        for i, ds_path in enumerate(track_ops.all_ds_path):

            suite2p_path = self._suite2p_plane_path(ds_path, plane)

            ops = np.load(suite2p_path / "ops.npy", allow_pickle=True).item()
            stat = np.load(suite2p_path / "stat.npy", allow_pickle=True)
            iscell = np.load(suite2p_path / "iscell.npy", allow_pickle=True)

            f = self._load_trace(suite2p_path, trace_type)

            stat, f = self._filter_iscell(stat, f, iscell, track_ops)

            idx = self.t2p_match_mat_allday[:, i].astype(int)

            self.all_stat_t2p.append(stat[idx])
            self.all_f_t2p.append(f[idx])
            self.all_ops.append(ops)
            self.all_iscell.append(iscell)

            if trace_type == "dF/F0":
                fneu = np.load(suite2p_path / "Fneu.npy", allow_pickle=True)
                self.all_fneu.append(self._match_fneu(fneu, iscell, idx, track_ops))

        # -------------------------
        # DF/F PROCESSING
        # -------------------------
        if trace_type == "dF/F0":
            self.all_f_t2p = [
                self.F_processing(f, fneu, ops["fs"])
                for f, fneu, ops in zip(self.all_f_t2p, self.all_fneu, self.all_ops)
            ]

        # -------------------------
        # CURATION VECTOR
        # -------------------------
        key = f"vector_curation_plane_{plane}"

        if key in track_ops_dict:
            self.vector_curation_t2p = track_ops_dict[key]
        else:
            self.vector_curation_t2p = np.ones(self.t2p_match_mat_allday.shape[0])
            track_ops_dict[key] = self.vector_curation_t2p

        # -------------------------
        # COLORS 
        # -------------------------
        key_color = f"colors_plane_{plane}"

        if key_color in track_ops_dict:   # ← FIXED
            self.colors = track_ops_dict[key_color]   # ← FIXED
        else:
            self.colors = self.generate_vibrant_colors(len(self.all_stat_t2p[0]))
            track_ops_dict[key_color] = self.colors

        # save track_ops
        np.save(
            t2p_folder_path / "track2p" / "track_ops.npy",
            track_ops_dict
        )

        self.track_ops = track_ops

        # -------------------------
        # UI
        # -------------------------
        self._setup_ui(len(self.t2p_match_mat_allday))

        self.main_window.status_bar.vector_curation_t2p = self.vector_curation_t2p

        # -------------------------
        # APPLY CURATION COLORS
        # -------------------------
        for i in range(len(self.t2p_match_mat_allday)):
            if self.vector_curation_t2p[i] == 0:
                self.colors[i] = (0.78, 0.78, 0.78)

        self.central_widget.create_mean_img(channel)
        self.central_widget.display_first_ROI(0)

    # =========================================================
    # TRACE LOADING
    # =========================================================
    def _load_trace(self, suite2p_path, trace_type):

        if trace_type == "F":
            return np.load(suite2p_path / "F.npy", allow_pickle=True)

        if trace_type == "spks":
            return np.load(suite2p_path / "spks.npy", allow_pickle=True)

        return np.load(suite2p_path / "F.npy", allow_pickle=True)

    # =========================================================
    # ISCELL FILTER
    # =========================================================
    def _filter_iscell(self, stat, f, iscell, track_ops):

        if track_ops.iscell_thr is None:
            mask = iscell[:, 0] == 1
        else:
            mask = iscell[:, 1] > track_ops.iscell_thr

        return stat[mask], f[mask]

    # =========================================================
    # NEUROPIL MATCH
    # =========================================================
    def _match_fneu(self, fneu, iscell, idx, track_ops):

        if track_ops.iscell_thr is None:
            fneu = fneu[iscell[:, 0] == 1]
        else:
            fneu = fneu[iscell[:, 1] > track_ops.iscell_thr]

        return fneu[idx]

    # =========================================================
    # UI
    # =========================================================
    def _setup_ui(self, n):
        sb = self.main_window.status_bar.spin_box
        sb.setSuffix(f"/{n - 1}")
        sb.setMinimum(0)
        sb.setMaximum(n - 1)

    # =========================================================
    # COLORS
    # =========================================================
    def generate_vibrant_colors(self, n):
        return [
            mcolors.hsv_to_rgb((random.random(), 1, np.random.uniform(0.55, 0.8)))
            for _ in range(n)
        ]

    # =========================================================
    # DF/F
    # =========================================================
    def F_processing(self, F, Fneu, fs,
                     neucoeff=0.0,
                     baseline="maximin",
                     sig_baseline=10.0,
                     win_baseline=60.0,
                     prctile_baseline=8):

        Fc = F - neucoeff * Fneu

        win = int(win_baseline * fs)

        if baseline == "maximin":
            Flow = gaussian_filter(Fc, [0., sig_baseline])
            Flow = minimum_filter1d(Flow, win)
            Flow = maximum_filter1d(Flow, win)

        elif baseline == "constant":
            Flow = np.amin(gaussian_filter(Fc, [0., sig_baseline]))

        elif baseline == "constant_prctile":
            Flow = np.percentile(Fc, prctile_baseline, axis=1, keepdims=True)

        else:
            Flow = 0

        return Fc - Flow