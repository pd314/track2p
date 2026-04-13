from track2p.ops.default import DefaultTrackOps
from types import SimpleNamespace
from pathlib import Path

from track2p.io.s2p_loaders import load_all_imgs, check_nplanes, load_all_ds_stat_iscell, load_all_ds_mean_img, load_all_ds_centroids
from track2p.io.savers import npy_to_s2p, save_track_ops, save_all_pl_match_mat

from track2p.register.loop import run_reg_loop, reg_all_ds_all_roi
from track2p.register.utils import get_all_ds_img_for_reg, get_all_ref_nonref_inters

from track2p.plot.progress import plot_all_planes
from track2p.plot.output import plot_reg_img_output, plot_thr_met_hist, plot_n_matched_roi, plot_roi_reg_output, plot_roi_match_multiplane, plot_allroi_match_multiplane

from track2p.match.loop import get_all_ds_assign, get_all_pl_match_mat
import numpy as np
import scipy
import pandas as pd


def _plane_path(ds_path: Path, plane: int) -> Path:
    """ds_path already points at the suite2p folder; just append planeN."""
    return Path(ds_path) / f"plane{plane}"


def _cell_mask(iscell: np.ndarray, thr) -> np.ndarray:
    return iscell[:, 1] > thr if thr is not None else iscell[:, 0] == 1


def run_t2p(track_ops):

    # 1) initialise save paths for figures and matched neurons output
    track_ops.init_save_paths()

    # 2) Load data
    check_nplanes(track_ops)

    if track_ops.input_format == "npy":
        print("Converting npy data to track2p-compatible format...")
        npy_to_s2p(track_ops)

    all_ds_avg_ch1, all_ds_avg_ch2, all_ds_avg_ch1E, all_ds_avg_ch2E = load_all_imgs(track_ops, return_es=True)

    # 3) Plot available planes for registration
    plot_all_planes(all_ds_avg_ch1, track_ops)
    if track_ops.nchannels == 2:
        plot_all_planes(all_ds_avg_ch2, track_ops, ch="anatomical")

    # 4) Register based on chosen channel
    all_ds_ref_img, all_ds_mov_img = get_all_ds_img_for_reg(all_ds_avg_ch1, all_ds_avg_ch2, track_ops)
    all_ds_mov_img_reg, all_ds_reg_params = run_reg_loop(all_ds_ref_img, all_ds_mov_img, track_ops)
    plot_reg_img_output(track_ops)

    # 5) Apply computed transform to all ROIs
    all_ds_all_roi_ref, all_ds_all_roi_mov, all_ds_all_roi_reg, all_ds_roi_counter = reg_all_ds_all_roi(all_ds_reg_params, track_ops)

    # 6) Generate 'yellow intersection' plots
    all_ds_ref_reg_inters = get_all_ref_nonref_inters(all_ds_all_roi_ref, all_ds_all_roi_reg, track_ops)
    all_ds_ref_mov_inters = get_all_ref_nonref_inters(all_ds_all_roi_ref, all_ds_all_roi_mov, track_ops)
    track_ops.all_ds_ref_mov_inters = all_ds_ref_mov_inters
    track_ops.all_ds_ref_reg_inters = all_ds_ref_reg_inters

    if track_ops.show_roi_reg_output:
        plot_roi_reg_output(track_ops)

    # 7) Optimal assignments for all dataset pairs
    all_ds_assign, all_ds_assign_thr, all_ds_thr_met, all_ds_thr = get_all_ds_assign(track_ops, all_ds_all_roi_ref, all_ds_all_roi_reg)
    plot_thr_met_hist(all_ds_thr_met, all_ds_thr, track_ops)
    plot_n_matched_roi(all_ds_thr_met, all_ds_thr, track_ops)

    # 8) Match matrices
    all_pl_match_mat = get_all_pl_match_mat(all_ds_all_roi_ref, all_ds_assign_thr, track_ops)

    # 9) Save results
    save_track_ops(track_ops)
    save_all_pl_match_mat(all_pl_match_mat, track_ops)

    print("Generating suite2p indices")
    generate_suite2p_indices(track_ops)

    # 10) Save in suite2p format
    if track_ops.save_in_s2p_format:
        print("Saving in suite2p format...")
        save_in_s2p_format(track_ops)

    # 11) Plot results
    print("Finished with algorithm!\n\nGenerating plots (this can take some time)...\n\n")
    all_ds_stat_iscell = load_all_ds_stat_iscell(track_ops)
    all_ds_centroids   = load_all_ds_centroids(all_ds_stat_iscell, track_ops)
    all_ds_mean_img    = load_all_ds_mean_img(track_ops)

    plot_roi_match_multiplane(all_ds_mean_img, all_ds_centroids, all_pl_match_mat, track_ops, win_size=track_ops.win_size)
    plot_allroi_match_multiplane(all_ds_mean_img, all_pl_match_mat, track_ops)

    if track_ops.nchannels == 2:
        all_ds_mean_img_ch2 = load_all_ds_mean_img(track_ops, ch=2)
        plot_roi_match_multiplane(all_ds_mean_img_ch2, all_ds_centroids, all_pl_match_mat, track_ops, win_size=track_ops.win_size, ch=2)
        plot_allroi_match_multiplane(all_ds_mean_img_ch2, all_pl_match_mat, track_ops, ch=2)

    print("\n\n\nDone!\n\n\n")


def generate_suite2p_indices(track_ops):
    save_path = Path(track_ops.save_path)
    thr = track_ops.iscell_thr

    for plane in range(track_ops.nplanes):
        t2p_match_mat = np.load(save_path / f"plane{plane}_match_mat.npy", allow_pickle=True)

        all_iscell = [
            np.load(_plane_path(ds_path, plane) / "iscell.npy", allow_pickle=True)
            for ds_path in track_ops.all_ds_path
        ]

        true_indices = []
        for line in t2p_match_mat:
            indexes = []
            for day, index_match in enumerate(line):
                if index_match is None:
                    indexes.append(None)
                else:
                    valid_indices = np.where(_cell_mask(all_iscell[day], thr))[0]
                    indexes.append(valid_indices[index_match])
            true_indices.append(indexes)

        true_indices     = np.array([[int(x)   if x is not None else None  for x in row] for row in true_indices])
        true_indices_nan = np.array([[float(x) if x is not None else np.nan for x in row] for row in true_indices])

        np.save(save_path / f"plane{plane}_suite2p_indices.npy",     true_indices)
        np.save(save_path / f"plane{plane}_suite2p_indices_nan.npy", true_indices_nan)
        scipy.io.savemat(str(save_path / f"plane{plane}_suite2p_indices.mat"), {"data": true_indices_nan})

        column_names = [Path(ds_path).name for ds_path in track_ops.all_ds_path]
        pd.DataFrame(true_indices, columns=column_names).to_csv(
            save_path / f"plane{plane}_suite2p_indices.csv",
            index=False, sep=";", na_rep="NaN",
        )

        print(true_indices.dtype)
        print(true_indices_nan.dtype)


def save_in_s2p_format(track_ops):
    folderpath  = Path(track_ops.save_path)
    track_ops   = SimpleNamespace(**np.load(folderpath / "track_ops.npy", allow_pickle=True).item())
    thr         = track_ops.iscell_thr
    two_channel = track_ops.nchannels == 2

    for j in range(track_ops.nplanes):
        print(f"plane {j}")

        t2p_match_mat = np.load(folderpath / f"plane{j}_match_mat.npy", allow_pickle=True)
        matched_rows  = t2p_match_mat[~np.any(t2p_match_mat == None, axis=1)]  # noqa: E711
        matched_idx   = matched_rows.astype(int)

        per_ds = []
        for i, ds_path in enumerate(track_ops.all_ds_path):
            pp = _plane_path(ds_path, j)

            def npl(fname):
                return np.load(pp / fname, allow_pickle=True)

            ops    = npl("ops.npy").item()
            stat   = npl("stat.npy")
            f      = npl("F.npy")
            fneu   = npl("Fneu.npy")
            spks   = npl("spks.npy")
            iscell = npl("iscell.npy")

            mask = _cell_mask(iscell, thr)

            arrays = dict(stat=stat[mask], f=f[mask], fneu=fneu[mask],
                          spks=spks[mask], iscell=iscell[mask], ops=ops)

            if two_channel:
                arrays.update(
                    f_chan2   = npl("F_chan2.npy")[mask],
                    fneu_chan2= npl("Fneu_chan2.npy")[mask],
                    redcell   = npl("redcell.npy")[mask],
                )

            idx = matched_idx[:, i]
            result = {k: (v[idx] if k != "ops" else v) for k, v in arrays.items()}
            per_ds.append(result)

        output_root = folderpath / "matched_suite2p"
        for i, ds_path in enumerate(track_ops.all_ds_path):
            plane_out = output_root / Path(ds_path).name / "suite2p" / f"plane{j}"
            plane_out.mkdir(parents=True, exist_ok=True)

            d = per_ds[i]
            np.save(plane_out / "stat.npy",   d["stat"])
            np.save(plane_out / "F.npy",      d["f"])
            np.save(plane_out / "ops.npy",    d["ops"])
            np.save(plane_out / "iscell.npy", d["iscell"])
            np.save(plane_out / "Fneu.npy",   d["fneu"])
            np.save(plane_out / "spks.npy",   d["spks"])

            if two_channel:
                np.save(plane_out / "F_chan2.npy",    d["f_chan2"])
                np.save(plane_out / "Fneu_chan2.npy", d["fneu_chan2"])
                np.save(plane_out / "redcell.npy",    d["redcell"])