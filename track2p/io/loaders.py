from pathlib import Path
import numpy as np
from ..logs import get_logger
logger = get_logger(__name__)

def load_track_ops(track_ops_path: Path):
    track_ops_path = Path(track_ops_path)
    file_path = track_ops_path / "track_ops_postreg.npy"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    track_ops = np.load(file_path, allow_pickle=True).item()
    return track_ops


def load_stat_ds_plane(track_ops_path: Path, track_ops, plane_idx: int = 0):
    track_ops_path = Path(track_ops_path)

    stat_path = track_ops_path / f"plane{plane_idx}" / "stat.npy"
    iscell_path = track_ops_path / f"plane{plane_idx}" / "iscell.npy"

    if not stat_path.exists():
        raise FileNotFoundError(f"Missing stat file: {stat_path}")
    if not iscell_path.exists():
        raise FileNotFoundError(f"Missing iscell file: {iscell_path}")

    stat = np.load(stat_path, allow_pickle=True)
    iscell = np.load(iscell_path, allow_pickle=True)

    len_stat_allcell = len(stat)

    # Apply filtering
    if getattr(track_ops, "iscell_thr", None) is None:
        stat = stat[iscell[:, 0] == 1]
    else:
        stat = stat[iscell[:, 1] > track_ops.iscell_thr]

    len_stat_iscell = len(stat)

    dataset_name = track_ops_path.name

    logger.info(f"Loading ROIs for plane {plane_idx} in dataset '{dataset_name}'")
    logger.info(
        f"Chose {len_stat_iscell}/{len_stat_allcell} ROIs "
        f"(iscell_thr={track_ops.iscell_thr})"
    )

    stat_summary = {
        "len_stat_allcell": len_stat_allcell,
        "len_stat_iscell": len_stat_iscell,
        "iscell_thr": track_ops.iscell_thr,
    }

    return stat, stat_summary


def get_all_roi_array_from_stat(stat, track_ops):
    # safer access with explicit indexing
    example_img = track_ops.all_ds_avg_ch1[0][0]
    n_xpix, n_ypix = example_img.shape

    all_roi_array = np.zeros((n_xpix, n_ypix, len(stat)), dtype=bool)

    for i, roi in enumerate(stat):
        roi_xpix = np.asarray(roi["xpix"], dtype=int)
        roi_ypix = np.asarray(roi["ypix"], dtype=int)

        # Direct assignment without intermediate grid (faster)
        all_roi_array[roi_ypix, roi_xpix, i] = True

    return all_roi_array