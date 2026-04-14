import numpy as np
from pathlib import Path

from ..logs import get_logger

logger = get_logger(__name__)


def save_track_ops(track_ops):
    # remove attributes taking a lot of memory (e.g. rois etc.)
    del track_ops.all_ds_all_roi_array_mov
    del track_ops.all_ds_all_roi_array_ref
    del track_ops.all_ds_all_roi_array_reg

    track_ops_dict = track_ops.to_dict()  # convert to dictionary for compatibility

    save_path = Path(track_ops.save_path)

    np.save(save_path / "track_ops.npy", track_ops_dict, allow_pickle=True)
    logger.info("Saved track_ops.npy in " + track_ops.save_path)


def save_all_pl_match_mat(all_pl_match_mat, track_ops):
    save_path = Path(track_ops.save_path)

    for i, all_pl_match_mat in enumerate(all_pl_match_mat):
        np.save(
            save_path / f"plane{i}_match_mat.npy", all_pl_match_mat, allow_pickle=True
        )


def npy_to_s2p(track_ops):

    for plane in range(track_ops.nplanes):
        logger.info(f"Processing plane {plane + 1}/{track_ops.nplanes}...")

        for ds_path in track_ops.all_ds_path:
            ds_path = Path(ds_path)

            # 1) define numpy and suite2p data paths (+ make sure they exist)
            npy_path = ds_path / "data_npy" / f"plane{plane}"
            s2p_path = Path(str(npy_path).replace("data_npy", "suite2p"))

            if not s2p_path.exists():
                s2p_path.mkdir(parents=True, exist_ok=False)
            else:
                logger.info(
                    f"Directory {s2p_path} already exists, skipping... (Delete or rename it if you want to overwrite)"
                )
                continue

            # 2) Load numpy data
            F = np.load(npy_path / "F.npy")
            fov = np.load(npy_path / "fov.npy")
            rois = np.load(npy_path / "rois.npy")

            # 3) Convert and save data in suite2p format
            np.save(s2p_path / "F.npy", F)

            ops = {"meanImg": fov}
            ops["nchannels"] = 1  # Assuming single channel
            ops["fs"] = 30  # Assuming a sampling frequency of 30 Hz
            ops["nframes"] = F.shape[1]
            np.save(s2p_path / "ops.npy", ops)

            # stat is a list of dictionaries, each with keys 'xpix', 'ypix'
            stat = []
            for i in range(rois.shape[0]):
                ypix, xpix = np.where(rois[i] > 0)
                med = [int(np.median(ypix).item()), int(np.median(xpix).item())]
                stat.append({"xpix": xpix, "ypix": ypix, "med": med})

            np.save(s2p_path / "stat.npy", stat)

            # iscell is two columns of 1 the columns are length n_cells
            n_cells = len(stat)
            iscell = np.ones((n_cells, 2), dtype=int)
            np.save(s2p_path / "iscell.npy", iscell)
