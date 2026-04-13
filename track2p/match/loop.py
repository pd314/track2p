import numpy as np
from skimage.filters import threshold_otsu, threshold_minimum
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor, as_completed

from track2p.match.utils import get_cost_mat, get_iou, init_all_pl_match_mat


# ============================================================
# Worker: compute ONE dataset pair + ONE plane
# ============================================================
def _process_pair(i, j, all_ds_all_roi_ref, all_ds_all_roi_reg, track_ops):

    all_roi_ref = all_ds_all_roi_ref[i][j]
    all_roi_reg = all_ds_all_roi_reg[i][j]

    # 1) cost matrix
    cost_mat, all_inds_ref_filt, all_inds_reg_filt = get_cost_mat(
        all_roi_ref, all_roi_reg, track_ops
    )

    # 2) assignment (Hungarian)
    ref_ind_filt, reg_ind_filt = linear_sum_assignment(cost_mat)

    # 3) map back to original ROI indices
    ref_ind = all_inds_ref_filt[ref_ind_filt]
    reg_ind = all_inds_reg_filt[reg_ind_filt]

    # 4) IOU computation
    thr_met = get_iou(
        all_roi_ref[:, :, ref_ind],
        all_roi_reg[:, :, reg_ind]
    )

    thr_met_compute = thr_met[thr_met > 0] if track_ops.thr_remove_zeros else thr_met

    # 5) threshold selection
    if track_ops.thr_method == "otsu":
        thr = threshold_otsu(thr_met_compute)
    elif track_ops.thr_method == "min":
        thr = threshold_minimum(thr_met_compute)
    else:
        raise ValueError(f"Unknown threshold method: {track_ops.thr_method}")

    return i, j, ref_ind, reg_ind, thr_met, thr


# ============================================================
# Parallel assignment across dataset pairs + planes
# ============================================================
def get_all_ds_assign(track_ops, all_ds_all_roi_ref, all_ds_all_roi_reg):

    n_ds = len(track_ops.all_ds_path) - 1
    nplanes = track_ops.nplanes

    # preallocate results
    all_ds_assign = [[None for _ in range(nplanes)] for _ in range(n_ds)]
    all_ds_assign_thr = [[None for _ in range(nplanes)] for _ in range(n_ds)]
    all_ds_thr_met = [[None for _ in range(nplanes)] for _ in range(n_ds)]
    all_ds_thr = [[None for _ in range(nplanes)] for _ in range(n_ds)]

    tasks = [(i, j) for i in range(n_ds) for j in range(nplanes)]

    print(f"Running {len(tasks)} (dataset, plane) tasks in parallel...")

    with ProcessPoolExecutor() as executor:

        futures = [
            executor.submit(
                _process_pair,
                i, j,
                all_ds_all_roi_ref,
                all_ds_all_roi_reg,
                track_ops
            )
            for i, j in tasks
        ]

        for fut in as_completed(futures):
            i, j, ref_ind, reg_ind, thr_met, thr = fut.result()

            mask = thr_met > thr

            all_ds_assign[i][j] = [ref_ind, reg_ind]
            all_ds_assign_thr[i][j] = [ref_ind[mask], reg_ind[mask]]
            all_ds_thr_met[i][j] = thr_met
            all_ds_thr[i][j] = thr

            print(f"Done dataset {i+1}/{n_ds}, plane {j+1}/{nplanes}")

    return all_ds_assign, all_ds_assign_thr, all_ds_thr_met, all_ds_thr


# ============================================================
# Propagate matches across days (unchanged logic)
# ============================================================
def get_all_pl_match_mat(all_ds_all_roi_ref, all_ds_assign_thr, track_ops):

    all_pl_match_mat = init_all_pl_match_mat(
        all_ds_all_roi_ref,
        all_ds_assign_thr,
        track_ops
    )

    for i in range(track_ops.nplanes):

        pl_match_mat = all_pl_match_mat[i]

        for roi_idx in range(pl_match_mat.shape[0]):

            if pl_match_mat[roi_idx, 0] is None:
                continue

            track_roi = np.array(pl_match_mat[roi_idx, 0])

            for ds_ind in range(pl_match_mat.shape[1] - 1):

                matches = all_ds_assign_thr[ds_ind][i]

                ref_ind = matches[0]
                reg_ind = matches[1]

                reg_ind_ind = np.where(ref_ind == track_roi.item())[0]

                if reg_ind_ind.size > 0:
                    track_roi = reg_ind[reg_ind_ind]
                    pl_match_mat[roi_idx, ds_ind + 1] = track_roi.item()
                else:
                    break

        n_tracked = np.sum(np.all(pl_match_mat != None, axis=1))
        print(f"Number of ROIs tracked in plane {i} across all days: {n_tracked}")

        track_ops.all_pl_match_mat = all_pl_match_mat
        track_ops.n_tracked = n_tracked

    return all_pl_match_mat