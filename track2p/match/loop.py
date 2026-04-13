import numpy as np
from skimage.filters import threshold_otsu, threshold_minimum
from scipy.optimize import linear_sum_assignment

from track2p.match.utils import get_cost_mat, get_iou, init_all_pl_match_mat

from ..logs import setup_logger

logger = setup_logger(__name__)


# assigment of ROIs in each ref-reg pair

def get_all_ds_assign(track_ops, all_ds_all_roi_ref, all_ds_all_roi_reg):

    all_ds_assign = []
    all_ds_assign_thr = []
    all_ds_thr_met = []
    all_ds_thr = []

    n_pairs = len(track_ops.all_ds_path) - 1

    for i in range(n_pairs):
        logger.info(f'Finding matches in ref-reg pair: {i+1}/{n_pairs}')

        ds_assign = []
        ds_assign_thr = []
        ds_thr_met = []
        ds_thr = []

        for j in range(track_ops.nplanes):

            try:
                all_roi_ref = all_ds_all_roi_ref[i][j]
                all_roi_reg = all_ds_all_roi_reg[i][j]

                logger.debug(
                    f"Pair {i}, plane {j} | ref shape={all_roi_ref.shape}, reg shape={all_roi_reg.shape}"
                )

                # 1) compute cost matrix (currently two methods available, see DefaultTrackOps)
                cost_mat, all_inds_ref_filt, all_inds_reg_filt = get_cost_mat(
                    all_roi_ref,
                    all_roi_reg,
                    track_ops
                )

                logger.debug(
                    f"Pair {i}, plane {j} | cost matrix shape={cost_mat.shape}"
                )

                # 2) optimally assign pairs
                ref_ind_filt, reg_ind_filt = linear_sum_assignment(cost_mat)

                # 3) convert them to pre-filtered indices (these are the indices of the ROIs after iscell)
                ref_ind = all_inds_ref_filt[ref_ind_filt]
                reg_ind = all_inds_reg_filt[reg_ind_filt]

                # 4) for each matched pair (len(all_roi_ref)) compute thresholding metric (in this case IOU, the filtering will be done afterwards in the all-day assignment)
                thr_met = get_iou(
                    all_roi_ref[:, :, ref_ind],
                    all_roi_reg[:, :, reg_ind]
                )

                thr_met_compute = (
                    thr_met[thr_met > 0]
                    if track_ops.thr_remove_zeros
                    else thr_met
                )

                # guard: empty similarity values
                if len(thr_met_compute) == 0:
                    logger.warning(f"Pair {i}, plane {j} | empty IoU array after filtering")
                    thr = 0.0
                else:
                    # 5) compute otsu threshold on thr_met
                    if track_ops.thr_method == 'otsu':
                        thr = threshold_otsu(thr_met_compute)
                    elif track_ops.thr_method == 'min':
                        thr = threshold_minimum(thr_met_compute)
                    else:
                        logger.error(f"Unknown threshold method: {track_ops.thr_method}")
                        raise ValueError(track_ops.thr_method)

                ds_assign.append([ref_ind, reg_ind])

                ds_assign_thr.append([
                    ref_ind[thr_met > thr],
                    reg_ind[thr_met > thr]
                ])

                ds_thr_met.append(thr_met)
                ds_thr.append(thr)

                logger.debug(
                    f"Pair {i}, plane {j} | matches={len(ref_ind)}, thr={thr:.4f}"
                )

            except Exception as e:
                logger.error(
                    f"Assignment failed | pair={i}, plane={j}",
                    exc_info=True
                )
                raise

        all_ds_assign.append(ds_assign)
        all_ds_assign_thr.append(ds_assign_thr)
        all_ds_thr_met.append(ds_thr_met)
        all_ds_thr.append(ds_thr)

        logger.info(f'Done ref-reg pair: {i+1}/{n_pairs}')

    return all_ds_assign, all_ds_assign_thr, all_ds_thr_met, all_ds_thr


# propagating matches across all days

def get_all_pl_match_mat(all_ds_all_roi_ref, all_ds_assign_thr, track_ops):

    all_pl_match_mat = init_all_pl_match_mat(
        all_ds_all_roi_ref,
        all_ds_assign_thr,
        track_ops
    )

    logger.info("Starting cross-day ROI tracking propagation")

    for i in range(track_ops.nplanes):

        logger.info(f"Processing plane {i}")

        pl_match_mat = all_pl_match_mat[i]

        # now for each row in the match matrix (each ROI in the ref recording) we need to find the match across all days, if there is none then we leave it as None

        for roi_idx in range(pl_match_mat.shape[0]):  # roi_idx is the index on first session

            try:
                # if first column is none then we skip this row
                if pl_match_mat[roi_idx, 0] is None:
                    continue

                # otherwise we find the match in the all_ds_assign_thr
                ref_roi_ds0 = pl_match_mat[roi_idx, 0]
                track_roi = np.array(ref_roi_ds0)

                for ds_ind in range(pl_match_mat.shape[1] - 1):

                    matches = all_ds_assign_thr[ds_ind][i]

                    ref_ind = matches[0]
                    reg_ind = matches[1]

                    reg_ind_ind = np.where(ref_ind == track_roi.item())[0]

                    # if there is a match then we update the track_roi
                    if reg_ind_ind.size > 0:
                        track_roi = reg_ind[reg_ind_ind]
                        pl_match_mat[roi_idx, ds_ind + 1] = track_roi.item()

                    # if there is no match then we stop tracking this ROI
                    else:
                        break

            except Exception as e:
                logger.error(
                    f"Tracking failed | plane={i}, roi_idx={roi_idx}",
                    exc_info=True
                )
                raise

        # compute how many ROIs are tracked across all days
        n_tracked = np.sum(np.all(pl_match_mat != None, axis=1))

        logger.info(f'Number of ROIs tracked in plane{i} across all days: {n_tracked}')

        track_ops.all_pl_match_mat = all_pl_match_mat
        track_ops.n_tracked = n_tracked

    return all_pl_match_mat