import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure
from skimage.filters import threshold_otsu

from ..logs import setup_logger

logger = setup_logger(__name__)


# compute centroids
def get_centroids(all_roi):

    logger.debug(f"Computing centroids | shape={all_roi.shape}")

    centroids = []

    for i in range(all_roi.shape[2]):
        roi = all_roi[:, :, i]

        labels = measure.label(roi)
        features = measure.regionprops(labels)

        try:
            centroids.append(features[0].centroid)

        except IndexError:
            logger.warning(f"Empty ROI detected at index={i}")
            centroids.append([0, 0])

    return np.array(centroids)


def get_cent_dist_mat(all_roi_ref, all_roi_reg):

    logger.debug("Computing centroid distance matrix")

    centroids_ref = get_centroids(all_roi_ref)
    centroids_reg = get_centroids(all_roi_reg)

    if len(centroids_ref) == 0 or len(centroids_reg) == 0:
        logger.error("Empty centroid set detected")
        raise ValueError("Empty centroid set")

    distances = cdist(centroids_ref, centroids_reg)

    logger.debug(f"Distance matrix shape={distances.shape}")

    return distances


def filt_non_overlap(all_roi1, all_roi2, cent_dist_mat):

    logger.debug("Filtering non-overlapping ROIs")

    all_inds = np.arange(all_roi1.shape[2])
    filt_inds_ref = []

    for i in range(all_roi1.shape[2]):

        roi = all_roi1[:, :, i]

        closest_roi_idx = np.argmin(cent_dist_mat[i, :])
        closest_roi = all_roi2[:, :, closest_roi_idx]

        intersection = roi * closest_roi

        if np.sum(intersection) == 0:
            filt_inds_ref.append(i)

    all_inds_filt = all_inds[~np.isin(all_inds, filt_inds_ref)]

    logger.debug(
        f"Filtered ROIs: kept={len(all_inds_filt)}/{len(all_inds)}"
    )

    return all_inds_filt


def get_cent_dist_mat_non_overlap(all_roi_ref, all_roi_reg):

    logger.debug("Computing non-overlap centroid distance matrix")

    cent_dist_mat = get_cent_dist_mat(all_roi_ref, all_roi_reg)

    all_inds_ref_filt = filt_non_overlap(all_roi_ref, all_roi_reg, cent_dist_mat)
    all_inds_reg_filt = filt_non_overlap(all_roi_reg, all_roi_ref, cent_dist_mat.T)

    cost_mat = cent_dist_mat[all_inds_ref_filt, :]
    cost_mat = cost_mat[:, all_inds_reg_filt]

    logger.debug(
        f"Filtered cost matrix shape={cost_mat.shape} | "
        f"ref={len(all_inds_ref_filt)}, reg={len(all_inds_reg_filt)}"
    )

    return cost_mat, all_inds_ref_filt, all_inds_reg_filt


def get_cost_mat(all_roi_ref, all_roi_reg, track_ops):

    logger.debug(f"Computing cost matrix | method={track_ops.matching_method}")

    # compute distances
    if track_ops.matching_method == 'cent':

        cost_mat = get_cent_dist_mat(all_roi_ref, all_roi_reg)

        all_inds_ref_filt = np.arange(all_roi_ref.shape[2])
        all_inds_reg_filt = np.arange(all_roi_reg.shape[2])

    elif track_ops.matching_method == 'cent_int-filt':

        cost_mat, all_inds_ref_filt, all_inds_reg_filt = (
            get_cent_dist_mat_non_overlap(all_roi_ref, all_roi_reg)
        )

    elif track_ops.matching_method == 'iou':

        cost_mat = 1 - get_cross_iou_mat(
            all_roi_ref,
            all_roi_reg,
            dist_thr=track_ops.iou_dist_thr
        )

        all_inds_ref_filt = np.arange(all_roi_ref.shape[2])
        all_inds_reg_filt = np.arange(all_roi_reg.shape[2])

    else:
        logger.error(f"Matching method not implemented: {track_ops.matching_method}")
        raise Exception('Matching method not implemented')

    logger.debug(
        f"cost_mat stats | shape={cost_mat.shape}, "
        f"min={np.min(cost_mat):.4f}, max={np.max(cost_mat):.4f}"
    )

    return cost_mat, all_inds_ref_filt, all_inds_reg_filt


def get_iou(all_roi_ref, all_roi_reg):

    logger.debug(f"Computing IoU | n_pairs={all_roi_ref.shape[2]}")

    ious = []

    for i in range(all_roi_ref.shape[2]):

        roi_ref = all_roi_ref[:, :, i]
        roi_reg = all_roi_reg[:, :, i]

        intersection = np.sum(np.logical_and(roi_ref, roi_reg))
        union = np.sum(np.logical_or(roi_ref, roi_reg))

        if union == 0:
            logger.warning(f"IoU union=0 at index={i}")
            ious.append(0.0)
        else:
            ious.append(intersection / union)

    return np.array(ious)


def get_cross_iou_mat(all_roi_ref, all_roi_reg, dist_thr=16):

    logger.debug(f"Computing cross IoU matrix | dist_thr={dist_thr}")

    distances = get_cent_dist_mat(all_roi_ref, all_roi_reg)

    cross_iou_mat = np.zeros(
        (all_roi_ref.shape[2], all_roi_reg.shape[2])
    )

    for i in range(all_roi_ref.shape[2]):
        for j in range(all_roi_reg.shape[2]):

            if distances[i, j] > dist_thr:
                continue

            intersection = np.logical_and(
                all_roi_ref[:, :, i],
                all_roi_reg[:, :, j]
            )

            union = np.logical_or(
                all_roi_ref[:, :, i],
                all_roi_reg[:, :, j]
            )

            union_sum = np.sum(union)

            if union_sum == 0:
                continue

            cross_iou_mat[i, j] = np.sum(intersection) / union_sum

    return cross_iou_mat


def init_all_pl_match_mat(all_ds_all_roi_ref, all_ds_assign_thr, track_ops):

    logger.info("Initializing plane-wise match matrices")

    all_pl_match_mat = []

    for i in range(track_ops.nplanes):

        pl_match_mat = np.full(
            (all_ds_all_roi_ref[0][i].shape[2], len(track_ops.all_ds_path)),
            None
        )

        all_pl_match_mat.append(pl_match_mat)

    # populate first row of the match matrices with the matches from the first ref-reg pair
    for i in range(track_ops.nplanes):

        pl_match_mat = all_pl_match_mat[i]
        assign_thr = all_ds_assign_thr[0][i]

        ref_ind = assign_thr[0]
        pl_match_mat[ref_ind, 0] = ref_ind

    logger.debug("Initialized match matrices successfully")

    return all_pl_match_mat


def filt_by_otsu(vect_filt, vect_comp):

    logger.debug("Applying Otsu filtering")

    thresh = threshold_otsu(vect_comp)

    logger.debug(f"Otsu threshold={thresh:.4f}")

    return vect_filt[vect_comp > thresh]