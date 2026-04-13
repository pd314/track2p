import numpy as np
from enum import IntEnum

from ..logs import setup_logger

logger = setup_logger(__name__)


class Channel(IntEnum):
    FUNCTIONAL = 0
    ANATOMICAL = 1


def get_all_ds_img_for_reg(all_ds_avg_ch1, all_ds_avg_ch2, track_ops):

    logger.info("Building dataset image pairs for registration")

    if track_ops.reg_chan == Channel.FUNCTIONAL:
        all_ds_avg = all_ds_avg_ch1
        logger.debug("Using FUNCTIONAL channel")

    elif track_ops.reg_chan == Channel.ANATOMICAL:
        all_ds_avg = all_ds_avg_ch2
        logger.warning("Using ANATOMICAL channel for registration (may be missing in some datasets)")

    else:
        logger.error(f"Unknown reg_chan: {track_ops.reg_chan}")
        raise ValueError(f"Invalid reg_chan: {track_ops.reg_chan}")

    all_ds_ref_img = []
    all_ds_mov_img = []

    n_pairs = len(track_ops.all_ds_path) - 1

    for i in range(n_pairs):
        logger.debug(f"Preparing image pair {i+1}/{n_pairs}")

        ds_ref_img = []
        ds_mov_img = []

        for j in range(track_ops.nplanes):

            try:
                ds_ref_img.append(all_ds_avg[i][j])
                ds_mov_img.append(all_ds_avg[i + 1][j])

            except Exception as e:
                logger.error(
                    f"Failed building image pair | dataset={i}, plane={j}",
                    exc_info=True
                )
                raise

        all_ds_ref_img.append(ds_ref_img)
        all_ds_mov_img.append(ds_mov_img)

    track_ops.all_ds_ref_img = all_ds_ref_img
    track_ops.all_ds_mov_img = all_ds_mov_img

    logger.info("Completed dataset image pairing")

    return all_ds_ref_img, all_ds_mov_img


def get_ref_reg_inters(all_roi_array_ref, all_roi_array_nonref):

    logger.debug(
        f"Computing ROI intersection | ref shape={all_roi_array_ref.shape}, "
        f"nonref shape={all_roi_array_nonref.shape}"
    )

    try:
        # projection across ROI dimension
        ref_proj = np.sum(all_roi_array_ref, axis=2) > 0
        nonref_proj = np.sum(all_roi_array_nonref, axis=2) > 0

        inters = np.logical_and(ref_proj, nonref_proj)

        # RGB visualization (white base)
        ref_reg_inters = np.ones((inters.shape[0], inters.shape[1], 3), dtype=np.float32)

        # overlay intersection mask (orange tint)
        ref_reg_inters[:, :, 1] -= inters / 6.0
        ref_reg_inters[:, :, 2] -= inters

        return ref_reg_inters

    except Exception as e:
        logger.error("Failed computing ROI intersection", exc_info=True)
        raise


def get_all_ref_nonref_inters(
    all_ds_all_roi_array_ref,
    all_ds_all_roi_array_nonref,
    track_ops
):

    logger.info("Computing all reference/non-reference ROI intersections")

    all_ds_all_ref_nonref_inters = []

    n_pairs = len(track_ops.all_ds_path) - 1

    try:
        for i in range(n_pairs):

            logger.debug(f"Processing pair {i+1}/{n_pairs}")

            ds_all_ref_nonref_inters = []

            for j in range(track_ops.nplanes):

                try:
                    ref = all_ds_all_roi_array_ref[i][j]
                    nonref = all_ds_all_roi_array_nonref[i][j]

                    inters = get_ref_reg_inters(ref, nonref)

                    ds_all_ref_nonref_inters.append(inters)

                except Exception as e:
                    logger.error(
                        f"Intersection failed | pair={i}, plane={j}",
                        exc_info=True
                    )
                    raise

            all_ds_all_ref_nonref_inters.append(ds_all_ref_nonref_inters)

        logger.info("Completed ROI intersection computation")

        return all_ds_all_ref_nonref_inters

    except Exception:
        logger.error("ROI intersection pipeline failed", exc_info=True)
        raise