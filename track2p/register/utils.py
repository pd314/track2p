import numpy as np
from enum import IntEnum
from time import perf_counter

from ..logs import setup_logger

logger = setup_logger(__name__)


class Channel(IntEnum):
    FUNCTIONAL = 0
    ANATOMICAL = 1


def _log_img_info(tag, img):
    try:
        logger.debug(
            f"{tag} | shape={img.shape}, dtype={img.dtype}, "
            f"min={np.min(img):.4f}, max={np.max(img):.4f}"
        )
    except Exception:
        logger.debug(f"{tag} | shape={getattr(img, 'shape', None)}")


def get_all_ds_img_for_reg(all_ds_avg_ch1, all_ds_avg_ch2, track_ops):

    logger.info("Building dataset image pairs for registration")
    t0 = perf_counter()

    # channel selection
    if track_ops.reg_chan == Channel.FUNCTIONAL:
        all_ds_avg = all_ds_avg_ch1
        logger.debug("Using FUNCTIONAL channel")

    elif track_ops.reg_chan == Channel.ANATOMICAL:
        all_ds_avg = all_ds_avg_ch2
        logger.warning("Using ANATOMICAL channel (may be missing / lower quality)")

    else:
        logger.error(f"Invalid reg_chan: {track_ops.reg_chan}")
        raise ValueError(f"Invalid reg_chan: {track_ops.reg_chan}")

    # sanity check
    if len(all_ds_avg) != len(track_ops.all_ds_path):
        logger.warning(
            f"Mismatch: images={len(all_ds_avg)} vs paths={len(track_ops.all_ds_path)}"
        )

    all_ds_ref_img = []
    all_ds_mov_img = []

    n_pairs = len(track_ops.all_ds_path) - 1

    for i in range(n_pairs):

        logger.debug(f"[PAIR BUILD] {i+1}/{n_pairs}")

        ds_ref_img = []
        ds_mov_img = []

        for j in range(track_ops.nplanes):

            try:
                ref_img = all_ds_avg[i][j]
                mov_img = all_ds_avg[i + 1][j]

                _log_img_info(f"ref ds={i} plane={j}", ref_img)
                _log_img_info(f"mov ds={i+1} plane={j}", mov_img)

                ds_ref_img.append(ref_img)
                ds_mov_img.append(mov_img)

            except Exception:
                logger.error(
                    f"[PAIR BUILD FAIL] ds={i}, plane={j}",
                    exc_info=True
                )
                raise

        all_ds_ref_img.append(ds_ref_img)
        all_ds_mov_img.append(ds_mov_img)

    track_ops.all_ds_ref_img = all_ds_ref_img
    track_ops.all_ds_mov_img = all_ds_mov_img

    logger.info(
        f"Completed dataset pairing | pairs={n_pairs} | time={perf_counter() - t0:.2f}s"
    )

    return all_ds_ref_img, all_ds_mov_img


def get_ref_reg_inters(all_roi_array_ref, all_roi_array_nonref):

    logger.debug(
        f"[INTERSECT] ref shape={all_roi_array_ref.shape}, "
        f"nonref shape={all_roi_array_nonref.shape}"
    )

    try:
        # projections
        ref_proj = np.sum(all_roi_array_ref, axis=2) > 0
        nonref_proj = np.sum(all_roi_array_nonref, axis=2) > 0

        # density info (useful for debugging segmentation issues)
        ref_density = np.mean(ref_proj)
        nonref_density = np.mean(nonref_proj)

        logger.debug(
            f"[INTERSECT] density | ref={ref_density:.4f}, nonref={nonref_density:.4f}"
        )

        inters = np.logical_and(ref_proj, nonref_proj)

        inters_ratio = np.sum(inters) / inters.size
        logger.debug(f"[INTERSECT] overlap ratio={inters_ratio:.4f}")

        # RGB visualization
        ref_reg_inters = np.ones(
            (inters.shape[0], inters.shape[1], 3),
            dtype=np.float32
        )

        ref_reg_inters[:, :, 1] -= inters / 6.0
        ref_reg_inters[:, :, 2] -= inters

        return ref_reg_inters

    except Exception:
        logger.error("[INTERSECT FAIL]", exc_info=True)
        raise


def get_all_ref_nonref_inters(
    all_ds_all_roi_array_ref,
    all_ds_all_roi_array_nonref,
    track_ops
):

    logger.info("Computing all reference/non-reference ROI intersections")
    t0 = perf_counter()

    all_ds_all_ref_nonref_inters = []

    n_pairs = len(track_ops.all_ds_path) - 1

    try:
        for i in range(n_pairs):

            t_pair = perf_counter()

            logger.debug(f"[PAIR] {i+1}/{n_pairs}")

            ds_all_ref_nonref_inters = []

            for j in range(track_ops.nplanes):

                try:
                    ref = all_ds_all_roi_array_ref[i][j]
                    nonref = all_ds_all_roi_array_nonref[i][j]

                    logger.debug(
                        f"[PLANE] pair={i}, plane={j} | "
                        f"ref_n={ref.shape[2]}, nonref_n={nonref.shape[2]}"
                    )

                    inters = get_ref_reg_inters(ref, nonref)

                    ds_all_ref_nonref_inters.append(inters)

                except Exception:
                    logger.error(
                        f"[INTERSECTION FAIL] pair={i}, plane={j}",
                        exc_info=True
                    )
                    raise

            logger.debug(
                f"[PAIR DONE] {i} | time={perf_counter() - t_pair:.2f}s"
            )

            all_ds_all_ref_nonref_inters.append(ds_all_ref_nonref_inters)

        logger.info(
            f"Completed ROI intersection computation | total_time={perf_counter() - t0:.2f}s"
        )

        return all_ds_all_ref_nonref_inters

    except Exception:
        logger.error("ROI intersection pipeline failed", exc_info=True)
        raise