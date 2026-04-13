from time import perf_counter

from track2p.register.elastix import reg_img_elastix, itk_reg_all_roi
from track2p.io.loaders import load_stat_ds_plane, get_all_roi_array_from_stat

from ..logs import get_logger

logger = get_logger(__name__)


def run_reg_loop(all_ds_ref_img, all_ds_mov_img, track_ops):

    logger.info("Starting registration loop")
    t_global = perf_counter()

    all_ds_mov_img_reg = []
    all_ds_reg_params = []

    try:
        for i, ds_ref_img in enumerate(all_ds_ref_img):

            t_ds = perf_counter()

            logger.info(
                f"Dataset {i+1}/{len(all_ds_ref_img)} | "
                f"n_planes={track_ops.nplanes}"
            )

            ds_mov_img = all_ds_mov_img[i]
            ds_mov_img_reg = []
            ds_reg_params = []

            for j in range(track_ops.nplanes):

                t_plane = perf_counter()

                try:
                    ref_img = ds_ref_img[j]
                    mov_img = ds_mov_img[j]

                    logger.debug(
                        f"[REG] ds={i}, plane={j} | "
                        f"ref_shape={getattr(ref_img, 'shape', None)}, "
                        f"mov_shape={getattr(mov_img, 'shape', None)}"
                    )

                    mov_img_reg, reg_params = reg_img_elastix(
                        ref_img,
                        mov_img,
                        track_ops
                    )

                    ds_mov_img_reg.append(mov_img_reg)
                    ds_reg_params.append(reg_params)

                    logger.debug(
                        f"[REG OK] ds={i}, plane={j} | "
                        f"time={perf_counter() - t_plane:.2f}s"
                    )

                except Exception:
                    logger.error(
                        f"[REG FAIL] ds={i}, plane={j}",
                        exc_info=True
                    )
                    raise

            logger.info(
                f"Dataset {i} done | time={perf_counter() - t_ds:.2f}s"
            )

            all_ds_mov_img_reg.append(ds_mov_img_reg)
            all_ds_reg_params.append(ds_reg_params)

        track_ops.all_ds_mov_img_reg = all_ds_mov_img_reg

        logger.info(
            f"Registration loop completed | total_time={perf_counter() - t_global:.2f}s"
        )

        return all_ds_mov_img_reg, all_ds_reg_params

    except Exception:
        logger.error("Registration loop aborted", exc_info=True)
        raise


def reg_all_ds_all_roi(all_ds_reg_params, track_ops):

    logger.info("Starting ROI registration pipeline")

    t_global = perf_counter()

    all_ds_all_roi_array_ref = []
    all_ds_all_roi_array_mov = []
    all_ds_all_roi_array_reg = []
    all_ds_roi_counter = []

    n_pairs = len(track_ops.all_ds_path) - 1

    try:
        for i in range(n_pairs):

            t_ds = perf_counter()

            ref_ds_path = track_ops.all_ds_path[i]
            reg_ds_path = track_ops.all_ds_path[i + 1]

            logger.info(
                f"[ROI] Pair {i+1}/{n_pairs} | "
                f"ref={ref_ds_path} → mov={reg_ds_path}"
            )

            ds_all_roi_array_ref = []
            ds_all_roi_array_mov = []
            ds_all_roi_array_reg = []

            ds_roi_counter_ref = []
            ds_roi_counter_mov = []

            for j in range(track_ops.nplanes):

                t_plane = perf_counter()

                try:
                    reg_params = all_ds_reg_params[i][j]

                    logger.debug(f"[LOAD] ds_pair={i}, plane={j}")

                    stat_ref, roi_counter_ref = load_stat_ds_plane(
                        ref_ds_path,
                        track_ops,
                        plane_idx=j
                    )

                    stat_mov, roi_counter_mov = load_stat_ds_plane(
                        reg_ds_path,
                        track_ops,
                        plane_idx=j
                    )

                    all_roi_array_ref = get_all_roi_array_from_stat(stat_ref, track_ops)
                    all_roi_array_mov = get_all_roi_array_from_stat(stat_mov, track_ops)

                    logger.debug(
                        f"[ROI] ds={i}, plane={j} | "
                        f"ref_n={all_roi_array_ref.shape[2]}, "
                        f"mov_n={all_roi_array_mov.shape[2]}"
                    )

                    logger.debug(f"[TRANSFORM] ds={i}, plane={j}")

                    t_roi = perf_counter()

                    all_roi_array_reg = itk_reg_all_roi(
                        all_roi_array_mov,
                        reg_params
                    )

                    logger.debug(
                        f"[TRANSFORM OK] ds={i}, plane={j} | "
                        f"time={perf_counter() - t_roi:.2f}s"
                    )

                    ds_all_roi_array_ref.append(all_roi_array_ref)
                    ds_all_roi_array_mov.append(all_roi_array_mov)
                    ds_all_roi_array_reg.append(all_roi_array_reg)

                    ds_roi_counter_ref.append(roi_counter_ref)
                    ds_roi_counter_mov.append(roi_counter_mov)

                    logger.debug(
                        f"[PLANE DONE] ds={i}, plane={j} | "
                        f"time={perf_counter() - t_plane:.2f}s"
                    )

                except Exception:
                    logger.error(
                        f"[ROI FAIL] ds_pair={i}, plane={j}",
                        exc_info=True
                    )
                    raise

            logger.info(
                f"[PAIR DONE] {i} | time={perf_counter() - t_ds:.2f}s"
            )

            all_ds_all_roi_array_ref.append(ds_all_roi_array_ref)
            all_ds_all_roi_array_mov.append(ds_all_roi_array_mov)
            all_ds_all_roi_array_reg.append(ds_all_roi_array_reg)

            all_ds_roi_counter.append(ds_roi_counter_ref)

            if i == n_pairs - 1:
                all_ds_roi_counter.append(ds_roi_counter_mov)

        track_ops.all_ds_all_roi_array_ref = all_ds_all_roi_array_ref
        track_ops.all_ds_all_roi_array_mov = all_ds_all_roi_array_mov
        track_ops.all_ds_all_roi_array_reg = all_ds_all_roi_array_reg
        track_ops.all_ds_roi_counter = all_ds_roi_counter

        logger.info(
            f"ROI pipeline completed | total_time={perf_counter() - t_global:.2f}s"
        )

        return (
            all_ds_all_roi_array_ref,
            all_ds_all_roi_array_mov,
            all_ds_all_roi_array_reg,
            all_ds_roi_counter
        )

    except Exception:
        logger.error("ROI pipeline aborted", exc_info=True)
        raise