from time import perf_counter

from track2p.register.elastix import reg_img_elastix, itk_reg_all_roi
from track2p.io.loaders import load_stat_ds_plane, get_all_roi_array_from_stat

from ..logs import setup_logger

logger = setup_logger(__name__)


def run_reg_loop(all_ds_ref_img, all_ds_mov_img, track_ops):

    logger.info("Starting registration loop")
    start = perf_counter()

    all_ds_mov_img_reg = []
    all_ds_reg_params = []

    try:
        for i, ds_ref_img in enumerate(all_ds_ref_img):
            logger.info(f"Processing dataset {i+1}/{len(all_ds_ref_img)}")

            ds_mov_img = all_ds_mov_img[i]
            ds_mov_img_reg = []
            ds_reg_params = []

            for j in range(track_ops.nplanes):
                logger.debug(f"Dataset {i}, plane {j}: registration start")

                try:
                    ref_img = ds_ref_img[j]
                    mov_img = ds_mov_img[j]

                    mov_img_reg, reg_params = reg_img_elastix(
                        ref_img,
                        mov_img,
                        track_ops
                    )

                    ds_mov_img_reg.append(mov_img_reg)
                    ds_reg_params.append(reg_params)

                    logger.debug(f"Dataset {i}, plane {j}: registration OK")

                except Exception as e:
                    logger.error(
                        f"Registration failed | dataset={i}, plane={j}",
                        exc_info=True
                    )
                    raise

            all_ds_mov_img_reg.append(ds_mov_img_reg)
            all_ds_reg_params.append(ds_reg_params)

        track_ops.all_ds_mov_img_reg = all_ds_mov_img_reg

        end = perf_counter()
        logger.info(f"Registration loop completed in {end - start:.2f} seconds")

        return all_ds_mov_img_reg, all_ds_reg_params

    except Exception:
        logger.error("Registration loop aborted due to error", exc_info=True)
        raise


def reg_all_ds_all_roi(all_ds_reg_params, track_ops):

    logger.info("Starting ROI registration pipeline")

    all_ds_all_roi_array_ref = []
    all_ds_all_roi_array_mov = []
    all_ds_all_roi_array_reg = []
    all_ds_roi_counter = []

    n_pairs = len(track_ops.all_ds_path) - 1

    try:
        for i in range(n_pairs):

            logger.info(f"Processing dataset pair {i+1}/{n_pairs}")

            ds_all_roi_array_ref = []
            ds_all_roi_array_mov = []
            ds_all_roi_array_reg = []

            ds_roi_counter_ref = []
            ds_roi_counter_mov = []

            ref_ds_path = track_ops.all_ds_path[i]
            reg_ds_path = track_ops.all_ds_path[i + 1]

            for j in range(track_ops.nplanes):

                logger.debug(f"Pair {i}, plane {j}: loading ROIs")

                try:
                    reg_params = all_ds_reg_params[i][j]

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

                    logger.debug(f"Pair {i}, plane {j}: applying transform")

                    all_roi_array_reg = itk_reg_all_roi(
                        all_roi_array_mov,
                        reg_params
                    )

                    ds_all_roi_array_ref.append(all_roi_array_ref)
                    ds_all_roi_array_mov.append(all_roi_array_mov)
                    ds_all_roi_array_reg.append(all_roi_array_reg)

                    ds_roi_counter_ref.append(roi_counter_ref)
                    ds_roi_counter_mov.append(roi_counter_mov)

                    logger.debug(f"Pair {i}, plane {j}: ROI transform OK")

                except Exception as e:
                    logger.error(
                        f"ROI pipeline failed | pair={i}, plane={j}",
                        exc_info=True
                    )
                    raise

            logger.info(f"Completed dataset pair {i}")

            all_ds_all_roi_array_ref.append(ds_all_roi_array_ref)
            all_ds_all_roi_array_mov.append(ds_all_roi_array_mov)
            all_ds_all_roi_array_reg.append(ds_all_roi_array_reg)

            all_ds_roi_counter.append(ds_roi_counter_ref)

            # last dataset has no "next ref"
            if i == n_pairs - 1:
                all_ds_roi_counter.append(ds_roi_counter_mov)

        track_ops.all_ds_all_roi_array_ref = all_ds_all_roi_array_ref
        track_ops.all_ds_all_roi_array_mov = all_ds_all_roi_array_mov
        track_ops.all_ds_all_roi_array_reg = all_ds_all_roi_array_reg
        track_ops.all_ds_roi_counter = all_ds_roi_counter

        logger.info("ROI registration pipeline completed successfully")

        return (
            all_ds_all_roi_array_ref,
            all_ds_all_roi_array_mov,
            all_ds_all_roi_array_reg,
            all_ds_roi_counter
        )

    except Exception:
        logger.error("ROI registration pipeline aborted", exc_info=True)
        raise