import itk
import numpy as np
from time import perf_counter

from ..logs import setup_logger
logger = setup_logger(__name__)


def _log_img_stats(name, img):
    """Helper: avoids repeating debug boilerplate"""
    try:
        logger.debug(
            f"{name} | shape={img.shape}, dtype={img.dtype}, "
            f"min={np.min(img):.4f}, max={np.max(img):.4f}, mean={np.mean(img):.4f}"
        )
    except Exception:
        logger.debug(f"{name} | shape={getattr(img, 'shape', None)} (stats unavailable)")


def reg_img_elastix(ref_img, mov_img, track_ops):

    t0 = perf_counter()
    logger.info("Starting elastix registration")

    try:
        _log_img_stats("Ref image", ref_img)
        _log_img_stats("Mov image", mov_img)

        logger.debug(f"Transform type = {track_ops.transform_type}")

        # convert to itk images
        logger.debug("Converting numpy → ITK")
        ref_img_itk = itk.GetImageFromArray(ref_img)
        mov_img_itk = itk.GetImageFromArray(mov_img)

        # parameter map
        logger.debug("Building elastix parameter object")
        parameter_object = itk.ParameterObject.New()

        parameter_map = parameter_object.GetDefaultParameterMap(track_ops.transform_type)

        logger.debug(
            f"Parameter map created | type={track_ops.transform_type}, "
            f"keys={len(parameter_map)}"
        )

        parameter_object.AddParameterMap(parameter_map)

        # registration
        logger.info("Running elastix registration...")
        t_reg0 = perf_counter()

        mov_img_reg_itk, reg_params = itk.elastix_registration_method(
            ref_img_itk,
            mov_img_itk,
            parameter_object=parameter_object
        )

        t_reg1 = perf_counter()

        mov_img_reg = itk.GetArrayFromImage(mov_img_reg_itk)

        logger.info(f"Elastix finished in {t_reg1 - t_reg0:.2f}s")

        # interpolation fix
        logger.debug("Setting ROI-friendly interpolation order = 0")
        reg_params.SetParameter("FinalBSplineInterpolationOrder", "0")

        _log_img_stats("Registered image", mov_img_reg)

        logger.info(f"Registration total time: {perf_counter() - t0:.2f}s")

        return mov_img_reg, reg_params

    except Exception as e:
        logger.error(
            "Elastix registration failed",
            exc_info=True
        )
        logger.error(f"Transform type: {track_ops.transform_type}")
        logger.error(f"Ref shape: {getattr(ref_img, 'shape', None)}")
        logger.error(f"Mov shape: {getattr(mov_img, 'shape', None)}")
        raise


def itk_reg_roi(roi, reg_params):

    try:
        _log_img_stats("ROI input", roi)

        roi_uint8 = roi.astype(np.uint8)

        if roi_uint8.max() == 0:
            logger.warning("Empty ROI detected (all zeros)")

        roi_itk = itk.GetImageFromArray(roi_uint8)

        roi_itk_trans = itk.transformix_filter(roi_itk, reg_params)

        roi_trans = itk.GetArrayFromImage(roi_itk_trans)

        _log_img_stats("ROI output", roi_trans)

        return roi_trans

    except Exception:
        logger.error("ROI transform failed", exc_info=True)
        raise


def itk_reg_all_roi(all_roi, reg_params):

    logger.info(f"Transforming ROI stack | shape={all_roi.shape}")

    if all_roi.ndim != 3:
        logger.error(f"Invalid ROI stack shape: {all_roi.shape}")
        raise ValueError("Expected 3D ROI array (H, W, N_ROIs)")

    all_roi_array_reg = np.zeros_like(all_roi)

    n_rois = all_roi.shape[2]

    t0 = perf_counter()

    for i in range(n_rois):

        logger.debug(f"ROI {i+1}/{n_rois}")

        try:
            all_roi_array_reg[:, :, i] = itk_reg_roi(
                all_roi[:, :, i],
                reg_params
            )

        except Exception:
            logger.error(f"Failed ROI index={i}", exc_info=True)
            raise

    logger.info(
        f"ROI stack done | n={n_rois} | time={perf_counter() - t0:.2f}s"
    )

    return all_roi_array_reg