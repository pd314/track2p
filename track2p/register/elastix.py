import itk
import numpy as np

from ..logs import setup_logger
logger = setup_logger(__name__)


def reg_img_elastix(ref_img, mov_img, track_ops):
    logger.info("Starting elastix registration")

    try:
        logger.debug(f"Ref image shape={ref_img.shape}, dtype={ref_img.dtype}")
        logger.debug(f"Mov image shape={mov_img.shape}, dtype={mov_img.dtype}")
        logger.debug(f"Transform type={track_ops.transform_type}")

        # convert to itk images
        ref_img_itk = itk.GetImageFromArray(ref_img)
        mov_img_itk = itk.GetImageFromArray(mov_img)

        # parameter map
        parameter_object = itk.ParameterObject.New()
        parameter_map = parameter_object.GetDefaultParameterMap(track_ops.transform_type)
        parameter_object.AddParameterMap(parameter_map)

        logger.debug("Parameter map created successfully")

        # registration
        mov_img_reg_itk, reg_params = itk.elastix_registration_method(
            ref_img_itk,
            mov_img_itk,
            parameter_object=parameter_object
        )

        mov_img_reg = itk.GetArrayFromImage(mov_img_reg_itk)

        # force ROI-friendly interpolation
        reg_params.SetParameter("FinalBSplineInterpolationOrder", "0")

        logger.info("Elastix registration completed successfully")
        logger.debug(f"Output shape={mov_img_reg.shape}")

        return mov_img_reg, reg_params

    except Exception as e:
        logger.error("Elastix registration failed", exc_info=True)
        logger.error(f"Error details: {e}")
        raise


def itk_reg_roi(roi, reg_params):
    logger.debug(f"Transforming ROI | shape={roi.shape}, dtype={roi.dtype}")

    try:
        roi_uint8 = roi.astype(np.uint8)

        if roi_uint8.max() == 0:
            logger.warning("Empty ROI detected (all zeros)")

        roi_itk = itk.GetImageFromArray(roi_uint8)
        roi_itk_trans = itk.transformix_filter(roi_itk, reg_params)
        roi_trans = itk.GetArrayFromImage(roi_itk_trans)

        return roi_trans

    except Exception as e:
        logger.error("ROI transform failed", exc_info=True)
        raise


def itk_reg_all_roi(all_roi, reg_params):
    logger.info(f"Transforming ROI stack | shape={all_roi.shape}")

    if all_roi.ndim != 3:
        logger.error(f"Expected 3D ROI stack, got shape={all_roi.shape}")
        raise ValueError("all_roi must be a 3D array (H, W, N_ROIs)")

    all_roi_array_reg = np.zeros_like(all_roi)

    n_rois = all_roi.shape[2]

    for i in range(n_rois):
        logger.debug(f"Transforming ROI {i+1}/{n_rois}")

        roi_array = all_roi[:, :, i]

        try:
            all_roi_array_reg[:, :, i] = itk_reg_roi(roi_array, reg_params)
        except Exception as e:
            logger.error(f"Failed ROI index={i}", exc_info=True)
            raise

    logger.info("Completed ROI stack transformation")

    return all_roi_array_reg