from track2p.register.elastix import reg_img_elastix, itk_reg_all_roi
from track2p.io.loaders import load_stat_ds_plane, get_all_roi_array_from_stat


def _process_reg_pair(i, j, all_ds_ref_img, all_ds_mov_img, all_ds_reg_params, track_ops):

    ref_img = all_ds_ref_img[i][j]
    mov_img = all_ds_mov_img[i][j]

    # ---------------------------
    # 1) Elastix registration
    # ---------------------------
    mov_img_reg, reg_params = reg_img_elastix(ref_img, mov_img, track_ops)

    # ---------------------------
    # 2) Load ROIs
    # ---------------------------
    ref_ds_path = track_ops.all_ds_path[i]
    mov_ds_path = track_ops.all_ds_path[i + 1]

    stat_ref, roi_counter_ref = load_stat_ds_plane(ref_ds_path, track_ops, plane_idx=j)
    stat_mov, roi_counter_mov = load_stat_ds_plane(mov_ds_path, track_ops, plane_idx=j)

    all_roi_array_ref = get_all_roi_array_from_stat(stat_ref, track_ops)
    all_roi_array_mov = get_all_roi_array_from_stat(stat_mov, track_ops)

    # ---------------------------
    # 3) Apply transform
    # ---------------------------
    all_roi_array_reg = itk_reg_all_roi(all_roi_array_mov, reg_params)

    return (
        i, j,
        mov_img_reg,
        reg_params,
        all_roi_array_ref,
        all_roi_array_mov,
        all_roi_array_reg,
        roi_counter_ref,
        roi_counter_mov
    )