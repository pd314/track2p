from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import logging
from time import perf_counter

import numpy as np
import scipy
import pandas as pd

from track2p.io.s2p_loaders import (
    load_all_imgs,
    check_nplanes,
    load_all_ds_stat_iscell,
    load_all_ds_mean_img,
    load_all_ds_centroids,
)
from track2p.io.savers import npy_to_s2p, save_track_ops, save_all_pl_match_mat
from track2p.register.loop import run_reg_loop, reg_all_ds_all_roi
from track2p.register.utils import (
    get_all_ds_img_for_reg,
    get_all_ref_nonref_inters,
)
from track2p.plot.progress import plot_all_planes
from track2p.plot.output import (
    plot_reg_img_output,
    plot_thr_met_hist,
    plot_n_matched_roi,
    plot_roi_reg_output,
    plot_roi_match_multiplane,
    plot_allroi_match_multiplane,
)
from track2p.match.loop import get_all_ds_assign, get_all_pl_match_mat
from .logs import setup_logger

logger = setup_logger(__name__)


def _get_plane_directory_path(dataset_path: Path, plane_index: int) -> Path:
    """Get the path to a specific plane directory within a dataset.
    
    Args:
        dataset_path: Path pointing to the suite2p folder
        plane_index: Index of the plane

    Returns:
        Path to the planeN directory
    """
    return dataset_path / f"plane{plane_index}"


def _get_valid_cell_mask(
    iscell_array: np.ndarray,
    probability_threshold: float | None,
) -> np.ndarray:
    """Generate a boolean mask for valid cells based on probability threshold.
    
    Args:
        iscell_array: Array where column 0 is binary classification, column 1 is probability
        probability_threshold: Confidence threshold for cells. If None, uses binary classification

    Returns:
        Boolean array indicating valid cells
    """
    if probability_threshold is not None:
        return iscell_array[:, 1] > probability_threshold
    return iscell_array[:, 0] == 1


def run_t2p(track_ops):
    """Execute the complete track2p pipeline for multi-plane 2-photon imaging analysis.
    
    This function orchestrates the complete workflow:
    1. Initialize logging and output directories
    2. Load imaging data from multiple datasets
    3. Register datasets across planes
    4. Match ROIs across datasets and planes
    5. Save results in multiple formats
    6. Generate visualization plots
    
    Args:
        track_ops: Configuration namespace containing all track2p parameters
    """
    start_time = perf_counter()

    # Initialize save paths and logging
    track_ops.init_save_paths()

    log_directory = Path(track_ops.save_path) / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = log_directory / f"track2p_{timestamp}.log"

    task_logger = setup_logger(
        name="track2p",
        log_file=log_file_path,
        level=logging.DEBUG,
        to_console=True,
        force_reinit=True,
    )

    task_logger.info(f"Logging initialized → {log_file_path}")

    # Load and validate data
    check_nplanes(track_ops)

    if track_ops.input_format == "npy":
        task_logger.info("Converting suite2p .npy outputs to internal format...")
        npy_to_s2p(track_ops)

    (
        all_datasets_avg_ch1,
        all_datasets_avg_ch2,
        all_datasets_avg_ch1_enhanced,
        all_datasets_avg_ch2_enhanced,
    ) = load_all_imgs(track_ops, return_es=True)

    # Visualize available planes for registration
    plot_all_planes(all_datasets_avg_ch1, track_ops)
    if track_ops.nchannels == 2:
        plot_all_planes(all_datasets_avg_ch2, track_ops, ch="anatomical")

    # Register datasets
    task_logger.info("Starting registration...")
    reference_images, moving_images = get_all_ds_img_for_reg(
        all_datasets_avg_ch1, all_datasets_avg_ch2, track_ops
    )
    registered_moving_images, registration_parameters = run_reg_loop(
        reference_images, moving_images, track_ops
    )
    plot_reg_img_output(track_ops)

    # Apply registration transforms to all ROIs
    task_logger.info("Applying computed transforms to all ROIs...")
    (
        all_datasets_reference_rois,
        all_datasets_moving_rois,
        all_datasets_registered_rois,
        roi_match_counter,
    ) = reg_all_ds_all_roi(registration_parameters, track_ops)

    # Generate intersection visualization plots
    registered_vs_reference_intersections = get_all_ref_nonref_inters(
        all_datasets_reference_rois, all_datasets_registered_rois, track_ops
    )
    moving_vs_reference_intersections = get_all_ref_nonref_inters(
        all_datasets_reference_rois, all_datasets_moving_rois, track_ops
    )
    track_ops.all_ds_ref_mov_inters = moving_vs_reference_intersections
    track_ops.all_ds_ref_reg_inters = registered_vs_reference_intersections

    if track_ops.show_roi_reg_output:
        plot_roi_reg_output(track_ops)

    # Compute optimal ROI assignments across datasets
    task_logger.info("Computing optimal assignments for all dataset pairs...")
    (
        all_datasets_assignments,
        all_datasets_assignments_thresholded,
        threshold_metrics_all_datasets,
        threshold_values_all_datasets,
    ) = get_all_ds_assign(
        track_ops, all_datasets_reference_rois, all_datasets_registered_rois
    )
    plot_thr_met_hist(threshold_metrics_all_datasets, threshold_values_all_datasets, track_ops)
    plot_n_matched_roi(threshold_metrics_all_datasets, threshold_values_all_datasets, track_ops)

    # Generate cross-plane match matrices
    all_planes_match_matrix = get_all_pl_match_mat(
        all_datasets_reference_rois, all_datasets_assignments_thresholded, track_ops
    )

    # Save tracking results
    task_logger.info("Saving results...")
    save_track_ops(track_ops)
    save_all_pl_match_mat(all_planes_match_matrix, track_ops)

    task_logger.info("Generating suite2p indices")
    generate_suite2p_indices(track_ops)

    # Export in suite2p format
    if track_ops.save_in_s2p_format:
        task_logger.info("Saving in suite2p format...")
        save_in_s2p_format(track_ops)

    # Generate output visualizations
    task_logger.info(
        "Finished with algorithm! Generating plots (this can take some time)..."
    )
    all_datasets_cell_classification = load_all_ds_stat_iscell(track_ops)
    all_datasets_roi_centroids = load_all_ds_centroids(
        all_datasets_cell_classification, track_ops
    )
    all_datasets_mean_image = load_all_ds_mean_img(track_ops)

    plot_roi_match_multiplane(
        all_datasets_mean_image,
        all_datasets_roi_centroids,
        all_planes_match_matrix,
        track_ops,
        win_size=track_ops.win_size,
    )
    plot_allroi_match_multiplane(all_datasets_mean_image, all_planes_match_matrix, track_ops)

    if track_ops.nchannels == 2:
        all_datasets_mean_image_ch2 = load_all_ds_mean_img(track_ops, ch=2)
        plot_roi_match_multiplane(
            all_datasets_mean_image_ch2,
            all_datasets_roi_centroids,
            all_planes_match_matrix,
            track_ops,
            win_size=track_ops.win_size,
            ch=2,
        )
        plot_allroi_match_multiplane(
            all_datasets_mean_image_ch2, all_planes_match_matrix, track_ops, ch=2
        )

    end_time = perf_counter()
    task_logger.info(f"All done! Total time: {end_time - start_time:.2f} seconds)")


def generate_suite2p_indices(track_ops):
    """Convert matched ROI indices to suite2p format for each plane.
    
    Takes the match matrices and converts them to suite2p-compatible indices,
    accounting for the cell probability threshold. Saves in multiple formats:
    .npy, .mat, and .csv.
    
    Args:
        track_ops: Configuration namespace containing paths and parameters
    """
    save_directory = Path(track_ops.save_path)
    cell_probability_threshold = track_ops.iscell_thr

    for plane_idx in range(track_ops.nplanes):
        # Load the match matrix for this plane
        plane_match_matrix = np.load(
            save_directory / f"plane{plane_idx}_match_mat.npy", allow_pickle=True
        )

        # Load iscell arrays for all datasets
        all_iscell_arrays = [
            np.load(
                _get_plane_directory_path(dataset_path, plane_idx) / "iscell.npy",
                allow_pickle=True,
            )
            for dataset_path in track_ops.all_ds_path
        ]

        # Convert match matrix indices to true cell indices
        mapped_indices_list = []
        for match_line in plane_match_matrix:
            converted_line_indices = []
            for dataset_idx, matched_index in enumerate(match_line):
                if matched_index is None:
                    converted_line_indices.append(None)
                else:
                    # Get valid cell indices for this dataset
                    valid_cell_indices = np.where(
                        _get_valid_cell_mask(all_iscell_arrays[dataset_idx], cell_probability_threshold)
                    )[0]
                    # Convert relative match index to absolute cell index
                    converted_line_indices.append(valid_cell_indices[matched_index])
            mapped_indices_list.append(converted_line_indices)

        # Convert to numpy arrays with appropriate dtypes
        mapped_indices = np.array(
            [[int(idx) if idx is not None else None for idx in row] for row in mapped_indices_list]
        )
        mapped_indices_with_nan = np.array(
            [
                [float(idx) if idx is not None else np.nan for idx in row]
                for row in mapped_indices_list
            ]
        )

        # Save in multiple formats
        np.save(save_directory / f"plane{plane_idx}_suite2p_indices.npy", mapped_indices)
        np.save(
            save_directory / f"plane{plane_idx}_suite2p_indices_nan.npy",
            mapped_indices_with_nan,
        )
        scipy.io.savemat(
            str(save_directory / f"plane{plane_idx}_suite2p_indices.mat"),
            {"data": mapped_indices_with_nan},
        )

        # Save as CSV with dataset names as column headers
        dataset_names = [Path(ds_path).name for ds_path in track_ops.all_ds_path]
        pd.DataFrame(mapped_indices, columns=dataset_names).to_csv(
            save_directory / f"plane{plane_idx}_suite2p_indices.csv",
            index=False,
            sep=";",
            na_rep="NaN",
        )
        
        logger.info(
            f"Saved suite2p indices for plane {plane_idx} in .npy, .mat, and .csv formats"
        )
        logger.debug(
            f"Data types - mapped_indices: {mapped_indices.dtype}, "
            f"mapped_indices_with_nan: {mapped_indices_with_nan.dtype}"
        )


def save_in_s2p_format(track_ops):
    """Save matched ROI data in suite2p directory structure and format.
    
    Applies the matching results to all datasets and exports them in the
    official suite2p format for downstream analysis compatibility.
    
    Args:
        track_ops: Configuration namespace containing paths and parameters
    """
    save_directory = Path(track_ops.save_path)
    
    # Reload track_ops from saved file to ensure consistency
    track_ops = SimpleNamespace(
        **np.load(
            save_directory / "track_ops.npy", allow_pickle=True
        ).item()
    )
    
    cell_probability_threshold = track_ops.iscell_thr
    is_two_channel = track_ops.nchannels == 2

    for plane_idx in range(track_ops.nplanes):
        logger.info(f"Processing plane {plane_idx}")

        # Load match matrix and extract successfully matched rows
        plane_match_matrix = np.load(
            save_directory / f"plane{plane_idx}_match_mat.npy", allow_pickle=True
        )
        fully_matched_rows = plane_match_matrix[
            ~np.any(plane_match_matrix == None, axis=1)  # noqa: E711
        ]
        matched_indices = fully_matched_rows.astype(int)

        # Process data for each dataset
        per_dataset_matched_data = []
        for dataset_idx, dataset_path in enumerate(track_ops.all_ds_path):
            plane_directory = _get_plane_directory_path(dataset_path, plane_idx)

            # Helper function to load .npy files from plane directory
            def load_npy_file(filename: str) -> np.ndarray:
                """Load a .npy file from the current plane directory."""
                return np.load(plane_directory / filename, allow_pickle=True)

            # Load all required arrays
            ops_dict = load_npy_file("ops.npy").item()
            cell_stat = load_npy_file("stat.npy")
            fluorescence_signal = load_npy_file("F.npy")
            neuropil_fluorescence = load_npy_file("Fneu.npy")
            spike_rates = load_npy_file("spks.npy")
            iscell_array = load_npy_file("iscell.npy")

            # Generate mask for valid cells
            valid_cell_mask = _get_valid_cell_mask(iscell_array, cell_probability_threshold)

            # Create dictionary of masked arrays
            masked_arrays = {
                "stat": cell_stat[valid_cell_mask],
                "f": fluorescence_signal[valid_cell_mask],
                "fneu": neuropil_fluorescence[valid_cell_mask],
                "spks": spike_rates[valid_cell_mask],
                "iscell": iscell_array[valid_cell_mask],
                "ops": ops_dict,
            }

            # Add channel 2 data if available
            if is_two_channel:
                masked_arrays.update(
                    {
                        "f_chan2": load_npy_file("F_chan2.npy")[valid_cell_mask],
                        "fneu_chan2": load_npy_file("Fneu_chan2.npy")[valid_cell_mask],
                        "redcell": load_npy_file("redcell.npy")[valid_cell_mask],
                    }
                )

            # Apply matched indices to select only tracked ROIs
            dataset_matched_indices = matched_indices[:, dataset_idx]
            tracked_roi_data = {
                array_name: (
                    array_data[dataset_matched_indices]
                    if array_name != "ops"
                    else array_data
                )
                for array_name, array_data in masked_arrays.items()
            }
            per_dataset_matched_data.append(tracked_roi_data)

        # Save matched data in suite2p structure
        output_root = save_directory / "matched_suite2p"
        for dataset_idx, dataset_path in enumerate(track_ops.all_ds_path):
            plane_output_directory = (
                output_root / Path(dataset_path).name / "suite2p" / f"plane{plane_idx}"
            )
            plane_output_directory.mkdir(parents=True, exist_ok=True)

            dataset_tracked_data = per_dataset_matched_data[dataset_idx]
            
            # Save core arrays
            np.save(
                plane_output_directory / "stat.npy", dataset_tracked_data["stat"]
            )
            np.save(
                plane_output_directory / "F.npy", dataset_tracked_data["f"]
            )
            np.save(
                plane_output_directory / "ops.npy", dataset_tracked_data["ops"]
            )
            np.save(
                plane_output_directory / "iscell.npy", dataset_tracked_data["iscell"]
            )
            np.save(
                plane_output_directory / "Fneu.npy", dataset_tracked_data["fneu"]
            )
            np.save(
                plane_output_directory / "spks.npy", dataset_tracked_data["spks"]
            )

            # Save channel 2 data if available
            if is_two_channel:
                np.save(
                    plane_output_directory / "F_chan2.npy",
                    dataset_tracked_data["f_chan2"],
                )
                np.save(
                    plane_output_directory / "Fneu_chan2.npy",
                    dataset_tracked_data["fneu_chan2"],
                )
                np.save(
                    plane_output_directory / "redcell.npy",
                    dataset_tracked_data["redcell"],
                )
