import numpy as np
from pathlib import Path


def check_nplanes(track_ops):
    folder = "suite2p" if track_ops.input_format == "suite2p" else "data_npy"
    all_nplanes = []

    for ds_path in map(Path, track_ops.all_ds_path):
        n_planes = sum(1 for p in (ds_path / folder).iterdir() if p.name.startswith("plane"))
        print(f"Found {n_planes} planes in {ds_path}")
        all_nplanes.append(n_planes)

    track_ops.all_nplanes = all_nplanes

    if len(set(all_nplanes)) != 1:
        raise ValueError(
            f"Inconsistent plane counts across datasets: {all_nplanes}. "
            "Please check your dataset paths."
        )

    track_ops.nplanes = all_nplanes[0]
    print(f"Found {track_ops.nplanes} planes in all datasets")


def _load_ops(ds_path: Path, plane: int) -> dict:
    return np.load(ds_path / "suite2p" / f"plane{plane}" / "ops.npy", allow_pickle=True).item()


def load_all_imgs(track_ops):
    all_ds_avg_ch1, all_ds_avg_ch2, all_ds_nchannels = [], [], []

    for ds_path in map(Path, track_ops.all_ds_path):
        plane_ops = [_load_ops(ds_path, i) for i in range(track_ops.nplanes)]

        nchannels = [ops["nchannels"] for ops in plane_ops]
        avg_ch1   = [ops["meanImg"] for ops in plane_ops]
        avg_ch2   = [
            ops["meanImg_chan2"] if n == 2 else None
            for ops, n in zip(plane_ops, nchannels)
        ]

        for i, n in enumerate(nchannels):
            print(f"nchannels: {n} for plane {i} in dataset {ds_path}")

        all_ds_avg_ch1.append(avg_ch1)
        all_ds_avg_ch2.append(avg_ch2)
        all_ds_nchannels.append(nchannels)

    track_ops.all_ds_avg_ch1   = all_ds_avg_ch1
    track_ops.all_ds_avg_ch2   = all_ds_avg_ch2
    track_ops.all_ds_nchannels = all_ds_nchannels

    if len(set(map(tuple, all_ds_nchannels))) != 1:
        raise ValueError(
            "Inconsistent channel counts across datasets. Please check your dataset paths."
        )

    track_ops.nchannels = all_ds_nchannels[0][0]
    print(f"Found {track_ops.nchannels} channels in all datasets")

    return all_ds_avg_ch1, all_ds_avg_ch2


def load_all_ds_stat_iscell(track_ops):
    thr = track_ops.iscell_thr

    def filter_stat(stat, iscell):
        mask = iscell[:, 1] > thr if thr is not None else iscell[:, 0] == 1
        return stat[mask]

    return [
        [
            filter_stat(
                np.load(Path(ds_path) / "suite2p" / f"plane{j}" / "stat.npy",   allow_pickle=True),
                np.load(Path(ds_path) / "suite2p" / f"plane{j}" / "iscell.npy", allow_pickle=True),
            )
            for j in range(track_ops.nplanes)
        ]
        for ds_path in track_ops.all_ds_path
    ]


def load_all_ds_ops(track_ops):
    return [
        [_load_ops(Path(ds_path), j) for j in range(track_ops.nplanes)]
        for ds_path in track_ops.all_ds_path
    ]


def load_all_ds_mean_img(track_ops, ch=1):
    img_key = "meanImg" if ch == 1 else "meanImg_chan2"
    return [
        [ops[img_key] for ops in ds_ops]
        for ds_ops in load_all_ds_ops(track_ops)
    ]


def load_all_ds_centroids(all_ds_stat_iscell, track_ops):
    return [
        [
            np.array([roi["med"] for roi in stat_iscell])
            for stat_iscell in ds_stat_iscell
        ]
        for ds_stat_iscell in all_ds_stat_iscell
    ]