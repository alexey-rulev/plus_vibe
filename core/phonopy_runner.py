import os
import re
import tempfile
from typing import List, Sequence, Tuple

import numpy as np
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

DEFAULT_QPATH_TEXT = """G 0 0 0
X 0.5 0 0
M 0.5 0.5 0
G 0 0 0"""


def _write_temp_file(upload, suffix: str) -> str:
    data = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    if not data:
        raise ValueError("Uploaded file is empty")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def _parse_special_points(text: str) -> Tuple[List[str], List[np.ndarray]]:
    labels: List[str] = []
    points: List[np.ndarray] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 4:
            raise ValueError(f"Invalid q-point line '{raw_line}'. Expected: LABEL qx qy qz")
        label = parts[0]
        try:
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Failed to parse coordinates in line '{raw_line}': {exc}") from exc
        labels.append(label)
        points.append(coords)
    if len(points) < 2:
        raise ValueError("Provide at least two q-points to define a path")
    return labels, points


def _primitive_meta(primitive) -> Tuple[List[dict], np.ndarray]:
    atoms = []
    lattice = np.array(primitive.cell, dtype=float)
    scaled = primitive.scaled_positions
    masses = primitive.masses
    for sym, frac, mass in zip(primitive.symbols, scaled, masses):
        atoms.append({
            "symbol": sym,
            "frac": np.array(frac, dtype=float).tolist(),
            "mass": float(mass),
        })
    return atoms, lattice


def _flatten_band_dict(band_dict: dict):
    qpoints_frac = []
    distances = []
    freq_rows = []
    eig_rows = [] if "eigenvectors" in band_dict else None

    global_idx = -1
    for path_idx, qpath in enumerate(band_dict["qpoints"]):
        dist_path = band_dict["distances"][path_idx]
        freq_path = band_dict["frequencies"][path_idx]
        eig_path = band_dict.get("eigenvectors")
        for iq in range(len(qpath)):
            qpoints_frac.append(np.array(qpath[iq], dtype=float).tolist())
            distances.append(dist_path[iq])
            freq_rows.append(freq_path[iq])
            if eig_rows is not None:
                eig_rows.append(eig_path[path_idx][iq])
            global_idx += 1

    frequencies = np.array(freq_rows, dtype=float)
    eigenvectors = None
    if eig_rows is not None:
        eigenvectors = np.asarray(eig_rows, dtype=np.complex128)
    return qpoints_frac, distances, frequencies, eigenvectors


def build_band_from_phonopy_files(phonopy_yaml, force_constants, qpath_text: str, points_per_segment: int):
    labels, points = _parse_special_points(qpath_text)
    #paths = _build_band_paths(points, points_per_segment)
    qpoints, connections = get_band_qpoints_and_path_connections([points], npoints=points_per_segment)
    yaml_path = _write_temp_file(phonopy_yaml, suffix=".yaml")
    fc_path = _write_temp_file(force_constants, suffix=".dat")
    try:
        phonon = phonopy.load(phonopy_yaml=yaml_path, force_constants_filename=fc_path)
        phonon.run_band_structure(qpoints, path_connections=connections, with_eigenvectors=True)
        band = phonon.get_band_structure_dict()
    finally:
        for tmp_path in (yaml_path, fc_path):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    qpoints_frac, distances, frequencies, eigenvectors = _flatten_band_dict(band)
    atoms_meta, lattice = _primitive_meta(phonon.primitive)

    if eigenvectors is not None:
        eig_arr = np.asarray(eigenvectors, dtype=np.complex128)
        if eig_arr.ndim == 3:
            nq, nbranches, ncomp = eig_arr.shape
            if len(atoms_meta) == 0 or ncomp % 3 != 0:
                raise ValueError(
                    "Unexpected eigenvector shape from phonopy (cannot reshape to (natoms, 3))"
                )
            natoms = ncomp // 3
            eig_arr = eig_arr.reshape(nq, natoms, 3, nbranches)
            eig_arr = np.transpose(eig_arr, (0, 3, 1, 2))  # [nq, nbranches, natoms, 3]
        elif eig_arr.ndim != 4:
            raise ValueError("Unexpected eigenvector dimensions from phonopy")
        eigenvectors = eig_arr

    uc_ph = {
        "qpoints": distances,
        "qpoints_frac": qpoints_frac,
        "frequencies": frequencies,
        "nbranches": frequencies.shape[1],
        "has_eigenvectors": eigenvectors is not None,
    }
    if eigenvectors is not None:
        uc_ph["eigenvectors"] = eigenvectors.tolist()

    meta = {
        "lattice": lattice.tolist(),
        "atoms": atoms_meta,
        "labels": labels,
    }
    return uc_ph, meta