from __future__ import annotations

import numpy as np
from ase import Atoms


def infer_cubic_repeats(ts_nat: int, uc_nat: int) -> tuple[int, int, int]:
    if uc_nat == 0 or ts_nat % uc_nat != 0:
        raise ValueError("Atom counts are not commensurate with a cubic supercell")
    ratio = ts_nat // uc_nat
    root = round(ratio ** (1 / 3))
    if root ** 3 != ratio:
        raise ValueError("Cannot infer an integer cubic supercell from the atom counts")
    return (root, root, root)


def build_supercell_mode_stepwise(eigvec_uc: np.ndarray, q_frac: np.ndarray,
                                  repeats: tuple[int, int, int]) -> np.ndarray:
    eigvec_uc = np.asarray(eigvec_uc, dtype=np.complex128)
    q_frac = np.asarray(q_frac, dtype=float)
    nx, ny, nz = repeats

    def replicate(block: np.ndarray, count: int, phase_component: float) -> np.ndarray:
        pieces = [block]
        if count > 1:
            for m in range(1, count):
                phase = np.exp(2j * np.pi * phase_component * m)
                pieces.append(block * phase)
        return np.concatenate(pieces, axis=0)

    mode_x = replicate(eigvec_uc, nx, q_frac[0])
    mode_xy = replicate(mode_x, ny, q_frac[1])
    mode_xyz = replicate(mode_xy, nz, q_frac[2])
    return mode_xyz


def expand_unit_cell_modes(unit_eigvecs: np.ndarray, qpoints_frac: np.ndarray,
                           repeats: tuple[int, int, int]) -> np.ndarray:
    unit_eigvecs = np.asarray(unit_eigvecs, dtype=np.complex128)
    qpoints_frac = np.asarray(qpoints_frac, dtype=float)
    nq, nbranches, _, _ = unit_eigvecs.shape
    supercell_natoms = unit_eigvecs.shape[2] * int(np.prod(repeats))
    super_modes = np.empty((nq, nbranches, supercell_natoms, 3), dtype=np.complex128)
    for iq in range(nq):
        q_frac = qpoints_frac[iq]
        for ib in range(nbranches):
            mode = build_supercell_mode_stepwise(unit_eigvecs[iq, ib], q_frac, repeats)
            super_modes[iq, ib] = mode
    return super_modes


def map_ts_to_supercell(ts_atoms: Atoms, supercell_atoms: Atoms, tol: float = 1e-3) -> list[int]:
    ts_frac = ts_atoms.get_scaled_positions()
    sc_frac = supercell_atoms.get_scaled_positions()
    ts_syms = np.array(ts_atoms.get_chemical_symbols())
    sc_syms = np.array(supercell_atoms.get_chemical_symbols())
    cell = np.array(supercell_atoms.get_cell(), dtype=float)

    mapping: list[int] = []
    used = set()
    for i, (sym, frac) in enumerate(zip(ts_syms, ts_frac)):
        candidates = np.where(sc_syms == sym)[0]
        if not len(candidates):
            raise ValueError(f"No supercell atoms with symbol {sym} to match TS atom {i}")

        best = None
        best_dist = None
        for j in candidates:
            if j in used:
                continue
            diff = frac - sc_frac[j]
            diff -= np.round(diff)
            if np.max(np.abs(diff)) > tol:
                continue
            cart = diff @ cell
            dist = np.linalg.norm(cart)
            if best is None or dist < best_dist:
                best = j
                best_dist = dist

        if best is None:
            for j in candidates:
                if j in used:
                    continue
                diff = frac - sc_frac[j]
                diff -= np.round(diff)
                cart = diff @ cell
                dist = np.linalg.norm(cart)
                if best is None or dist < best_dist:
                    best = j
                    best_dist = dist

        if best is None:
            raise ValueError(f"Failed to map TS atom {i} ({sym}) to any supercell site")
        used.add(best)
        mapping.append(best)
    return mapping


def reorder_modes(supercell_modes: np.ndarray, mapping: list[int]) -> np.ndarray:
    mapping = np.asarray(mapping, dtype=int)
    return supercell_modes[:, :, mapping, :]
