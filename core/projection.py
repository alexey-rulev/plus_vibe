from __future__ import annotations

from typing import Tuple

import numpy as np
from ase import Atoms


def _wrap_fractional(dfrac: np.ndarray) -> np.ndarray:
    return (dfrac + 0.5) % 1.0 - 0.5


def perp_vec_T_to_line_IF(I_atoms: Atoms, F_atoms: Atoms, T_atoms: Atoms) -> np.ndarray:
    if not (len(I_atoms) == len(F_atoms) == len(T_atoms)):
        raise ValueError("All structures must contain the same number of atoms")

    cell = np.array(I_atoms.get_cell(), dtype=float)
    if not (np.allclose(cell, F_atoms.get_cell(), atol=1e-6) and np.allclose(cell, T_atoms.get_cell(), atol=1e-6)):
        raise ValueError("Initial, final, and TS structures must share identical lattice vectors")

    frac_I = I_atoms.get_scaled_positions()
    frac_F = F_atoms.get_scaled_positions()
    frac_T = T_atoms.get_scaled_positions()

    dfrac_IF = _wrap_fractional(frac_F - frac_I)
    dfrac_IT = _wrap_fractional(frac_T - frac_I)

    dcart_IF = dfrac_IF @ cell
    dcart_IT = dfrac_IT @ cell

    a = dcart_IF.reshape(-1)
    w = dcart_IT.reshape(-1)
    denom = np.dot(a, a)
    if denom == 0.0:
        raise ValueError("Initial and final configurations coincide; direction undefined")

    lam = np.dot(w, a) / denom
    v_flat = lam * a - w
    return v_flat.reshape(dcart_IT.shape)


def trim_structures(initial: Atoms, ts: Atoms, final: Atoms, drop_index: int) -> Tuple[Atoms, Atoms, Atoms]:
    from .structures import drop_atom

    return drop_atom(initial, drop_index), drop_atom(ts, drop_index), drop_atom(final, drop_index)


def compute_projection_modulus(supercell_modes: np.ndarray, dx_cart: np.ndarray,
                               masses: np.ndarray | None = None,
                               mass_weight: bool = False,
                               center: bool = True) -> np.ndarray:
    dx = np.asarray(dx_cart, dtype=float)
    if center:
        dx = dx - dx.mean(axis=0, keepdims=True)
    if mass_weight and masses is not None:
        factors = 1.0 / np.sqrt(np.asarray(masses, dtype=float))[:, None]
        dx = dx * factors

    dx_flat = dx.reshape(-1)
    nq, nbranches, _, _ = supercell_modes.shape
    projections = np.empty((nq, nbranches), dtype=float)
    for iq in range(nq):
        for ib in range(nbranches):
            eig_flat = supercell_modes[iq, ib].reshape(-1)
            proj = np.vdot(eig_flat, dx_flat)
            projections[iq, ib] = np.abs(proj)
    return projections
