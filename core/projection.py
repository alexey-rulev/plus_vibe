from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple

def atoms_to_arrays(atoms: List[dict]):
    """Return fractional coords, symbols, and masses as arrays."""
    frac = np.array([a.get("frac", a.get("coordinates")) for a in atoms], dtype=float)
    sym = np.array([a.get("symbol","X") for a in atoms])
    mass = np.array([float(a.get("mass", 0.0)) for a in atoms], dtype=float)
    return frac, sym, mass

def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional to Cartesian using lattice rows a,b,c (Angstrom)."""
    return frac @ lattice  # row-vector convention

def cart_to_frac(cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return cart @ np.linalg.inv(lattice)

@dataclass
class ProjectionSettings:
    exclude_atom_index: int = -1
    mass_weight: bool = True
    normalize: bool = True

def infer_supercell_mapping(uc_frac: np.ndarray, uc_sym: np.ndarray, sc_atoms: List[dict], tol=1e-3):
    sc_frac, sc_sym, sc_mass = atoms_to_arrays(sc_atoms)
    n_uc = len(uc_frac)

    sym_to_indices = {}
    for i, s in enumerate(sc_sym):
        sym_to_indices.setdefault(s, []).append(i)

    mapping: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for j in range(n_uc):
        s = uc_sym[j]
        cand = sym_to_indices.get(s, [])
        found = []
        for i in cand:
            df = sc_frac[i] - uc_frac[j]
            R = np.round(df)
            dwrap = df - R
            if np.max(np.abs(dwrap)) < tol:
                found.append((i, R.astype(int)))
        if not found:
            for i in cand:
                df = sc_frac[i] - uc_frac[j]
                R = np.round(df)
                dwrap = df - R
                if np.max(np.abs(dwrap)) < 5*tol:
                    found.append((i, R.astype(int)))
                    break
        if not found:
            return None
        mapping[j] = found
    return mapping

def build_periodic_displacement_q(dr_sc_cart: np.ndarray,
                                  mapping: Dict[int, List[Tuple[int, np.ndarray]]],
                                  uc_lattice: np.ndarray,
                                  qpoints_frac):
    """
    Construct u_j(q) = (1/sqrt(Nimg)) * sum_over_images dr_{j,R} * exp(-i 2pi qfrac dot Rint)
    where Rint is the integer lattice vector (Ra,Rb,Rc) for each image.
    Returns array of shape [nq, n_uc, 3] (complex128).
    """
    nq = len(qpoints_frac)
    n_uc = len(mapping.keys())
    u_q = np.zeros((nq, n_uc, 3), dtype=np.complex128)

    for j in range(n_uc):
        imgs = mapping[j]
        Nimg = max(1, len(imgs))

        Rint = np.array([R for (_, R) in imgs], dtype=int)
        dr_arr = np.array([dr_sc_cart[i] for (i, _) in imgs], dtype=float)

        for iq in range(nq):
            qfrac = np.array(qpoints_frac[iq], dtype=float)
            phases = np.exp(-1j * 2*np.pi * (Rint @ qfrac))
            contrib = (phases[:, None] * dr_arr).sum(axis=0) / np.sqrt(Nimg)
            u_q[iq, j, :] = contrib
    return u_q

def project_onto_modes(u_q: np.ndarray, uc_ph: dict, uc_meta: dict,
                       uc_mass: np.ndarray, settings: ProjectionSettings):
    """
    u_q: [nq, n_uc, 3] complex periodic displacement per atom
    uc_ph: dict with frequencies [nq, nbranches] and eigenvectors [nq][b][natoms][3]
    Returns A: [nq, nbranches] projection amplitudes.
    """
    nq = len(uc_ph["qpoints"])
    nbranches = uc_ph["nbranches"]
    nat = u_q.shape[1]
    ex = settings.exclude_atom_index if settings.exclude_atom_index >= 0 else (nat-1)

    A = np.zeros((nq, nbranches), dtype=float)

    if settings.normalize:
        mask = np.ones(nat, dtype=bool)
        if 0 <= ex < nat:
            mask[ex] = False
        if settings.mass_weight:
            norms = np.linalg.norm((np.sqrt(uc_mass[mask])[:,None] * u_q[:,mask,:]), axis=(1,2)) + 1e-12
        else:
            norms = np.linalg.norm(u_q[:,mask,:], axis=(1,2)) + 1e-12
    else:
        norms = np.ones(nq, dtype=float)

    for iq in range(nq):
        for ib in range(nbranches):
            vecs = uc_ph["eigenvectors"][iq][ib]
            accum = 0+0j
            for j in range(nat):
                if j == ex:
                    continue
                e_j = np.array(vecs[j], dtype=np.complex128)
                u_j = u_q[iq, j, :]
                w = np.sqrt(uc_mass[j]) if settings.mass_weight else 1.0
                accum += w * np.vdot(e_j, u_j)
            A[iq, ib] = np.abs(accum) / norms[iq]
    return A
