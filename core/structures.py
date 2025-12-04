import io
from typing import Tuple

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.io import read as ase_read
from ase.io.espresso import read_espresso_in


def _is_qe_file(name: str | None) -> bool:
    return bool(name and name.lower().endswith(".in"))


def read_structure(source) -> Atoms:
    """Read an ASE-compatible structure upload (QE .in routed through read_espresso_in)."""
    if source is None:
        raise ValueError("Structure source is missing")

    original_name = getattr(source, "name", None)

    # If the source is already a filesystem path
    if isinstance(source, str):
        if _is_qe_file(source):
            with open(source, "r", encoding="utf-8") as handle:
                return read_espresso_in(handle)
        return ase_read(source)

    # Normalize various upload/file-like types to raw bytes
    if hasattr(source, "getvalue"):
        data = source.getvalue()
        if original_name is None:
            original_name = getattr(source, "name", None)
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif hasattr(source, "read"):
        data = source.read()
    else:
        raise TypeError("Unsupported structure source type")

    if not data:
        raise ValueError("Structure file is empty")

    if _is_qe_file(original_name):
        text = data.decode("utf-8", errors="ignore")
        return read_espresso_in(io.StringIO(text))

    buffer = io.BytesIO(data)
    if original_name:
        buffer.name = original_name

    try:
        buffer.seek(0)
        return ase_read(buffer)
    except TypeError as exc:
        msg = str(exc)
        if "startswith first arg" not in msg:
            raise
        text_io = io.StringIO(data.decode("utf-8", errors="ignore"))
        if original_name:
            text_io.name = original_name  # type: ignore[attr-defined]
        return ase_read(text_io)


def drop_atom(atoms: Atoms, index: int) -> Atoms:
    """Return a copy of atoms without the atom at the requested index (supports negatives)."""
    n = len(atoms)
    idx = index if index >= 0 else n + index
    if idx < 0 or idx >= n:
        raise IndexError("Atom index out of range")
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    return atoms[mask]


def meta_to_ase_atoms(meta: dict) -> Atoms:
    lattice = np.asarray(meta.get("lattice"), dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError("Meta lacks a valid 3x3 lattice")
    atoms_meta = meta.get("atoms") or []
    if not atoms_meta:
        raise ValueError("Meta lacks atom coordinates")
    symbols = [a["symbol"] for a in atoms_meta]
    scaled = [a["frac"] for a in atoms_meta]
    masses = [a.get("mass") for a in atoms_meta]
    return Atoms(symbols=symbols, scaled_positions=scaled, cell=lattice, pbc=True, masses=masses)


def build_supercell_atoms(meta: dict, repeats: Tuple[int, int, int]) -> Atoms:
    base = meta_to_ase_atoms(meta)
    P = np.diag(repeats)
    return make_supercell(base, P)
