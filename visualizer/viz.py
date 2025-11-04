from collections import OrderedDict
from io import StringIO

import numpy as np
import plotly.graph_objects as go
from ase import Atoms
from ase.io import write

# ---------------- Band figure ----------------

def make_phonon_band_figure(phonon, meta=None):
    """
    Plot phonon bands vs k-point INDEX (0..N-1), not distance.
    Labels in meta['labels'] are placed at their q-point indices.
    """
    y = np.asarray(phonon["frequencies"], float)  # [nq, nb]
    nq, nb = y.shape
    x = np.arange(nq, dtype=int)  # k-point index

    fig = go.Figure()
    for b in range(nb):
        fig.add_trace(go.Scatter(x=x, y=y[:, b], mode="lines", showlegend=False))

    labels = (meta or {}).get("labels", {})
    if labels:
        idxs = sorted(int(i) for i in labels.keys() if 0 <= int(i) < nq)
        tickvals, ticktext = [], []
        for i in idxs:
            lab = str(labels[i]).replace("GAMMA", "Γ").replace("\\Gamma", "Γ").replace("$\\Gamma$", "Γ")
            tickvals.append(i); ticktext.append(lab)
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, title_text="k-point index")
        # Optional: faint vertical separators at labeled points
        for i in idxs:
            fig.add_vline(x=i, line_width=1, line_dash="dot")
    else:
        fig.update_xaxes(title_text="k-point index")

    fig.update_yaxes(title_text="Frequency (THz)")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig

# ---------------- Supercell builder ----------------

def build_supercell(meta, reps=(1, 1, 1)):
    """
    Build an ASE Atoms object for the supercell.

    meta must contain:
      - 'lattice' : (3,3) cell vectors (Å)
      - 'atoms' : list of { 'symbol': str, 'frac': (3,), 'mass': float }

    Returns:
      atoms: ASE Atoms object representing the supercell
    """
    lattice = np.asarray(meta["lattice"], float)
    
    # Create base unit cell as ASE Atoms
    symbols = [a["symbol"] for a in meta["atoms"]]
    scaled_positions = [a["frac"] for a in meta["atoms"]]
    masses = [a.get("mass", 0.0) for a in meta["atoms"]]
    
    base_atoms = Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=lattice,
        pbc=True,
        masses=masses,
    )
    
    # Build supercell by replicating
    reps = tuple(int(x) for x in reps)
    super_lattice = np.array([lattice[idx] * reps[idx] for idx in range(3)], dtype=float)

    all_frac, all_species, all_masses = [], [], []
    for i in range(reps[0]):
        for j in range(reps[1]):
            for k in range(reps[2]):
                shift = np.array([i, j, k], float)
                for p, s, m in zip(scaled_positions, symbols, masses):
                    all_frac.append((p + shift) / reps)
                    all_species.append(s)
                    all_masses.append(m)
    
    supercell_atoms = Atoms(
        symbols=all_species,
        scaled_positions=all_frac,
        cell=super_lattice,
        pbc=True,
        masses=all_masses,
    )
    
    return supercell_atoms


def _mass_weighted_mode_displacement(atoms, eigenvectors, qvec, unit_cell_lattice=None):
    """
    Core helper that builds the complex mass-weighted displacement for a mode.
    Returns an array shaped (N, 3) of complex numbers.
    
    Args:
        atoms: ASE Atoms object (supercell)
        eigenvectors: array of shape (n_prim, 3) complex, from primitive cell
        qvec: (3,) fractional q-vector in reciprocal lattice units (unit cell basis)
        unit_cell_lattice: (3,3) unit cell lattice vectors (optional, inferred if None)
    """
    R = atoms.get_positions()  # [N, 3] Cartesian
    ev = np.asarray(eigenvectors, complex)
    if ev.ndim != 2 or ev.shape[1] != 3:
        raise ValueError("Eigenvectors must have shape (natoms, 3)")

    N = len(atoms)
    nat = len(ev)
    if nat == 0:
        raise ValueError("Eigenvectors list is empty")

    # Infer supercell repetition factor (assuming cubic supercell)
    reps_total = N / nat
    if abs(reps_total - round(reps_total)) > 1e-6:
        raise ValueError("Eigenvectors cannot be tiled to match supercell size")
    reps_total = int(round(reps_total))
    
    # For cubic supercell: reps = (n, n, n) where n = (reps_total)^(1/3)
    reps = tuple(int(round(reps_total ** (1/3))) for _ in range(3))
    if reps[0] * reps[1] * reps[2] != reps_total:
        # Non-cubic or non-uniform repetition - infer from cell dimensions
        supercell_cell = atoms.cell
        if unit_cell_lattice is None:
            # Try to infer unit cell lattice from supercell
            # For a simple repetition, cell vectors are scaled by reps
            # We'll use the fact that we can infer reps from cell dimensions
            # For now, assume uniform repetition
            reps = (reps_total**(1/3),) * 3
    
    ev_sc = np.vstack([ev for _ in range(reps_total)])

    masses = atoms.get_masses()  # [N]
    if len(masses) != N:
        raise ValueError(
            f"Mass array length {len(masses)} does not match number of atoms {N}"
        )
    mass_factor = 1.0 / np.sqrt(masses)[:, None]

    # Calculate fractional coordinates in UNIT CELL coordinate system
    # This is critical: the q-vector is in reciprocal lattice units of the unit cell,
    # so we need fractional coordinates relative to the unit cell, not the supercell.
    if unit_cell_lattice is not None:
        # Use provided unit cell lattice
        unit_lattice = np.asarray(unit_cell_lattice, float)
        # Convert Cartesian to fractional using unit cell lattice
        # For row-vector lattice: r_frac = r_cart @ lattice^(-1)
        unit_lattice_inv = np.linalg.inv(unit_lattice.T).T
        r_frac = R @ unit_lattice_inv
    else:
        # Infer unit cell lattice from supercell
        # For a supercell with reps=(n1,n2,n3), unit_cell = supercell_cell / reps
        supercell_cell = atoms.cell
        reps_array = np.array(reps, float)
        unit_lattice = supercell_cell / reps_array[:, None]
        unit_lattice_inv = np.linalg.inv(unit_lattice.T).T
        r_frac = R @ unit_lattice_inv
    
    # Phase factor: exp(2πi q·r) where q is fractional q-vector and r is fractional position
    # Both should be in unit cell coordinate system
    phase = np.exp(2j * np.pi * (r_frac @ np.array(qvec)))
    # Displacement: eigenvector * phase * mass_factor * amplitude
    # The real part gives the displacement at phase=0
    disp_complex = ev_sc * phase[:, None] * mass_factor

    return ev_sc * phase[:, None] * mass_factor


def compute_mode_displacement(atoms, eigenvectors, qvec, amp=0.2, phase=0.0, unit_cell_lattice=None):
    """
    Return the Cartesian displacement for a phonon mode at a given phase.

    Args:
        atoms: ASE Atoms object (supercell)
        eigenvectors: array of shape (n_prim, 3) complex, from primitive cell
        qvec: (3,) fractional q-vector in reciprocal lattice units (unit cell basis)
        amp: dimensionless amplitude scale factor
        phase: phase in radians (default 0.0 = maximum displacement)
        unit_cell_lattice: (3,3) unit cell lattice vectors (optional, inferred if None)
    
    Returns:
        disp: (N, 3) array of Cartesian displacements in Å
    """
    disp0 = _mass_weighted_mode_displacement(atoms, eigenvectors, qvec, unit_cell_lattice)
    disp = np.real(disp0 * np.exp(1j * phase)) * amp
    return disp

# ---------------- PDB helpers ----------------

def _cell_params(lattice):
    a, b, c = lattice
    la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)

    def ang(u, v):
        cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        cosang = np.clip(cosang, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    alpha = ang(b, c)
    beta = ang(a, c)
    gamma = ang(a, b)
    return la, lb, lc, alpha, beta, gamma

def _pdb_from_atoms_single(atoms, include_cryst1=True):
    """Return a PDB string for a single frame (no MODEL/ENDMDL)."""
    lines = []
    if include_cryst1:
        la, lb, lc, alpha, beta, gamma = _cell_params(atoms.cell)
        lines.append(f"CRYST1{la:9.3f}{lb:9.3f}{lc:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1")
    R = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    for i, (sym, pos) in enumerate(zip(symbols, R), start=1):
        lines.append(
            f"ATOM  {i:5d} {sym:>2s}   MOL A   1    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00          {sym:>2s}"
        )
    lines.append("END")
    return "\n".join(lines)

# ---------------- Mode frames (numerics) ----------------

def _generate_mode_frames(atoms, eigenvectors, qvec, amp=0.2, steps=24, unit_cell_lattice=None):
    """
    Build a list of per-frame coordinates for a phonon mode animation.
    
    Args:
        atoms: ASE Atoms object (supercell)
        eigenvectors: [n_prim,3] complex, from primitive cell
        qvec: (3,) fractional q-vector in unit cell basis
        amp: dimensionless amplitude scale factor
        steps: number of frames per period
        unit_cell_lattice: (3,3) unit cell lattice vectors (optional, inferred if None)
    
    Returns:
        frames: list of numpy arrays, each [N,3] Cartesian coordinates
    """
    R = atoms.get_positions()  # [N, 3]
    disp0 = _mass_weighted_mode_displacement(atoms, eigenvectors, qvec, unit_cell_lattice)

    frames = []
    for t in range(steps):
        angle = 2 * np.pi * (t / steps)
        disp_t = np.real(disp0 * np.exp(1j * angle)) * amp
        frames.append(R + disp_t)
    return frames

def _serialize_frames_js(frames_R):
    """
    Convert list of [N,3] arrays into a JS array of Float32Array literals.
    Returns a big string like: [new Float32Array([...]), new Float32Array([...]), ...]
    """
    js_chunks = []
    for R in frames_R:
        flat = R.reshape(-1)
        nums = ",".join(f"{x:.6f}" for x in flat)
        js_chunks.append(f"new Float32Array([{nums}])")
    return "[" + ",".join(js_chunks) + "]"

# ---------------- HTML animation (widget-less) ----------------

def make_mode_animation_html(atoms, eigenvectors, qvec, amp=0.2, steps=24, height_px=520, fps=20, unit_cell_lattice=None):
    """
    Return a standalone HTML string that animates the mode using NGL.js
    by updating atomic coordinates every tick (no ipywidgets, no NGL trajectory).
    
    Args:
        atoms: ASE Atoms object (supercell)
        eigenvectors: [n_prim,3] complex, from primitive cell
        qvec: (3,) fractional q-vector in unit cell basis
        amp: dimensionless amplitude scale factor
        steps: number of frames per period
        height_px: height of viewport in pixels
        fps: frames per second
        unit_cell_lattice: (3,3) unit cell lattice vectors (optional, inferred if None)
    """
    frames_R = _generate_mode_frames(atoms, eigenvectors, qvec, amp=amp, steps=steps, unit_cell_lattice=unit_cell_lattice)
    # Create temporary Atoms object for first frame
    atoms_frame0 = atoms.copy()
    atoms_frame0.set_positions(frames_R[0])
    pdb0 = _pdb_from_atoms_single(atoms_frame0, include_cryst1=True)
    frames_js = _serialize_frames_js(frames_R)
    interval_ms = max(10, int(1000 / max(1, fps)))

    # Double braces {{ }} to emit literal braces inside f-string.
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html, body {{ margin:0; padding:0; }}
  #viewport {{ width:100%; height:{int(height_px)}px; }}
</style>
</head>
<body>
<div id="viewport"></div>
<script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
<script>
  (function() {{
    var stage = new NGL.Stage("viewport");
    window.addEventListener("resize", function() {{ stage.handleResize(); }});

    var pdbText = `{pdb0.replace("`","\\`")}`;
    var frames = {frames_js};   // Array of Float32Array, length = {len(frames_R)}

    stage.loadFile(new Blob([pdbText], {{ type: "text/plain" }}), {{ ext: "pdb" }}).then(function(comp) {{
      comp.addRepresentation("ball+stick");
      comp.addRepresentation("unitcell");
      comp.autoView();

      var structure = comp.structure;
      var atomCount = structure.atomStore.count;
      var coordsLen = atomCount * 3;

      if (frames.length === 0 || frames[0].length !== coordsLen) {{
        console.error("Frame size mismatch:", frames.length, frames[0] && frames[0].length, "expected", coordsLen);
        return;
      }}

      var i = 0;
      function tick() {{
        structure.updatePosition(frames[i]);
        comp.updateRepresentations({{ what: {{ position: true }} }});
        i = (i + 1) % frames.length;
      }}

      tick();
      var timer = setInterval(tick, {interval_ms});
    }});
  }})();
</script>
</body>
</html>
"""
    return html


def make_poscar_with_displacements(atoms, disp_cart, comment="Generated by Phonon Visualizer"):
    """
    Build a VASP POSCAR-style string for the displaced configuration using ASE.

    Each atomic line includes a trailing comment containing the Cartesian
    displacement in Å for clarity.

    Args:
        atoms: ASE Atoms object with displaced positions
        disp_cart: (N, 3) array of Cartesian displacements in Å
        comment: comment string for POSCAR header
    
    Returns:
        poscar_string: VASP POSCAR format string
    """
    # Create a copy of atoms with displaced positions
    atoms_disp = atoms.copy()
    positions = atoms.get_positions()
    disp_cart = np.asarray(disp_cart, float)

    if positions.shape != disp_cart.shape:
        raise ValueError("Coordinate and displacement arrays must have matching shapes")

    atoms_disp.set_positions(positions + disp_cart)
    
    # Use ASE to write POSCAR, but we need to add displacement comments
    # ASE doesn't support comments in POSCAR, so we'll write it and then add comments
    poscar_io = StringIO()
    write(poscar_io, atoms_disp, format='vasp', vasp5=True, direct=True)
    poscar_lines = poscar_io.getvalue().split('\n')
    
    # Replace the comment line with our custom comment
    if len(poscar_lines) > 0:
        poscar_lines[0] = str(comment)
    
    # Add displacement comments to atomic lines
    # POSCAR format: Direct coordinates start after "Direct" line
    direct_idx = None
    for i, line in enumerate(poscar_lines):
        if line.strip().upper() == 'DIRECT' or line.strip().upper() == 'CARTESIAN':
            direct_idx = i
            break
    
    if direct_idx is not None:
        # Get fractional coordinates and wrap to [0,1) for POSCAR format
        frac = atoms_disp.get_scaled_positions()
        frac = np.mod(frac, 1.0)  # Wrap to [0,1) as per POSCAR convention
        symbols = atoms_disp.get_chemical_symbols()
        
        # Replace atomic coordinate lines with commented versions
        atom_lines = []
        for i, (sym, fcoord, disp) in enumerate(zip(symbols, frac, disp_cart)):
            atom_lines.append(
                f"  {fcoord[0]:>12.8f} {fcoord[1]:>12.8f} {fcoord[2]:>12.8f}   "
                f"! {sym:>2s} disp_cart=({disp[0]:+.6f} {disp[1]:+.6f} {disp[2]:+.6f})"
            )
        
        # Replace the coordinate section
        poscar_lines = poscar_lines[:direct_idx+1] + atom_lines
    
    return '\n'.join(poscar_lines) + '\n'
