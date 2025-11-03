from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go

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
    Returns (R, species, lattice).

    meta must contain:
      - 'lattice' : (3,3) cell vectors (Å)
      - 'atoms' : list of { 'symbol': str, 'frac': (3,) }
    """
    lattice = np.asarray(meta["lattice"], float)
    base = np.array([a["frac"] for a in meta["atoms"]], dtype=float)
    species = [a["symbol"] for a in meta["atoms"]]

    reps = tuple(int(x) for x in reps)
    super_lattice = np.array([lattice[idx] * reps[idx] for idx in range(3)], dtype=float)

    all_frac, all_species = [], []
    for i in range(reps[0]):
        for j in range(reps[1]):
            for k in range(reps[2]):
                shift = np.array([i, j, k], float)
                for p, s in zip(base, species):
                    all_frac.append((p + shift) / reps)
                    all_species.append(s)
    all_frac = np.vstack(all_frac)
    R = all_frac @ super_lattice  # [N,3]
    return R, all_species, super_lattice


def _mass_weighted_mode_displacement(R, eigenvectors, qvec, lattice, masses):
    """
    Core helper that builds the complex mass-weighted displacement for a mode.
    Returns an array shaped (N, 3) of complex numbers.
    """
    R = np.asarray(R, float)
    ev = np.asarray(eigenvectors, complex)
    if ev.ndim != 2 or ev.shape[1] != 3:
        raise ValueError("Eigenvectors must have shape (natoms, 3)")

    N = len(R)
    nat = len(ev)
    if nat == 0:
        raise ValueError("Eigenvectors list is empty")

    reps = int(round(N / nat))
    if reps * nat != N:
        raise ValueError("Eigenvectors cannot be tiled to match supercell size")
    ev_sc = np.vstack([ev for _ in range(reps)])

    masses = np.array(masses, float)
    if len(masses) != N:
        raise ValueError(
            f"Mass array length {len(masses)} does not match number of atoms {N}"
        )
    mass_factor = 1.0 / np.sqrt(masses)[:, None]

    a, b, c = lattice
    M = np.vstack([a, b, c]).T
    finv = np.linalg.inv(M)
    r_frac = (R @ finv.T)
    phase = np.exp(2j * np.pi * (r_frac @ np.array(qvec)))

    return ev_sc * phase[:, None] * mass_factor


def compute_mode_displacement(R, eigenvectors, qvec, lattice, masses, amp=0.2, phase=0.0):
    """
    Return the Cartesian displacement for a phonon mode at a given phase.

    phase is expressed in radians. The default (0.0) corresponds to the
    maximum displacement along the real component of the eigenvector.
    """
    disp0 = _mass_weighted_mode_displacement(R, eigenvectors, qvec, lattice, masses)
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

def _pdb_from_atoms_single(R, species, lattice, include_cryst1=True):
    """Return a PDB string for a single frame (no MODEL/ENDMDL)."""
    lines = []
    if include_cryst1:
        la, lb, lc, alpha, beta, gamma = _cell_params(lattice)
        lines.append(f"CRYST1{la:9.3f}{lb:9.3f}{lc:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1")
    for i, (sym, pos) in enumerate(zip(species, R), start=1):
        lines.append(
            f"ATOM  {i:5d} {sym:>2s}   MOL A   1    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00          {sym:>2s}"
        )
    lines.append("END")
    return "\n".join(lines)

# ---------------- Mode frames (numerics) ----------------

def _generate_mode_frames(R, eigenvectors, qvec, lattice, masses, amp=0.2, steps=24):
    """
    Build a list of per-frame coordinates for a phonon mode animation.
    - R: [N,3] Cartesian Å (supercell)
    - eigenvectors: [n_prim,3] complex, repeated to supercell size if needed
    - qvec: (3,) fractional
    - masses: (N,) masses in amu
    Returns: list of numpy arrays, each [N,3]
    """
    R = np.asarray(R, float)
    disp0 = _mass_weighted_mode_displacement(R, eigenvectors, qvec, lattice, masses)

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

def make_mode_animation_html(R, species, lattice, eigenvectors, qvec, masses, amp=0.2, steps=24, height_px=520, fps=20):
    """
    Return a standalone HTML string that animates the mode using NGL.js
    by updating atomic coordinates every tick (no ipywidgets, no NGL trajectory).
    """
    frames_R = _generate_mode_frames(R, eigenvectors, qvec, lattice, masses, amp=amp, steps=steps)
    pdb0 = _pdb_from_atoms_single(frames_R[0], species, lattice, include_cryst1=True)
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


def make_poscar_with_displacements(
    lattice,
    species,
    coords_cart,
    disp_cart,
    comment="Generated by Phonon Visualizer",
):
    """
    Build a VASP POSCAR-style string for the displaced configuration.

    Each atomic line includes a trailing comment containing the Cartesian
    displacement in Å for clarity.
    """
    lattice = np.asarray(lattice, float)
    coords_cart = np.asarray(coords_cart, float)
    disp_cart = np.asarray(disp_cart, float)

    if coords_cart.shape != disp_cart.shape:
        raise ValueError("Coordinate and displacement arrays must have matching shapes")

    if len(coords_cart) != len(species):
        raise ValueError("Species list length must match number of coordinates")

    counts = OrderedDict()
    for sym in species:
        counts.setdefault(sym, 0)
        counts[sym] += 1

    a, b, c = lattice
    M = np.vstack([a, b, c]).T
    finv = np.linalg.inv(M)
    frac = (coords_cart @ finv.T)
    frac = np.mod(frac, 1.0)

    lines = []
    lines.append(str(comment))
    lines.append("1.0")
    for vec in lattice:
        lines.append(f"  {vec[0]:>16.10f} {vec[1]:>16.10f} {vec[2]:>16.10f}")

    lines.append("  " + "  ".join(counts.keys()))
    lines.append("  " + "  ".join(str(v) for v in counts.values()))
    lines.append("Direct")

    for sym, coord, disp in zip(species, frac, disp_cart):
        lines.append(
            "  {0:>12.8f} {1:>12.8f} {2:>12.8f}   ! {3:>2s} disp_cart=({4:+.6f} {5:+.6f} {6:+.6f})".format(
                coord[0],
                coord[1],
                coord[2],
                sym,
                disp[0],
                disp[1],
                disp[2],
            )
        )

    return "\n".join(lines) + "\n"
