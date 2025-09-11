import numpy as np
import plotly.graph_objects as go

# ---------------- Band figure ----------------

def make_phonon_band_figure(band):
    """Plot phonon bands vs k-point index using a Phonopy BandStructure."""
    y = np.asarray(band.frequencies, float)  # [nq, nb]
    nq, nb = y.shape
    x = np.arange(nq, dtype=int)  # k-point index

    fig = go.Figure()
    for b in range(nb):
        fig.add_trace(go.Scatter(x=x, y=y[:, b], mode="lines", showlegend=False))

    labels = getattr(band, "labels", None)
    if labels:
        idxs = [i for i, lab in enumerate(labels) if lab]
        tickvals, ticktext = [], []
        for i in idxs:
            lab = str(labels[i]).replace("GAMMA", "Γ").replace("\\Gamma", "Γ").replace("$\\Gamma$", "Γ")
            tickvals.append(i)
            ticktext.append(lab)
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, title_text="k-point index")
        for i in idxs:
            fig.add_vline(x=i, line_width=1, line_dash="dot")
    else:
        fig.update_xaxes(title_text="k-point index")

    fig.update_yaxes(title_text="Frequency (THz)")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig

# ---------------- Supercell builder ----------------

def build_supercell(phonon, reps=(1, 1, 1)):
    """Return supercell cartesian coordinates, species and lattice."""
    cell = phonon.unitcell
    lattice = np.asarray(cell.cell, float)
    base = np.asarray(cell.scaled_positions, float)
    species = list(cell.symbols)

    reps = tuple(int(x) for x in reps)
    all_frac, all_species = [], []
    for i in range(reps[0]):
        for j in range(reps[1]):
            for k in range(reps[2]):
                shift = np.array([i, j, k], float)
                for p, s in zip(base, species):
                    all_frac.append((p + shift) / reps)
                    all_species.append(s)
    all_frac = np.vstack(all_frac)
    R = all_frac @ lattice
    return R, all_species, lattice

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
    ev = np.asarray(eigenvectors, complex)
    N = len(R)
    nat = len(ev)
    reps = int(round(N / nat)) if nat > 0 else 1
    ev_sc = np.vstack([ev for _ in range(reps)])

    masses = np.array(masses, float)
    if len(masses) != N:
        raise ValueError(f"Mass array length {len(masses)} does not match number of atoms {N}")
    mass_factor = 1.0 / np.sqrt(masses)[:, None]  # shape (N,1)

    # fractional coords corresponding to R
    a, b, c = lattice
    M = np.vstack([a, b, c]).T
    finv = np.linalg.inv(M)
    r_frac = (R @ finv.T)
    phase = np.exp(2j * np.pi * (r_frac @ np.array(qvec)))

    # apply phase and mass weighting
    disp0 = ev_sc * phase[:, None] * mass_factor

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
