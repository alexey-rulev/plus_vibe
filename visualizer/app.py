import yaml
import numpy as np
import streamlit as st
from phonon_parser import parse_band_yaml
from viz import (
    make_phonon_band_figure,
    build_supercell,
    make_mode_animation_html,
    compute_mode_displacement,
    make_poscar_with_displacements,
)

st.set_page_config(page_title="Phonon (Phonopy) Visualizer", layout="wide")
st.title("📈 Phonon Visualizer (Phonopy band.yaml)")
st.caption(
    "Upload a Phonopy **band.yaml** (generate with eigenvectors if you want to animate modes). "
    "Band plot is vs. k-point index; animation uses mass-weighted displacements (1/√m)."
)

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("band.yaml", type=["yaml", "yml"], accept_multiple_files=False)

    st.header("Supercell & Animation")
    supercell = st.slider("Supercell repetitions (integer)", 1, 3, 1)
    # Amplitude is now dimensionless (scale factor) and ~10x larger by default
    amp = st.slider("Amplitude (dimensionless scale)", 0.0, 10.0, 2.0, 0.1)
    frames = st.slider("Animation frames per period", 10, 60, 24, 2)
    fps = st.slider("Playback FPS", 5, 60, 20, 1)

if not uploaded:
    st.info("Drop a **band.yaml** here to begin.")
    st.stop()

# Load YAML
try:
    data = yaml.safe_load(uploaded)
except Exception as e:
    st.error(f"Failed to parse YAML: {e}")
    st.stop()

# Parse to structured dicts
try:
    phonon, meta = parse_band_yaml(data)
except Exception as e:
    st.exception(e)
    st.stop()

col1, col2 = st.columns([1.2, 1.0], gap="large")

# Band plot (k-point index on x-axis)
with col1:
    fig_band = make_phonon_band_figure(phonon, meta)
    st.plotly_chart(fig_band, use_container_width=True)

# Mode animation
with col2:
    st.subheader("Animate a mode")
    if not phonon.get("has_eigenvectors", False):
        st.warning("This file lacks eigenvectors. Re-run Phonopy with `--eigenvectors` (or `EIGENVECTORS = .TRUE.`).")

    nq = len(phonon["qpoints"])
    nb = phonon["nbranches"]

    # Choose how to enter indices
    entry_mode = st.radio(
        "Set indices via:",
        options=["Sliders", "Keyboard"],
        horizontal=True,
    )

    if entry_mode == "Sliders":
        q_idx = st.slider("q-point index", 0, nq - 1, min(10, nq - 1))
        b_idx = st.slider("Branch index (0 = acoustic)", 0, nb - 1, min(3, nb - 1))
    else:
        q_idx = st.number_input("q-point index", min_value=0, max_value=nq - 1, value=min(10, nq - 1), step=1)
        b_idx = st.number_input("Branch index (0 = acoustic)", min_value=0, max_value=nb - 1, value=min(3, nb - 1), step=1)

    # Display selected q-vector information
    if "qpoints_frac" in phonon and len(phonon["qpoints_frac"]) > q_idx:
        qvec_frac = phonon["qpoints_frac"][q_idx]
        qvec_frac_str = f"({qvec_frac[0]:.6f}, {qvec_frac[1]:.6f}, {qvec_frac[2]:.6f})"
        
        # Compute Cartesian q-vector if lattice is available
        qvec_cart_str = None
        if meta.get("lattice") is not None:
            lattice = np.asarray(meta["lattice"], float)
            # Reciprocal lattice vectors: b_i = 2π * (a_j × a_k) / (a_i · (a_j × a_k))
            # Using formula: b_i = 2π * ε_{ijk} * a_j × a_k / V where V is cell volume
            volume = np.abs(np.linalg.det(lattice))
            reciprocal_lattice = 2 * np.pi * np.linalg.inv(lattice).T
            qvec_cart = qvec_frac @ reciprocal_lattice
            qvec_cart_str = f"({qvec_cart[0]:.6f}, {qvec_cart[1]:.6f}, {qvec_cart[2]:.6f}) Å⁻¹"
        
        # Display in an info box
        q_info = f"**q-point index:** {q_idx}  \n**Fractional q-vector:** {qvec_frac_str}"
        if qvec_cart_str:
            q_info += f"  \n**Cartesian q-vector:** {qvec_cart_str}"
        st.info(q_info)

    if phonon.get("has_eigenvectors", False) and meta.get("atoms"):
        # Build supercell geometry as ASE Atoms object
        supercell_atoms = build_supercell(meta, reps=(supercell, supercell, supercell))
        
        # Get unit cell lattice for correct phase calculation
        unit_cell_lattice = np.asarray(meta["lattice"], float)

        try:
            displacement = compute_mode_displacement(
                atoms=supercell_atoms,
                eigenvectors=phonon["eigenvectors"][q_idx][b_idx],
                qvec=phonon["qpoints_frac"][q_idx],
                amp=amp,
                unit_cell_lattice=unit_cell_lattice,
            )
            # Create displaced atoms object
            displaced_atoms = supercell_atoms.copy()
            displaced_atoms.set_positions(supercell_atoms.get_positions() + displacement)

            html = make_mode_animation_html(
                atoms=supercell_atoms,
                eigenvectors=phonon["eigenvectors"][q_idx][b_idx],
                qvec=phonon["qpoints_frac"][q_idx],
                amp=amp,               # now dimensionless scale factor (larger default)
                steps=frames,
                height_px=520,
                fps=fps,
                unit_cell_lattice=unit_cell_lattice,
            )
            st.components.v1.html(html, height=540, scrolling=False)
            st.caption("Mass-weighted animation (displacements scaled by 1/√m and a dimensionless amplitude). Rendered with pure NGL.js for Streamlit.")

            poscar_comment = (
                f"Phonon Visualizer q-index={q_idx} branch={b_idx} supercell={supercell} amplitude={amp}"
            )
            poscar_comment_opposite = (
                f"Phonon Visualizer q-index={q_idx} branch={b_idx} supercell={supercell} amplitude={-amp}"
            )
            
            # Positive displacement
            poscar_data = make_poscar_with_displacements(
                atoms=supercell_atoms,
                disp_cart=displacement,
                comment=poscar_comment,
            )
            file_name = f"phonon_mode_q{q_idx}_b{b_idx}_sc{supercell}.vasp"
            
            # Negative displacement (opposite direction)
            poscar_data_opposite = make_poscar_with_displacements(
                atoms=supercell_atoms,
                disp_cart=-displacement,
                comment=poscar_comment_opposite,
            )
            file_name_opposite = f"phonon_mode_q{q_idx}_b{b_idx}_sc{supercell}_opposite.vasp"
            
            # Download buttons side by side
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.download_button(
                    "💾 Download displaced configuration (+)",
                    data=poscar_data,
                    file_name=file_name,
                    mime="text/plain",
                )
            with col_neg:
                st.download_button(
                    "💾 Download displaced configuration (-)",
                    data=poscar_data_opposite,
                    file_name=file_name_opposite,
                    mime="text/plain",
                )
        except Exception as e:
            st.exception(e)
    elif phonon.get("has_eigenvectors", False):
        st.info("YAML lacks atom list/lattice — cannot animate. Ensure `band.yaml` contains `lattice` and `points` entries.")
