import streamlit as st
from phonon_parser import load_band_yaml
from viz import (
    make_phonon_band_figure,
    build_supercell,
    make_mode_animation_html,
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

try:
    phonon, band = load_band_yaml(uploaded)
except Exception as e:
    st.exception(e)
    st.stop()

col1, col2 = st.columns([1.2, 1.0], gap="large")

# Band plot (k-point index on x-axis)
with col1:
    fig_band = make_phonon_band_figure(band)
    st.plotly_chart(fig_band, use_container_width=True)

# Mode animation
with col2:
    st.subheader("Animate a mode")
    if band.eigenvectors is None:
        st.warning("This file lacks eigenvectors. Re-run Phonopy with `--eigenvectors` (or `EIGENVECTORS = .TRUE.`).")

    nq = len(band.qpoints)
    nb = band.frequencies.shape[1]

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

    if band.eigenvectors is not None:
        # Build supercell geometry
        R, species, lattice = build_supercell(phonon, reps=(supercell, supercell, supercell))

        # Mass list replicated to the supercell order (mass-weighted displacement 1/sqrt(m))
        base_masses = list(phonon.unitcell.masses)
        masses = base_masses * (supercell ** 3)

        try:
            html = make_mode_animation_html(
                R=R,
                species=species,
                lattice=lattice,
                eigenvectors=band.eigenvectors[q_idx][b_idx],
                qvec=band.qpoints[q_idx],
                masses=masses,
                amp=amp,               # now dimensionless scale factor (larger default)
                steps=frames,
                height_px=520,
                fps=fps,
            )
            st.components.v1.html(html, height=540, scrolling=False)
            st.caption("Mass-weighted animation (displacements scaled by 1/√m and a dimensionless amplitude). Rendered with pure NGL.js for Streamlit.")
        except Exception as e:
            st.exception(e)
