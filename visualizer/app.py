
import yaml
import numpy as np
import streamlit as st
from phonon_parser import parse_band_yaml
from viz import make_phonon_band_figure, make_mode_animation_widget, build_supercell

st.set_page_config(page_title="Phonon (Phonopy) Visualizer", layout="wide")
st.title("📈 Phonon Visualizer (Phonopy band.yaml)")
st.caption("Upload a Phonopy **band.yaml** file (generated with `--eigenvectors` if you want to animate modes). See Phonopy docs.")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("band.yaml", type=["yaml", "yml"], accept_multiple_files=False)
    supercell = st.slider("Supercell repetitions (integer)", 1, 3, 1)
    amp = st.slider("Amplitude (Å)", 0.0, 0.6, 0.2, 0.02)
    frames = st.slider("Animation frames per period", 10, 60, 24, 2)

if not uploaded:
    st.info("Drop a **band.yaml** here to begin.")
    st.stop()

try:
    data = yaml.safe_load(uploaded)
except Exception as e:
    st.error(f"Failed to parse YAML: {e}")
    st.stop()

try:
    phonon, meta = parse_band_yaml(data)
except Exception as e:
    st.exception(e)
    st.stop()

col1, col2 = st.columns([1.2, 1.0], gap="large")

with col1:
    fig_band = make_phonon_band_figure(phonon, meta)
    st.plotly_chart(fig_band, use_container_width=True)

with col2:
    st.subheader("Animate a mode")
    if not phonon.get("has_eigenvectors", False):
        st.warning("This file lacks eigenvectors. Re-run Phonopy with `--eigenvectors` (or `EIGENVECTORS = .TRUE.`).")
    nq = len(phonon["qpoints"])
    nb = phonon["nbranches"]
    q_idx = st.slider("q-point index", 0, nq-1, min(10, nq-1))
    b_idx = st.slider("Branch index (0 = acoustic)", 0, nb-1, min(3, nb-1))

    if phonon.get("has_eigenvectors", False):
        R, species, lattice = build_supercell(meta, reps=(supercell, supercell, supercell))
        view = make_mode_animation_widget(
            R=R,
            species=species,
            lattice=lattice,
            eigenvectors=phonon["eigenvectors"][q_idx][b_idx],
            qvec=phonon["qpoints_frac"][q_idx],
            amp=amp,
            steps=frames,
        )
        st.components.v1.html(view._repr_html_(), height=500)
    else:
        st.info("Band plot shown on the left. Upload a file with eigenvectors to animate modes.")
