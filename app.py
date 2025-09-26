
import yaml, numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from core.loaders import load_band_yaml_from_bytes
from core.projection import (
    infer_supercell_mapping, atoms_to_arrays, frac_to_cart,
    project_onto_modes, build_periodic_displacement_q, ProjectionSettings
)
from core.plotting import plot_bands_with_projection

st.set_page_config(page_title="Phonon Projection", layout="wide")
st.title("Phonon-mode projection of TS displacement onto unit-cell modes")

st.sidebar.header("Inputs")
uc_file = st.sidebar.file_uploader("Unit cell band.yaml (with eigenvectors)", type=["yaml","yml"])
sc_eq_file = st.sidebar.file_uploader("Supercell — equilibrium band.yaml", type=["yaml","yml"])
sc_ts_file = st.sidebar.file_uploader("Supercell — transition state band.yaml", type=["yaml","yml"])

exclude_idx = st.sidebar.number_input("Exclude unit-cell atom index", min_value=-1, step=1, value=-1,
                                      help="Set -1 to use the last atom by default")
amp_scale = st.sidebar.slider("Marker size scale", 10.0, 5000.0, 800.0, step=10.0)
normalize = st.sidebar.checkbox("Normalize by total displacement", value=True)
mass_weight = st.sidebar.checkbox("Use mass-weighted projection", value=True)

run_btn = st.sidebar.button("Compute projection")

st.write("Upload all three files, then press **Compute projection**.")

if run_btn:
    if not (uc_file and sc_eq_file and sc_ts_file):
        st.error("Please upload all three files.")
        st.stop()

    uc_ph, uc_meta = load_band_yaml_from_bytes(uc_file.read())
    sc_eq_ph, sc_eq_meta = load_band_yaml_from_bytes(sc_eq_file.read())
    sc_ts_ph, sc_ts_meta = load_band_yaml_from_bytes(sc_ts_file.read())

    if not uc_ph.get("has_eigenvectors", False):
        st.error("Unit-cell band.yaml must contain eigenvectors.")
        st.stop()

    sc_eq_atoms = sc_eq_meta.get("atoms", [])
    sc_ts_atoms = sc_ts_meta.get("atoms", [])
    if len(sc_eq_atoms) != len(sc_ts_atoms) or len(sc_eq_atoms) == 0:
        st.error("Supercell equilibrium and TS must have the same number of atoms with coordinates.")
        st.stop()

    sc_lat = sc_eq_meta.get("lattice", None)
    if sc_lat is None:
        st.error("Supercell band.yaml missing lattice in metadata; cannot compute Cartesian displacements.")
        st.stop()

    eq_frac, eq_sym, eq_mass = atoms_to_arrays(sc_eq_atoms)
    ts_frac, ts_sym, ts_mass = atoms_to_arrays(sc_ts_atoms)

    if not np.all(eq_sym == ts_sym):
        st.warning("Atom symbols differ between equilibrium and TS; proceeding by index.")
    if not np.allclose(eq_mass, ts_mass):
        st.warning("Atom masses differ between equilibrium and TS; proceeding with equilibrium masses.")

    dfrac = ts_frac - eq_frac
    dfrac -= np.round(dfrac)  # minimum-image convention in fractional space
    dr_cart = frac_to_cart(dfrac, sc_lat)  # [N_sc, 3]

    uc_atoms = uc_meta.get("atoms", [])
    uc_lat = uc_meta.get("lattice", None)
    if uc_lat is None:
        st.error("Unit-cell band.yaml missing lattice in metadata.")
        st.stop()

    uc_frac, uc_sym, uc_mass = atoms_to_arrays(uc_atoms)

    mapping = infer_supercell_mapping(uc_frac, uc_sym, sc_eq_atoms, tol=1e-3)
    if mapping is None:
        st.error("Failed to infer supercell mapping to unit cell.")
        st.stop()

    if exclude_idx < 0:
        ex_idx = len(uc_atoms) - 1
    else:
        ex_idx = int(exclude_idx)
    settings = ProjectionSettings(
        exclude_atom_index=ex_idx,
        mass_weight=mass_weight,
        normalize=normalize
    )

    u_q = build_periodic_displacement_q(
        dr_cart, mapping, uc_lat, uc_ph["qpoints_frac"]
    )  # [nq, nat_uc, 3]

    A = project_onto_modes(
        u_q, uc_ph, uc_meta, uc_mass, settings
    )  # [nq, nbranches]

    fig = plot_bands_with_projection(uc_ph, uc_meta, A, amp_scale=amp_scale)
    st.pyplot(fig, clear_figure=True)
