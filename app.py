import numpy as np
import streamlit as st

from core.loaders import load_band_yaml_from_bytes
from core.projection import (
    perp_vec_T_to_line_IF,
    trim_structures,
    compute_projection_modulus,
)
from core.structures import (
    read_qe_structure,
    build_supercell_atoms,
)
from core.supercell import (
    infer_cubic_repeats,
    expand_unit_cell_modes,
    map_ts_to_supercell,
    reorder_modes,
)
from core.plotting import plot_bands_with_projection
from core.phonopy_runner import (
    build_band_from_phonopy_files,
    DEFAULT_QPATH_TEXT,
)
from visualizer.viz import (
    make_phonon_band_figure,
    build_supercell,
    make_mode_animation_html,
)


st.set_page_config(page_title="Phonon Projection", layout="wide")
st.title("Phonon-mode projection of TS displacement onto unit-cell modes")

with st.sidebar:
    st.header("Unit cell phonons")
    source_mode = st.radio(
        "Provide phonons via",
        ["band.yaml", "phonopy inputs"],
        horizontal=True,
    )

    uc_file = None
    phonopy_yaml = None
    force_constants = None
    qpath_text = DEFAULT_QPATH_TEXT
    points_per_segment = 40

    if source_mode == "band.yaml":
        uc_file = st.file_uploader(
            "band.yaml (unit cell with eigenvectors)",
            type=["yaml", "yml"],
        )
    else:
        phonopy_yaml = st.file_uploader("phonopy.yaml", type=["yaml", "yml"])
        force_constants = st.file_uploader("FORCE_CONSTANTS")
        qpath_text = st.text_area(
            "Special q-points (LABEL qx qy qz)",
            value=DEFAULT_QPATH_TEXT,
            help="One point per line; adjacent points define a path segment.",
        )
        points_per_segment = int(
            st.number_input("Points per path segment", min_value=2, max_value=300, value=40, step=1)
        )

    st.header("Structure files (.in)")
    init_file = st.file_uploader("Initial structure (I)")
    ts_file = st.file_uploader("Transition state (TS)")
    final_file = st.file_uploader("Final structure (F)")

    st.header("Projection settings")
    exclude_idx = st.number_input(
        "Atom index to discard",
        min_value=-999,
        max_value=999,
        step=1,
        value=-1,
        help="Index removed from all structures before building dX",
    )
    mass_weight = st.checkbox("Mass-weight dX", value=False)
    center_dx = st.checkbox("Remove center-of-mass drift", value=True)
    amp_scale = st.slider("Marker size scale", 50.0, 2000.0, 400.0, step=10.0)
    map_tol = st.number_input("Mapping tolerance (fractional)", min_value=1e-5, max_value=5e-2, value=1e-3, format="%.1e")
    square_proj = st.checkbox("Plot |projection|^2", value=False)

    st.header("Supercell replication")
    sc_mode = st.radio("Choose repeats", ["Auto", "Manual"], horizontal=True)
    if sc_mode == "Manual":
        nx = st.number_input("nx", min_value=1, max_value=6, value=1, step=1)
        ny = st.number_input("ny", min_value=1, max_value=6, value=1, step=1)
        nz = st.number_input("nz", min_value=1, max_value=6, value=1, step=1)
        repeats = (int(nx), int(ny), int(nz))
    else:
        repeats = None

    run_btn = st.button("Compute projection")

st.caption(
    "Provide either a precomputed band.yaml or phonopy.yaml+FORCE_CONSTANTS to build the unit-cell modes, "
    "and three matching structures (.in) for the I/TS/F images. The TS displacement is taken as the perpendicular from the TS to the I→F line in 3N space."
)

uc_payload = None
if source_mode == "band.yaml":
    uc_bytes = uc_file.getvalue() if uc_file else None
    if uc_bytes:
        try:
            uc_payload = load_band_yaml_from_bytes(uc_bytes)
        except Exception as exc:
            st.error(f"Failed to parse unit-cell band.yaml: {exc}")
elif phonopy_yaml and force_constants:
    try:
        uc_payload = build_band_from_phonopy_files(
            phonopy_yaml,
            force_constants,
            qpath_text,
            points_per_segment,
        )
    except Exception as exc:
        st.error(f"Failed to build band structure from phonopy inputs: {exc}")

tab_proj, tab_viz = st.tabs(["Projection", "Visualizer"])


def _ensure_uc_payload():
    if uc_payload is None:
        if source_mode == "band.yaml":
            raise ValueError("Upload a unit-cell band.yaml first")
        raise ValueError("Upload phonopy.yaml and FORCE_CONSTANTS, then define a q-path")
    return uc_payload


with tab_proj:
    st.subheader("Projected displacement onto phonon modes")
    if run_btn:
        required_files = []
        if source_mode == "band.yaml":
            required_files.append((uc_file, "band.yaml"))
        else:
            required_files.append((phonopy_yaml, "phonopy.yaml"))
            required_files.append((force_constants, "FORCE_CONSTANTS"))
        required_files.extend([
            (init_file, "initial structure"),
            (ts_file, "transition-state structure"),
            (final_file, "final structure"),
        ])
        missing = [name for file_obj, name in required_files if not file_obj]
        if missing:
            st.error("Missing required inputs: " + ", ".join(missing))
        else:
            try:
                uc_ph, uc_meta = _ensure_uc_payload()
            except ValueError as exc:
                st.error(str(exc))
                uc_ph = uc_meta = None

            if uc_ph is not None and not uc_ph.get("has_eigenvectors", False):
                st.error("band.yaml must include eigenvectors to build projections")
                uc_ph = None

            structures = None
            if uc_ph is not None:
                try:
                    init_atoms = read_qe_structure(init_file)
                    ts_atoms = read_qe_structure(ts_file)
                    final_atoms = read_qe_structure(final_file)
                    structures = (init_atoms, ts_atoms, final_atoms)
                except Exception as exc:
                    st.error(f"Failed to read structures: {exc}")

            dx_vec = None
            trimmed = None
            if structures is not None:
                try:
                    trimmed = trim_structures(*structures, drop_index=int(exclude_idx))
                except Exception as exc:
                    st.error(f"Failed to discard atom index {exclude_idx}: {exc}")
                else:
                    try:
                        dx_vec = perp_vec_T_to_line_IF(trimmed[0], trimmed[2], trimmed[1])
                    except Exception as exc:
                        st.error(f"Failed to compute perpendicular displacement: {exc}")

            if dx_vec is not None:
                uc_atoms_meta = uc_meta.get("atoms", [])
                if not uc_atoms_meta:
                    st.error("band.yaml lacks atomic coordinates in the unit cell")
                else:
                    nat_uc = len(uc_atoms_meta)
                    nat_ts = len(trimmed[0])
                    try:
                        reps = repeats if repeats else infer_cubic_repeats(nat_ts, nat_uc)
                    except Exception as exc:
                        st.error(f"Failed to determine supercell repeats: {exc}")
                        reps = None

                    if reps is not None:
                        try:
                            supercell_atoms = build_supercell_atoms(uc_meta, reps)
                        except Exception as exc:
                            st.error(f"Failed to build supercell from unit cell: {exc}")
                            supercell_atoms = None

                    if reps is not None and supercell_atoms is not None:
                        if len(supercell_atoms) != nat_ts:
                            st.error("Supercell atom count does not match trimmed TS structure. Adjust repeats or atom index.")
                        else:
                            unit_eig = np.asarray(uc_ph["eigenvectors"], dtype=np.complex128)
                            qfrac = np.asarray(uc_ph["qpoints_frac"], dtype=float)
                            print(unit_eig.shape, qfrac.shape, reps)
                            try:
                                super_modes = expand_unit_cell_modes(unit_eig, qfrac, reps)
                            except Exception as exc:
                                st.error(f"Failed to build supercell eigenvectors: {exc}")
                                super_modes = None

                            if super_modes is not None:
                                try:
                                    mapping = map_ts_to_supercell(trimmed[1], supercell_atoms, tol=map_tol)
                                except Exception as exc:
                                    st.error(f"Mapping TS atoms to supercell failed: {exc}")
                                    super_modes = None

                            if super_modes is not None:
                                modes_ts_order = reorder_modes(super_modes, mapping)
                                masses = trimmed[1].get_masses()
                                projections = compute_projection_modulus(
                                    modes_ts_order,
                                    dx_vec,
                                    masses=masses,
                                    mass_weight=mass_weight,
                                    center=center_dx,
                                )
                                fig = plot_bands_with_projection(
                                    uc_ph,
                                    projections,
                                    amp_scale=amp_scale,
                                    square=square_proj,
                                )
                                st.pyplot(fig, clear_figure=True)
                                st.success(f"Projection complete for repeats {reps}.")
    else:
        st.info("Configure inputs in the sidebar and press **Compute projection** to run the analysis.")


with tab_viz:
    st.subheader("Interactive unit-cell visualizer")
    if uc_payload is None:
        if source_mode == "band.yaml":
            st.info("Upload a band.yaml with eigenvectors to enable the visualizer.")
        else:
            st.info("Upload phonopy.yaml + FORCE_CONSTANTS and define a q-path to enable the visualizer.")
    else:
        uc_ph, uc_meta = uc_payload
        col_band, col_anim = st.columns([1.2, 1.0], gap="large")
        with col_band:
            fig_band = make_phonon_band_figure(uc_ph, uc_meta)
            st.plotly_chart(fig_band, use_container_width=True)

        with col_anim:
            st.caption("Animate a selected phonon mode (mass-weighted displacements, rendered with NGL.js).")
            if not uc_ph.get("has_eigenvectors", False):
                st.warning("This band.yaml lacks eigenvectors. Re-run Phonopy with eigenvectors enabled to animate modes.")
            elif not uc_meta.get("atoms"):
                st.warning("Atom list missing in band.yaml; cannot build geometry for animation.")
            else:
                supercell_rep = st.slider("Supercell repetitions", 1, 3, 1, key="viz_supercell")
                amp = st.slider("Animation amplitude (arb. units)", 0.0, 10.0, 2.0, 0.1, key="viz_amp")
                frames = st.slider("Frames per period", 10, 60, 24, 2, key="viz_frames")
                fps = st.slider("Playback FPS", 5, 60, 20, 1, key="viz_fps")

                nq = len(uc_ph["qpoints"])
                nb = uc_ph["nbranches"]
                q_idx = st.slider("q-point index", 0, max(0, nq - 1), min(10, max(0, nq - 1)), key="viz_q")
                b_idx = st.slider("Branch index", 0, max(0, nb - 1), min(3, max(0, nb - 1)), key="viz_branch")

                R, species, lattice = build_supercell(
                    uc_meta,
                    reps=(supercell_rep, supercell_rep, supercell_rep)
                )
                base_masses = [a.get("mass", 1.0) or 1.0 for a in uc_meta["atoms"]]
                masses = base_masses * (supercell_rep ** 3)
                try:
                    html = make_mode_animation_html(
                        R=R,
                        species=species,
                        lattice=lattice,
                        eigenvectors=uc_ph["eigenvectors"][q_idx][b_idx],
                        qvec=uc_ph["qpoints_frac"][q_idx],
                        masses=masses,
                        amp=amp,
                        steps=frames,
                        fps=fps,
                    )
                    st.components.v1.html(html, height=540, scrolling=False)
                except Exception as exc:
                    st.error(f"Failed to animate mode: {exc}")
