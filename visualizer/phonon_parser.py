# phonon_parser.py
import numpy as np

def _complex_from_pair(x):
    """Convert [real, imag] to complex."""
    return complex(float(x[0]), float(x[1]))

def _parse_lattice(lat):
    """
    Accepts either:
      - dict: {'a':[...], 'b':[...], 'c':[...]}
      - list/tuple: [[...],[...],[...]]
    Returns a (3,3) numpy array or None if not available/invalid.
    """
    if lat is None:
        return None
    # dict-style
    if isinstance(lat, dict):
        if all(k in lat for k in ("a", "b", "c")):
            return np.array(
                [[lat["a"][0], lat["a"][1], lat["a"][2]],
                 [lat["b"][0], lat["b"][1], lat["b"][2]],
                 [lat["c"][0], lat["c"][1], lat["c"][2]]],
                dtype=float
            )
        # forgiving case-insensitive keys
        keys = {k.lower(): k for k in lat.keys()}
        if all(k in keys for k in ("a","b","c")):
            return np.array(
                [[lat[keys["a"]][0], lat[keys["a"]][1], lat[keys["a"]][2]],
                 [lat[keys["b"]][0], lat[keys["b"]][1], lat[keys["b"]][2]],
                 [lat[keys["c"]][0], lat[keys["c"]][1], lat[keys["c"]][2]]],
                dtype=float
            )
        return None
    # list/tuple-style
    if isinstance(lat, (list, tuple)) and len(lat) == 3:
        try:
            return np.array(
                [[float(lat[0][0]), float(lat[0][1]), float(lat[0][2])],
                 [float(lat[1][0]), float(lat[1][1]), float(lat[1][2])],
                 [float(lat[2][0]), float(lat[2][1]), float(lat[2][2])]],
                dtype=float
            )
        except Exception:
            return None
    return None

def parse_band_yaml(data):
    """
    Parse Phonopy band.yaml into a structured dict for plotting & animation.
    Returns (phonon, meta) where:
      phonon: {
        'qpoints': list of distances (cumulative abscissa),
        'qpoints_frac': list of fractional q (in recip. lattice units),
        'frequencies': array [nq, nbranches],
        'eigenvectors': nested list [nq][nbranches][natoms][3] (complex) if present,
        'has_eigenvectors': bool,
        'nbranches': int
      }
      meta: {
        'lattice': 3x3 (Å) or None,
        'atoms': list of dicts with 'symbol', 'frac', 'mass',
        'labels': dict of q-index -> label string (Γ, X, ...),
      }
    """
    lattice = _parse_lattice(data.get("lattice"))

    # atoms may be under 'points' (usual in band.yaml)
    atoms = []
    if "points" in data and isinstance(data["points"], list):
        for p in data["points"]:
            frac = np.array(p.get("coordinates", [0, 0, 0]), dtype=float)
            atoms.append({
                "symbol": p.get("symbol", "X"),
                "frac": frac,
                "mass": float(p.get("mass", 0.0)),
            })

    qpoints_frac = []
    distances = []
    freqs_list = []
    eigs_all = []
    labels = {}
    has_eigs = False

    phonon_list = data.get("phonon", [])
    if not isinstance(phonon_list, list) or not phonon_list:
        raise ValueError("band.yaml does not contain a valid 'phonon' list.")

    nbranches = None
    for iq, q in enumerate(phonon_list):
        # q-point
        qp = q.get("q-position", q.get("q-position-frac"))
        if qp is None:
            raise ValueError(f"Missing 'q-position' at phonon entry {iq}.")
        qpoints_frac.append(np.array(qp, dtype=float))

        # cumulative distance; if missing, use sequential index
        distances.append(float(q.get("distance", iq)))

        # bands/frequencies
        bands = q.get("band")
        if not bands:
            raise ValueError(f"Missing 'band' data at phonon entry {iq}.")
        if nbranches is None:
            nbranches = len(bands)
        freqs = [float(b.get("frequency", 0.0)) for b in bands]
        freqs_list.append(freqs)

        # eigenvectors (optional)
        if "eigenvector" in bands[0]:
            has_eigs = True
            eig_per_q = []
            for b in bands:
                vecs = []
                for at in b["eigenvector"]:
                    v = [_complex_from_pair(comp) for comp in at]  # [3] complex
                    vecs.append(v)
                eig_per_q.append(vecs)
            eigs_all.append(eig_per_q)

        # labels at special points
        if "label" in q:
            labels[iq] = q["label"]

    freqs_arr = np.array(freqs_list)  # [nq, nbranches]

    # main payload
    out_ph = {
        "qpoints": distances,
        "qpoints_frac": qpoints_frac,
        "frequencies": freqs_arr,
        "nbranches": int(freqs_arr.shape[1]),
        "has_eigenvectors": has_eigs,
    }
    # backward-compat alias for older plotters
    out_ph["distances"] = out_ph["qpoints"]

    if has_eigs:
        out_ph["eigenvectors"] = eigs_all

    meta = {
        "lattice": lattice,
        "atoms": atoms,    # can be empty if not present in YAML; animation needs it + lattice
        "labels": labels,
    }
    return out_ph, meta

def build_kpath_ticks(distances, labels):
    """Build tick positions and labels for special points."""
    xp, xl = [], []
    for i, d in enumerate(distances):
        if i in labels:
            lab = str(labels[i])
            lab = (lab.replace("GAMMA", "Γ")
                      .replace("\\Gamma", "Γ")
                      .replace("$\\Gamma$", "Γ")
                      .replace("$\\Gamma$", "Γ"))
            xp.append(d); xl.append(lab)
    if distances:
        if 0 not in labels:
            xp = [distances[0]] + xp
            xl = ([""] if not xl else [xl[0]]) + xl
        if (len(distances) - 1) not in labels:
            xp = xp + [distances[-1]]
            xl = xl + [""]
    return xp, xl
