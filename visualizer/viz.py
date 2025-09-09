
import numpy as np
import plotly.graph_objects as go

def make_phonon_band_figure(phonon, meta):
    x = phonon['qpoints']
    y = phonon['frequencies']  # [nq, nb]
    nb = phonon['nbranches']

    fig = go.Figure()
    for b in range(nb):
        fig.add_trace(go.Scatter(x=x, y=y[:, b], mode="lines", name=f"Branch {b}"))
    # x-ticks
    labels = meta.get('labels', {}) if meta else {}
    xticks, xlabels = [], []
    for i, d in enumerate(x):
        if i in labels:
            lab = str(labels[i]).replace("GAMMA", "Γ").replace("\\Gamma", "Γ").replace("$\\Gamma$", "Γ").replace("$\Gamma$", "Γ")
            xticks.append(d); xlabels.append(lab)
    if xticks:
        fig.update_xaxes(tickmode="array", tickvals=xticks, ticktext=xlabels, title_text="k-path")
    else:
        fig.update_xaxes(title_text="k-path distance")
    fig.update_yaxes(title_text="Frequency (THz)")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    return fig

def build_supercell(meta, reps=(1,1,1)):
    lattice = meta['lattice']
    base = np.array([a['frac'] for a in meta['atoms']], dtype=float)
    species = [a['symbol'] for a in meta['atoms']]
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
    R = all_frac @ lattice  # [N,3]
    return R, all_species, lattice

def _unit_cell_box(lattice):
    a, b, c = lattice
    origin = np.zeros(3)
    corners = np.array([origin, a, b, c, a+b, a+c, b+c, a+b+c])
    edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
    xe, ye, ze = [], [], []
    for i,j in edges:
        xe += [corners[i,0], corners[j,0], None]
        ye += [corners[i,1], corners[j,1], None]
        ze += [corners[i,2], corners[j,2], None]
    return go.Scatter3d(x=xe, y=ye, z=ze, mode="lines", name="cell")

def make_mode_animation_figure(R, species, lattice, eigenvectors, qvec, amp=0.2, steps=24, show_vectors=False):
    N = len(R)
    nat = len(eigenvectors)
    ev = np.array(eigenvectors, dtype=complex)  # [nat,3]
    reps = int(round(N / nat)) if nat>0 else 1
    ev_sc = np.vstack([ev for _ in range(reps)])  # [N,3]

    a, b, c = lattice
    M = np.vstack([a,b,c]).T  # columns
    finv = np.linalg.inv(M)
    r_frac = (R @ finv.T)
    phase = np.exp(2j*np.pi*(r_frac @ np.array(qvec)))

    disp0 = ev_sc * phase[:,None]
    norms = np.linalg.norm(np.real(disp0), axis=1) + 1e-12
    arrow_dir = (np.real(disp0) / norms[:,None])

    frames_list = []
    for t in range(steps):
        angle = 2*np.pi*(t/steps)
        disp_t = np.real(disp0 * np.exp(1j*angle)) * amp
        Rt = R + disp_t
        trace_atoms = go.Scatter3d(x=Rt[:,0], y=Rt[:,1], z=Rt[:,2], mode="markers", marker=dict(size=4), name="atoms")
        data = [trace_atoms, _unit_cell_box(lattice)]
        if show_vectors:
            xe, ye, ze = [], [], []
            scale = 0.4 * amp
            for i in range(N):
                start = Rt[i]; end = Rt[i] + scale * arrow_dir[i]
                xe += [start[0], end[0], None]
                ye += [start[1], end[1], None]
                ze += [start[2], end[2], None]
            data.append(go.Scatter3d(x=xe, y=ye, z=ze, mode="lines", name="vectors"))
        frames_list.append(go.Frame(data=data, name=f"f{t}"))

    fig = go.Figure(data=frames_list[0].data, frames=frames_list)
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
        updatemenus=[{"type": "buttons", "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]}
        ]}],
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False
    )
    return fig
