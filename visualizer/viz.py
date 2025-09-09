
import numpy as np
import plotly.graph_objects as go

# Jmol color map and covalent radii for all elements (index 0 unused)
_ELEMENTS = [
    None,
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
    "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
    "Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
    "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]

_JMOL_COLORS = [
    None,
    "#FFFFFF","#D9FFFF","#CC80FF","#C2FF00","#FFB5B5","#909090","#3050F8","#FF0D0D","#90E050","#B3E3F5",
    "#AB5CF2","#8AFF00","#BFA6A6","#F0C8A0","#FF8000","#FFFF30","#1FF01F","#80D1E3","#8F40D4","#3DFF00",
    "#E6E6E6","#BFC2C7","#A6A6AB","#8A99C7","#9C7AC7","#E06633","#F090A0","#50D050","#C88033","#7D80B0",
    "#C28F8F","#668F8F","#BD80E3","#FFA100","#A62929","#5CB8D1","#702EB0","#00FF00","#94FFFF","#94E0E0",
    "#73C2C9","#54B5B5","#3B9E9E","#248F8F","#0A7D8C","#006985","#C0C0C0","#FFD98F","#A67573","#668080",
    "#9E63B5","#D47A00","#940094","#429EB0","#57178F","#00C900","#70D4FF","#FFFFC7","#D9FFC7","#C7FFC7",
    "#A3FFC7","#8FFFC7","#61FFC7","#45FFC7","#30FFC7","#1FFFC7","#00FF9C","#00E675","#00D452","#00BF38",
    "#00AB24","#4DC2FF","#4DA6FF","#2194D6","#267DAB","#266696","#175487","#D0D0E0","#FFD123","#B8B8D0",
    "#A6544D","#575961","#9E4FB5","#AB5C00","#754F45","#428296","#420066","#007D00","#70ABFA","#00BAFF",
    "#00A1FF","#008FFF","#0080FF","#006BFF","#545CF2","#785CE3","#8A4FE3","#A136D4","#B31FD4","#B31FBA",
    "#B30DA6","#BD0D87","#C70066","#CC0059","#D1004F","#D90045","#E00038","#E6002E","#EB0026","#FF0000",
    "#FF0000","#FF0000","#FF0000","#FF0000","#FF0000","#FF0000","#FF0000","#FF0000",
]

_COVALENT_RADII = [
    0.0,
    0.31,0.28,1.28,0.96,0.84,0.76,0.71,0.66,0.57,0.58,
    1.66,1.41,1.21,1.11,1.07,1.05,1.02,1.06,2.03,1.76,
    1.70,1.60,1.53,1.39,1.61,1.52,1.26,1.24,1.32,1.22,
    1.22,1.20,1.19,1.20,1.20,1.16,2.20,1.95,1.90,1.75,
    1.64,1.54,1.47,1.46,1.42,1.39,1.45,1.44,1.42,1.39,
    1.39,1.38,1.39,1.40,2.44,2.15,2.07,2.04,2.03,2.01,
    1.99,1.98,1.98,1.96,1.94,1.92,1.92,1.89,1.90,1.87,
    1.87,1.75,1.70,1.62,1.51,1.44,1.41,1.36,1.36,1.32,
    1.45,1.46,1.48,1.40,1.50,1.50,2.60,2.21,2.15,2.06,
    2.00,1.96,1.90,1.87,1.80,1.69,1.60,1.60,1.60,1.60,
    1.60,1.60,1.60,1.60,1.60,1.60,1.60,1.60,1.60,1.60,
    1.60,1.60,1.60,1.60,1.60,1.60,1.60,1.60,
]

JMOL_COLORS = {sym: col for sym, col in zip(_ELEMENTS[1:], _JMOL_COLORS[1:])}
ATOMIC_RADII = {sym: rad for sym, rad in zip(_ELEMENTS[1:], _COVALENT_RADII[1:])}

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
    # Precompute color and size for each atom
    colors = [JMOL_COLORS.get(s, "#808080") for s in species]
    base_sizes = [ATOMIC_RADII.get(s, 1.0) for s in species]
    size_scale = 20  # scale factor for marker size
    sizes = [r * size_scale for r in base_sizes]

    for t in range(steps):
        angle = 2*np.pi*(t/steps)
        disp_t = np.real(disp0 * np.exp(1j*angle)) * amp
        Rt = R + disp_t
        trace_atoms = go.Scatter3d(
            x=Rt[:,0], y=Rt[:,1], z=Rt[:,2], mode="markers",
            marker=dict(size=sizes, color=colors, sizemode="diameter"),
            name="atoms"
        )
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

    anim_opts = dict(
        frame={"duration": 50, "redraw": False},
        transition={"duration": 0},
        fromcurrent=True,
        mode="immediate",
        loop=True,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, anim_opts]},
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig
