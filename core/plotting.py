import numpy as np
import matplotlib.pyplot as plt
from .phonon_parser import build_kpath_ticks

def plot_bands_with_projection(uc_ph: dict, uc_meta: dict, A: np.ndarray, amp_scale: float = 800.0):
    distances = uc_ph["qpoints"]
    freqs = uc_ph["frequencies"]  # [nq, nbranches]
    labels = uc_meta.get("labels", {})
    xp, xl = build_kpath_ticks(distances, labels)

    nq, nb = freqs.shape
    sizes = (A / (A.max() + 1e-12)) * amp_scale

    fig, ax = plt.subplots(figsize=(9,6))
    for b in range(nb):
        ax.plot(distances, freqs[:, b], lw=1.0, alpha=0.7, zorder=1)

    for b in range(nb):
        ax.scatter(distances, freqs[:, b], s=sizes[:, b], alpha=0.6, zorder=2)

    if xp:
        ax.set_xticks(xp, xl)
        for x in xp:
            ax.axvline(x, color='k', lw=0.5, alpha=0.3)

    ax.set_xlabel("q-path")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("Unit-cell band structure with TS displacement projection")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig
