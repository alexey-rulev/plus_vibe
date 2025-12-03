import numpy as np
import matplotlib.pyplot as plt

THZ_TO_MEV = 4.13567
MEV_TO_CM1 = 8.06554


def _auto_kpath_ticks(qpoints_frac: list[list[float]]):
    ticks = set()
    for i in range(len(qpoints_frac) - 1):
        if np.allclose(qpoints_frac[i], qpoints_frac[i + 1]):
            ticks.add(i)
    ticks.update({0, len(qpoints_frac) - 1})
    ticks = sorted(ticks)
    labels = [f"({qpoints_frac[i][0]:.1f},{qpoints_frac[i][1]:.1f},{qpoints_frac[i][2]:.1f})" for i in ticks]
    return ticks, labels


def _sizes_from_projections(proj: np.ndarray, min_size: float = 0.0, max_size: float = 200.0, power: float = 1.0) -> np.ndarray:
    proj = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-12)
    return min_size + (proj ** power) * (max_size - min_size)


def plot_bands_with_projection(uc_ph: dict, projections: np.ndarray, amp_scale: float = 200.0, square: bool = False):
    freqs_thz = np.asarray(uc_ph["frequencies"], dtype=float)
    freqs_mev = freqs_thz * THZ_TO_MEV
    print(uc_ph["qpoints"])
    q = np.asarray(uc_ph["qpoints"], dtype=float)
    print(q)
    qfrac = uc_ph.get("qpoints_frac", [])

    proj_vals = np.asarray(projections, dtype=float)
    power = 2.0 if square else 1.0

    sizes = _sizes_from_projections(proj_vals, max_size=amp_scale, power=power)
    fig, ax = plt.subplots(figsize=(6, 4))
    for b in range(freqs_mev.shape[1]):
        ax.plot(q, freqs_mev[:, b], color="black", lw=1.0)
        ax.scatter(q, freqs_mev[:, b], s=sizes[:, b], color="#2ECC71", alpha=1, edgecolors='none')

    if qfrac:
        ticks, labels = _auto_kpath_ticks(qfrac)
        ax.set_xticks(q[ticks], labels, rotation=45)

    ax.set_xlabel("q-point")
    ax.set_ylabel("Energy, meV")
    ax.set_xlim(q.min(), q.max())

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] * MEV_TO_CM1, ax.get_ylim()[1] * MEV_TO_CM1)
    ax2.set_ylabel("Wavenumber, cm$^{-1}$")

    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
