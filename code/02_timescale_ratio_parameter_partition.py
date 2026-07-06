import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ==========================================
# 1. Baseline parameters
# ==========================================
m = 0.720
Cp = 1100.0
Qcap = 25.0
Rcell = 0.0046
Rgas = 8.314
Tamb = 318.15
Iext = 5.0
zeta = 0.28

# OCV polynomial
p = [-2.622, 5.761, -4.145, 2.062, 3.131]

# Chemical reaction parameters
c_sei = 0.15
m_sei = 100.58
H_sei = 150.0
A_sei = 1.667e15
Ea_sei = 135080.0
c_sei_0_ref = 0.15

c_an = 1.0
m_an = 100.58
H_an = 1714.0
A_an = 0.038
Ea_an = 33000.0

g_z_anode = np.exp(-c_sei / c_sei_0_ref)

# ==========================================
# 2. Model functions
# ==========================================
def U_OCV(S):
    S = np.clip(S, 0.0, 1.0)
    return np.polyval(p, S)

def Q_chem(T):
    q_sei = (
        c_sei * m_sei * H_sei * A_sei
        * np.exp(-Ea_sei / (Rgas * T))
    )
    q_an = (
        c_an * m_an * H_an * A_an
        * g_z_anode
        * np.exp(-Ea_an / (Rgas * T))
    )
    return q_sei + q_an

def dQ_chem_dT(T):
    dq_sei = (
        c_sei * m_sei * H_sei * A_sei
        * np.exp(-Ea_sei / (Rgas * T))
        * (Ea_sei / (Rgas * T**2))
    )
    dq_an = (
        c_an * m_an * H_an * A_an
        * g_z_anode
        * np.exp(-Ea_an / (Rgas * T))
        * (Ea_an / (Rgas * T**2))
    )
    return dq_sei + dq_an

def epsilon_value(Risc, Ueq):
    S_grid = np.linspace(0.0, 1.0, 1001)
    Uref = np.max(U_OCV(S_grid))
    Iref = Iext + Uref / (Rcell + Risc)
    return (m * Cp * Iref) / (3600.0 * Qcap * Ueq)

# ==========================================
# 3. Parameter space
# ==========================================
Risc_array = np.logspace(-2, 1, 320)      # 0.01 to 10 Ohm
Ueq_array = np.linspace(0.05, 10.0, 320)  # 0.05 to 10 W/K

R_grid, U_grid = np.meshgrid(Risc_array, Ueq_array)
eps_grid = epsilon_value(R_grid, U_grid)

# Region labels:
# 0 -> epsilon < 0.1
# 1 -> 0.1 <= epsilon < 0.3
# 2 -> epsilon >= 0.3
region_grid = np.zeros_like(eps_grid)
region_grid[(eps_grid >= 0.1) & (eps_grid < 0.8)] = 1
region_grid[eps_grid >= 0.8] = 2

# ==========================================
# 4. Fold temperature Tf from dQ_chem/dT = Ueq
# ==========================================
def find_T_fold_for_Ueq(Ueq, T_min=Tamb + 1e-6, T_max=500.0):
    def residual(T):
        return dQ_chem_dT(T) - Ueq

    T_scan = np.linspace(T_min, T_max, 1200)
    vals = np.array([residual(T) for T in T_scan])

    for i in range(len(T_scan) - 1):
        if np.isnan(vals[i]) or np.isnan(vals[i + 1]):
            continue
        if vals[i] == 0:
            return T_scan[i]
        if vals[i] * vals[i + 1] < 0:
            return brentq(residual, T_scan[i], T_scan[i + 1])

    return np.nan

T_fold_array = np.array([
    find_T_fold_for_Ueq(Ueq, T_max=500.0)
    for Ueq in Ueq_array
])

# ==========================================
# 5. Physical fold existence condition
# ==========================================
Qext = Iext**2 * Rcell

Qchem_fold_array = np.array([
    Q_chem(Tf) if not np.isnan(Tf) else np.nan
    for Tf in T_fold_array
])

A_array = Ueq_array * (T_fold_array - Tamb) - Qchem_fold_array - Qext

S_dense = np.linspace(0.0, 1.0, 3000)
U2_values = U_OCV(S_dense)**2
U2_min = np.min(U2_values)
U2_max = np.max(U2_values)

Qisc_min_array = zeta * U2_min / (Rcell + Risc_array)
Qisc_max_array = zeta * U2_max / (Rcell + Risc_array)

A_matrix = A_array[:, None]
Qmin_matrix = Qisc_min_array[None, :]
Qmax_matrix = Qisc_max_array[None, :]

physical_fold_region = (
    ~np.isnan(A_matrix)
    & (A_matrix >= Qmin_matrix)
    & (A_matrix <= Qmax_matrix)
)

# ==========================================
# 6. Plot
# ==========================================
plt.figure(figsize=(9.4, 6.6))

# Background regions
levels = [-0.5, 0.5, 1.5, 2.5]
colors = ["#bfe6c8", "#fff2a8", "#f4a6a6"]  # green, yellow, red

plt.contourf(
    R_grid, U_grid, region_grid,
    levels=levels,
    colors=colors,
    alpha=0.90
)

# Boundary lines
c1 = plt.contour(
    R_grid, U_grid, eps_grid,
    levels=[0.1],
    colors="black",
    linewidths=2.0
)

c2 = plt.contour(
    R_grid, U_grid, eps_grid,
    levels=[0.8],
    colors="red",
    linewidths=2.0,
    linestyles="--"
)

# Fold region overlay: blue boundary + hatch
plt.contour(
    R_grid, U_grid, physical_fold_region.astype(float),
    levels=[0.5],
    colors="blue",
    linewidths=2.0
)

plt.contourf(
    R_grid, U_grid, physical_fold_region.astype(float),
    levels=[0.5, 1.5],
    colors="none",
    hatches=["///"],
    alpha=0
)

# Optional curve labels
plt.clabel(c1, fmt={0.1: r"$\epsilon=0.1$"}, fontsize=10, inline=True)
plt.clabel(c2, fmt={0.3: r"$\epsilon=0.3$"}, fontsize=10, inline=True)

# Axes
plt.xscale("log")
plt.xlim(1e-2, 1e1)
plt.ylim(0.05, 10.0)

plt.xlabel(r"Internal Short Circuit Resistance $R_{isc}$ [$\Omega$]", fontsize=13)
plt.ylabel(r"Equivalent Heat Dissipation Coefficient $U_{eq}$ [W/K]", fontsize=13)
plt.title("Time Scale Regions and Physical Fold Existence Region", fontsize=14)

plt.grid(True, which="major", linestyle="--", alpha=0.28)
plt.grid(True, which="minor", linestyle="--", alpha=0.15)

# Legend
legend_elements = [
    Patch(facecolor="#bfe6c8", edgecolor="none", label=r"Strong fast slow region ($\epsilon<0.1$)"),
    Patch(facecolor="#fff2a8", edgecolor="none", label=r"Weak fast slow region ($0.1\leq\epsilon<0.3$)"),
    Patch(facecolor="#f4a6a6", edgecolor="none", label=r"Non fast slow region ($\epsilon\geq0.3$)"),
    Line2D([0], [0], color="black", lw=2, label=r"Boundary $\epsilon=0.1$"),
    Line2D([0], [0], color="red", lw=2, linestyle="--", label=r"Boundary $\epsilon=0.3$"),
    Patch(facecolor="white", edgecolor="blue", hatch="///", label="Physical fold existence region")
]

plt.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=10,
    frameon=True,
    framealpha=0.95
)

plt.tight_layout()
plt.savefig("epsilon_partition_with_fold.pdf", dpi=300, bbox_inches="tight")
plt.savefig("epsilon_partition_with_fold.png", dpi=300, bbox_inches="tight")
plt.show()