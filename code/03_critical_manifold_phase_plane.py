import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D

# ==========================================
# 1. Baseline parameters
# ==========================================
m = 0.720
Cp = 1100.0
Qcap = 25.0
Rcell = 0.002
Rgas = 8.314
Tamb = 313.15
Iext = 5.0
zeta = 0.28

# OCV polynomial
# U_OCV(S) = 3.131 + 2.062 S - 4.145 S^2 + 5.761 S^3 - 2.622 S^4
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

S_grid_ref = np.linspace(0.0, 1.0, 1001)
Uref = np.max(U_OCV(S_grid_ref))

def epsilon_value(Risc, Ueq):
    Iref = Iext + Uref / (Rcell + Risc)
    return (m * Cp * Iref) / (3600.0 * Qcap * Ueq)

def Phi(T, S, Risc, Ueq):
    return (
        Iext**2 * Rcell
        + zeta * U_OCV(S)**2 / (Rcell + Risc)
        + Q_chem(T)
        - Ueq * (T - Tamb)
    )

def rhs(t, y, Risc, Ueq):
    S, T = y

    dSdt = -(1.0 / (3600.0 * Qcap)) * (
        Iext + U_OCV(S) / (Rcell + Risc)
    )

    dTdt = (1.0 / (m * Cp)) * Phi(T, S, Risc, Ueq)

    return [dSdt, dTdt]

def event_SOC_empty(t, y, Risc, Ueq):
    return y[0]

event_SOC_empty.terminal = True
event_SOC_empty.direction = -1

# ==========================================
# 3. Representative parameter pair
# ==========================================
Risc_star = 1.0
Ueq_star = 1.5

eps_star = epsilon_value(Risc_star, Ueq_star)
print("epsilon =", eps_star)

# ==========================================
# 4. Compute critical manifold C0
# ==========================================
S_vals = np.linspace(0.0, 1.0, 300)
T_roots = np.full_like(S_vals, np.nan, dtype=float)

T_min = Tamb
T_max = 500.0
T_scan = np.linspace(T_min, T_max, 1500)

for i, S in enumerate(S_vals):
    vals = Phi(T_scan, S, Risc_star, Ueq_star)

    for k in range(len(T_scan) - 1):
        if np.isnan(vals[k]) or np.isnan(vals[k + 1]):
            continue

        if vals[k] == 0:
            T_roots[i] = T_scan[k]
            break

        if vals[k] * vals[k + 1] < 0:
            T_roots[i] = brentq(
                lambda T: Phi(T, S, Risc_star, Ueq_star),
                T_scan[k], T_scan[k + 1]
            )
            break

mask = ~np.isnan(T_roots)
S_plot = S_vals[mask]
T_plot = T_roots[mask]

# ==========================================
# 5. Simulate trajectories approaching C0
# ==========================================
t_span = (0.0, 12000.0)
t_eval = np.linspace(t_span[0], t_span[1], 1600)

# Original initial condition + three additional initial conditions
initial_conditions = [
    [1.00, Tamb],
    [0.85, Tamb + 10.0],
    [0.65, Tamb - 5.0],
    [0.45, Tamb + 15.0],
]

trajectories = []

for y0 in initial_conditions:
    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        args=(Risc_star, Ueq_star),
        events=event_SOC_empty,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-8,
        atol=1e-10
    )
    trajectories.append(sol)

# ==========================================
# 6. Prepare plot limits
# ==========================================
y_min = min(
    np.nanmin(T_plot - 273.15),
    min(np.min(sol.y[1] - 273.15) for sol in trajectories)
) - 5

y_max = max(
    np.nanmax(T_plot - 273.15),
    max(np.max(sol.y[1] - 273.15) for sol in trajectories)
) + 5

x_min, x_max = 0.0, 1.0

# ==========================================
# 7. Vector field in the S-T phase plane
# ==========================================
S_field = np.linspace(0.05, 0.98, 18)
T_field_C = np.linspace(y_min, y_max, 18)

S_mesh, T_mesh_C = np.meshgrid(S_field, T_field_C)
T_mesh_K = T_mesh_C + 273.15

dS_field = -(1.0 / (3600.0 * Qcap)) * (
    Iext + U_OCV(S_mesh) / (Rcell + Risc_star)
)

dT_field = (1.0 / (m * Cp)) * Phi(
    T_mesh_K,
    S_mesh,
    Risc_star,
    Ueq_star
)

# Normalize arrows for clearer visualization in the plotted axes
x_range = x_max - x_min
y_range = y_max - y_min

u_screen = dS_field / x_range
v_screen = dT_field / y_range

speed_screen = np.sqrt(u_screen**2 + v_screen**2)
speed_screen[speed_screen == 0] = 1.0

arrow_len = 0.035

U_quiver = arrow_len * (u_screen / speed_screen) * x_range
V_quiver = arrow_len * (v_screen / speed_screen) * y_range

# ==========================================
# 8. Plot
# ==========================================
fig, ax = plt.subplots(figsize=(8.4, 5.8))

# Vector field
ax.quiver(
    S_mesh,
    T_mesh_C,
    U_quiver,
    V_quiver,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.0025,
    alpha=0.35,
    color="0.45",
    label="Vector field"
)

# Critical manifold
ax.plot(
    S_plot,
    T_plot - 273.15,
    color="black",
    linewidth=2.5,
    label=r"Critical manifold $C_0$"
)

# Trajectories
for i, sol in enumerate(trajectories):
    S_traj = sol.y[0]
    T_traj = sol.y[1] - 273.15

    ax.plot(
        S_traj,
        T_traj,
        linewidth=1.8,
        label="System trajectories" if i == 0 else None
    )

    # Initial point
    ax.scatter(
        S_traj[0],
        T_traj[0],
        s=45,
        marker="o",
        zorder=5,
        label="Initial conditions" if i == 0 else None
    )

    # Direction arrow along trajectory
    arrow_index = int(0.12 * len(S_traj))
    if arrow_index + 10 < len(S_traj):
        ax.annotate(
            "",
            xy=(S_traj[arrow_index + 10], T_traj[arrow_index + 10]),
            xytext=(S_traj[arrow_index], T_traj[arrow_index]),
            arrowprops=dict(arrowstyle="->", lw=1.5)
        )

# Axes and labels
ax.set_xlabel("SOC $S$", fontsize=13)
ax.set_ylabel("Temperature $T$ [$^\\circ$C]", fontsize=13)
ax.set_title("Trajectories Approaching the Critical Manifold", fontsize=14)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.grid(True, linestyle="--", alpha=0.3)

# Put parameter information into legend, not in the middle of the figure
param_handle = Line2D(
    [],
    [],
    linestyle="none",
    label=(
        rf"$R_{{isc}}={Risc_star:.2f}\,\Omega$, "
        rf"$U_{{eq}}={Ueq_star:.2f}\,\mathrm{{W/K}}$, "
        rf"$\epsilon={eps_star:.3f}$"
    )
)

handles, labels = ax.get_legend_handles_labels()
handles.append(param_handle)
labels.append(param_handle.get_label())

ax.legend(
    handles,
    labels,
    loc="best",
    fontsize=9,
    frameon=True,
    framealpha=0.95
)

fig.tight_layout()

plt.savefig(
    "critical_manifold_with_vector_field_and_trajectories.pdf",
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    "critical_manifold_with_vector_field_and_trajectories.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()