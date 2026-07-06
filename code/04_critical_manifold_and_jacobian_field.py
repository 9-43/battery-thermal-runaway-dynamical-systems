import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path


# ============================================================
# 1. Basic parameters
# ============================================================
m = 0.720
Cp = 1100.0
Qcap = 25.0          # Ah
Rcell = 0.002
Rgas = 8.314
Tamb = 313.15        # K, 40 C
Iext = 5.0           # A, external current included in both heat and SOC
zeta = 0.28

# OCV polynomial
p = [-2.622, 5.761, -4.145, 2.062, 3.131]

# SEI reaction
c_sei = 0.15
m_sei = 100.58
H_sei = 150.0
A_sei = 1.667e15
Ea_sei = 135080.0
c_sei_0_ref = 0.15

# Anode reaction
c_an = 1.0
m_an = 100.58
H_an = 1714.0
A_an = 0.038
Ea_an = 33000.0

g_z_anode = np.exp(-c_sei / c_sei_0_ref)

# Temperature thresholds
T_alarm = 343.15     # 70 C
T_limit = 393.15     # 120 C

# Simulation settings
T_stop = T_limit     # stop integration once 120 C is reached
t_end = 20000.0      # max simulation time, s
S0 = 1.0
T0 = Tamb


# ============================================================
# 2. Output path
# ============================================================
output_dir = Path(r"D:\files\毕业设计\corrected_Iext_SOCincluded")
output_dir.mkdir(parents=True, exist_ok=True)

save_path = output_dir / "Tmax_sweep_Iext5A_SOCincluded_Tamb40C_NR70_NU70.npz"


# ============================================================
# 3. Model functions
# ============================================================
def U_OCV(S):
    S = np.clip(S, 0.0, 1.0)
    return np.polyval(p, S)


def dU_OCV_dS(S):
    S = np.clip(S, 0.0, 1.0)
    dp = np.polyder(p)
    return np.polyval(dp, S)


def Q_chem(T):
    q_sei = (
        c_sei * m_sei * H_sei * A_sei
        * np.exp(-Ea_sei / (Rgas * T))
    )

    q_an = (
        c_an * m_an * H_an * A_an
        * np.exp(-Ea_an / (Rgas * T))
        * g_z_anode
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
        * np.exp(-Ea_an / (Rgas * T))
        * g_z_anode
        * (Ea_an / (Rgas * T**2))
    )

    return dq_sei + dq_an


# Conservative voltage reference for epsilon
S_dense = np.linspace(0.0, 1.0, 3001)
U_dense = U_OCV(S_dense)

Uref = np.max(U_dense)
Umax2 = np.max(U_dense**2)


def epsilon_value(Risc, Ueq):
    """
    External-current-included time-scale ratio.

    epsilon = t_T / t_S

    t_T = m Cp / Ueq

    t_S = 3600 Qcap / I_ref

    I_ref = Uref / (Rcell + Risc) + Iext

    Therefore:
    epsilon = m Cp I_ref / (3600 Qcap Ueq)
    """
    I_ref = Uref / (Rcell + Risc) + Iext

    return (m * Cp * I_ref) / (
        3600.0 * Qcap * Ueq
    )


def Ueq_for_epsilon(eps0, Risc):
    """
    Constant-epsilon curve with external current included.

    Ueq = m Cp / (eps0 * 3600 Qcap)
          * [Uref / (Rcell + Risc) + Iext]
    """
    I_ref = Uref / (Rcell + Risc) + Iext

    return (m * Cp * I_ref) / (
        eps0 * 3600.0 * Qcap
    )


def rhs(t, y, Risc, Ueq):
    S, T = y
    U = U_OCV(S)

    I_isc = U / (Rcell + Risc)

    # External-current-included SOC equation
    # dS/dt = -(I_isc + Iext) / (3600 Qcap)
    dSdt = -(I_isc + Iext) / (3600.0 * Qcap)

    Q_isc = zeta * U**2 / (Rcell + Risc)

    # External current heat generation
    Q_ext = Iext**2 * Rcell

    dTdt = (
        Q_isc
        + Q_ext
        + Q_chem(T)
        - Ueq * (T - Tamb)
    ) / (m * Cp)

    return [dSdt, dTdt]


# ============================================================
# 4. Event functions
# ============================================================
def make_event_T_stop(Risc, Ueq):
    def event_T_stop(t, y):
        return y[1] - T_stop

    event_T_stop.terminal = True
    event_T_stop.direction = 1
    return event_T_stop


def make_event_SOC_empty(Risc, Ueq):
    def event_SOC_empty(t, y):
        return y[0]

    event_SOC_empty.terminal = True
    event_SOC_empty.direction = -1
    return event_SOC_empty


def simulate_one_point(Risc, Ueq):
    """
    Simulate one parameter point.

    Stop if:
    1. T reaches 120 C;
    2. SOC reaches 0;
    3. final time t_end is reached.

    Return:
        Tmax          : maximum simulated temperature, capped by T_stop if reached
        reached_stop  : whether T_stop is reached
        reached_empty : whether S=0 is reached
        final_time    : final integration time
        lambda2_Tmax  : lambda2 evaluated at Tmax
        margin        : Ueq - dQ_chem_dT(Tmax)
    """

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, Risc, Ueq),
        t_span=(0.0, t_end),
        y0=[S0, T0],
        method="DOP853",
        rtol=1e-6,
        atol=1e-8,
        max_step=20.0,
        events=[
            make_event_T_stop(Risc, Ueq),
            make_event_SOC_empty(Risc, Ueq)
        ],
    )

    reached_stop = len(sol.t_events[0]) > 0
    reached_empty = len(sol.t_events[1]) > 0

    if reached_stop:
        Tmax = T_stop
    else:
        Tmax = np.max(sol.y[1])

    lam2 = (dQ_chem_dT(Tmax) - Ueq) / (m * Cp)
    margin = Ueq - dQ_chem_dT(Tmax)

    return Tmax, reached_stop, reached_empty, sol.t[-1], lam2, margin


# ============================================================
# 5. Parameter grid
# ============================================================
N_R = 70
N_U = 70

Risc_array = np.logspace(-2, 1, N_R)       # 0.01 ~ 10 Ohm
Ueq_array = np.linspace(0.05, 10.0, N_U)   # 0.05 ~ 10 W/K

R_grid, U_grid = np.meshgrid(Risc_array, Ueq_array)

# External-current-included epsilon grid
eps_grid = epsilon_value(R_grid, U_grid)

Tmax_map = np.full_like(R_grid, np.nan, dtype=float)
lambda2_map = np.full_like(R_grid, np.nan, dtype=float)
margin_map = np.full_like(R_grid, np.nan, dtype=float)
reached_stop_map = np.zeros_like(R_grid, dtype=bool)
reached_empty_map = np.zeros_like(R_grid, dtype=bool)
final_time_map = np.full_like(R_grid, np.nan, dtype=float)


# ============================================================
# 6. Simulation sweep
# ============================================================
total = N_R * N_U
count = 0

for i, Ueq in enumerate(Ueq_array):
    for j, Risc in enumerate(Risc_array):
        count += 1

        Tmax, reached_stop, reached_empty, final_time, lam2, margin = simulate_one_point(
            Risc, Ueq
        )

        Tmax_map[i, j] = Tmax
        reached_stop_map[i, j] = reached_stop
        reached_empty_map[i, j] = reached_empty
        final_time_map[i, j] = final_time
        lambda2_map[i, j] = lam2
        margin_map[i, j] = margin

    print(f"Finished Ueq row {i + 1}/{N_U}")

print("Simulation sweep finished.")


# ============================================================
# 7. Save results
# ============================================================
np.savez(
    save_path,

    # Metadata
    model_tag="Iext5A_SOCincluded",
    Iext=Iext,
    Iext_in_thermal=True,
    Iext_in_SOC=True,
    epsilon_formula="epsilon = m Cp [Uref/(Rcell+Risc)+Iext] / (3600 Qcap Ueq)",

    # Parameters
    m=m,
    Cp=Cp,
    Qcap=Qcap,
    Rcell=Rcell,
    Rgas=Rgas,
    Tamb=Tamb,
    Tamb_C=Tamb - 273.15,
    S0=S0,
    T0=T0,
    T0_C=T0 - 273.15,
    zeta=zeta,
    Uref=Uref,
    Umax2=Umax2,
    T_alarm=T_alarm,
    T_limit=T_limit,
    T_stop=T_stop,
    t_end=t_end,

    # Grid
    Risc_array=Risc_array,
    Ueq_array=Ueq_array,
    R_grid=R_grid,
    U_grid=U_grid,
    eps_grid=eps_grid,

    # Results
    Tmax_map=Tmax_map,
    margin_map=margin_map,
    lambda2_map=lambda2_map,
    reached_stop_map=reached_stop_map,
    reached_empty_map=reached_empty_map,
    final_time_map=final_time_map
)

print("Saved to:", save_path)


# ============================================================
# 8. Figure 1:
#    3D Tmax surface colored by thermal stability margin
# ============================================================
X = np.log10(R_grid)
Y = U_grid
Z = Tmax_map - 273.15   # C

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# margin = Ueq - Qchem'(Tmax)
# margin > 0 means lambda2(Tmax) < 0
vmax = np.nanpercentile(np.abs(margin_map), 95)
norm = Normalize(vmin=-vmax, vmax=vmax)
facecolors = cm.coolwarm(norm(margin_map))

surf = ax.plot_surface(
    X, Y, Z,
    facecolors=facecolors,
    linewidth=0,
    antialiased=True,
    alpha=0.92
)

mappable = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
mappable.set_array(margin_map)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.10)
cbar.set_label(r"$U_{eq}-Q'_{chem}(T_{\max})$ [W/K]", fontsize=11)

# Add 120 C cutoff plane
ax.plot_surface(
    X, Y, np.full_like(Z, T_stop - 273.15),
    color="gray",
    alpha=0.12,
    linewidth=0
)

ax.set_xlabel(r"$\log_{10}(R_{isc}\,[\Omega])$", fontsize=11)
ax.set_ylabel(r"$U_{eq}$ [W/K]", fontsize=11)
ax.set_zlabel(r"$T_{\max}$ [$^\circ$C]", fontsize=11)

ax.set_title(
    r"Maximum Temperature and Thermal-Direction Stability Margin "
    r"$(I_{ext}=5\,\mathrm{A},\ \mathrm{SOC\ included})$",
    fontsize=13
)

legend_elements = [
    Patch(facecolor=cm.coolwarm(norm(0.5 * vmax)), label=r"$\lambda_2(T_{\max})<0$"),
    Patch(facecolor=cm.coolwarm(norm(-0.5 * vmax)), label=r"$\lambda_2(T_{\max})>0$"),
    Patch(facecolor="gray", alpha=0.3, label=r"$120^\circ$C cutoff plane"),
]

ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

plt.tight_layout()
plt.show()


# ============================================================
# 9. Figure 2:
#    Time-scale separation 3D surface with Tmax as z-axis
# ============================================================
eps_class = np.zeros_like(eps_grid, dtype=int)
eps_class[(eps_grid >= 0.1) & (eps_grid < 0.3)] = 1
eps_class[eps_grid >= 0.3] = 2

eps_colors = np.empty(eps_grid.shape + (4,), dtype=float)

color_strong = np.array([0.30, 0.70, 0.90, 0.90])  # blue
color_mid = np.array([0.60, 0.35, 0.90, 0.90])     # purple
color_large = np.array([0.93, 0.45, 0.55, 0.90])   # red-pink

eps_colors[eps_class == 0] = color_strong
eps_colors[eps_class == 1] = color_mid
eps_colors[eps_class == 2] = color_large

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    X, Y, Z,
    facecolors=eps_colors,
    linewidth=0,
    antialiased=True,
    alpha=0.92
)

# Interpolator for drawing epsilon boundary lines on the surface
interp_Tmax = RegularGridInterpolator(
    points=(Ueq_array, Risc_array),
    values=Z,
    bounds_error=False,
    fill_value=np.nan
)

R_line = np.logspace(
    np.log10(Risc_array.min()),
    np.log10(Risc_array.max()),
    800
)

for eps0, style, label in [
    (0.1, "-", r"$\epsilon=0.1$"),
    (0.3, "--", r"$\epsilon=0.3$")
]:
    U_line = Ueq_for_epsilon(eps0, R_line)

    valid = (
        (U_line >= Ueq_array.min()) &
        (U_line <= Ueq_array.max())
    )

    R_valid = R_line[valid]
    U_valid = U_line[valid]

    pts = np.column_stack([U_valid, R_valid])
    Z_valid = interp_Tmax(pts)

    valid2 = np.isfinite(Z_valid)

    ax.plot(
        np.log10(R_valid[valid2]),
        U_valid[valid2],
        Z_valid[valid2],
        color="black",
        linestyle=style,
        linewidth=2.4,
        label=label
    )

ax.set_xlabel(r"$\log_{10}(R_{isc}\,[\Omega])$", fontsize=11)
ax.set_ylabel(r"$U_{eq}$ [W/K]", fontsize=11)
ax.set_zlabel(r"$T_{\max}$ [$^\circ$C]", fontsize=11)

ax.set_title(
    r"Time-Scale Separation Regions with Maximum Temperature "
    r"$(I_{ext}=5\,\mathrm{A},\ \mathrm{SOC\ included})$",
    fontsize=13
)

legend_elements = [
    Patch(facecolor=color_strong, label=r"$\epsilon<0.1$"),
    Patch(facecolor=color_mid, label=r"$0.1\leq\epsilon<0.3$"),
    Patch(facecolor=color_large, label=r"$\epsilon\geq0.3$"),
    Line2D([0], [0], color="black", linestyle="-", linewidth=2.4, label=r"$\epsilon=0.1$"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=2.4, label=r"$\epsilon=0.3$"),
]

ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

plt.tight_layout()
plt.show()


# ============================================================
# 10. 2D contour figure
# ============================================================
plt.figure(figsize=(11, 7))

levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 30)

cf = plt.contourf(
    R_grid,
    U_grid,
    Z,
    levels=levels,
    cmap="viridis"
)

cbar = plt.colorbar(cf)
cbar.set_label(r"$T_{\max}$ [$^\circ$C]", fontsize=12)

# epsilon boundaries with external current included
cs = plt.contour(
    R_grid,
    U_grid,
    eps_grid,
    levels=[0.1, 0.3],
    colors=["white", "red"],
    linestyles=["-", "--"],
    linewidths=[2.0, 2.0]
)

plt.clabel(
    cs,
    fmt={0.1: r"$\epsilon=0.1$", 0.3: r"$\epsilon=0.3$"},
    fontsize=11
)

# lambda2(Tmax)=0 boundary
cs2 = plt.contour(
    R_grid,
    U_grid,
    margin_map,
    levels=[0.0],
    colors="black",
    linewidths=2.2
)

plt.clabel(
    cs2,
    fmt={0.0: r"$\lambda_2(T_{\max})=0$"},
    fontsize=11
)

plt.xscale("log")
plt.xlabel(r"Internal Short Resistance $R_{isc}$ [$\Omega$]", fontsize=13)
plt.ylabel(r"Cooling Coefficient $U_{eq}$ [W/K]", fontsize=13)

plt.title(
    r"Maximum Temperature with Time-Scale and Thermal Stability Boundaries "
    r"$(I_{ext}=5\,\mathrm{A},\ \mathrm{SOC\ included})$",
    fontsize=13
)

plt.grid(True, which="both", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()


# ============================================================
# 11. Summary statistics
# ============================================================
stable_thermal_direction = margin_map > 0
unstable_thermal_direction = margin_map <= 0

print("\nSummary:")
print(f"Model tag: Iext5A_SOCincluded")
print(f"Iext = {Iext:.2f} A")
print("SOC equation: dSdt = -(I_isc + Iext) / (3600 Qcap)")
print("Epsilon formula: epsilon = m Cp [Uref/(Rcell+Risc)+Iext] / (3600 Qcap Ueq)")
print(f"Saved npz: {save_path}")

print(f"\nGrid size: {N_U} x {N_R} = {N_U * N_R}")
print(f"Reached cutoff T_stop = {T_stop - 273.15:.1f} C: {np.sum(reached_stop_map)} points")
print(f"Reached SOC empty: {np.sum(reached_empty_map)} points")
print(f"lambda2(Tmax) < 0: {np.sum(stable_thermal_direction)} points")
print(f"lambda2(Tmax) >= 0: {np.sum(unstable_thermal_direction)} points")

print("\nTemperature range:")
print(f"min Tmax = {np.nanmin(Z):.2f} C")
print(f"max Tmax = {np.nanmax(Z):.2f} C")