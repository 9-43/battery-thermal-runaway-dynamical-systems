import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 0. Global switches
# ============================================================
FORCE_RERUN = False
# False: if compatible npz already exists, load it directly.
# True : always rerun simulation for 20 C and 60 C.

INITIAL_T_EQUALS_AMBIENT = True
# True : T0 = Tamb for each ambient temperature case.
# False: T0 is fixed as FIXED_T0_C for all ambient temperatures.

FIXED_T0_C = 40.0


# ============================================================
# 1. Basic parameters
# ============================================================
m = 0.720
Cp = 1100.0
Qcap = 25.0          # Ah
Rcell = 0.002
Rgas = 8.314
Iext = 5.0           # A
zeta = 0.28

# External current is included in SOC evolution.
# This must be True to match the final thesis model.
include_Iext_in_SOC = True

# Ambient temperature cases
Tamb_cases_C = [20.0, 40.0, 60.0]
Tamb_cases_K = [T + 273.15 for T in Tamb_cases_C]

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
T_stop = T_limit     # stop integration once 120 C is reached

# Simulation settings
t_end = 20000.0      # max simulation time, s
S0 = 1.0


# ============================================================
# 2. Output settings
# ============================================================
output_dir = Path(r"D:\files\毕业设计\Tamb_sweep_results_Iext5A_SOCincluded")
output_dir.mkdir(parents=True, exist_ok=True)

fig_dir = output_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Existing corrected 40 C result
fixed_40C_path = Path(
    r"D:\files\毕业设计\corrected_Iext_SOCincluded\Tmax_sweep_Iext5A_SOCincluded_Tamb40C_NR70_NU70.npz"
)


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
    Final epsilon formula:

    epsilon(Risc, Ueq)
    =
    m Cp / (3600 Qcap Ueq)
    *
    [ Iext + Uref / (Rcell + Risc) ]
    """
    return (m * Cp / (3600.0 * Qcap * Ueq)) * (
        Iext + Uref / (Rcell + Risc)
    )


def Ueq_for_epsilon(eps0, Risc):
    """
    Inverse of the final epsilon formula:

    Ueq =
    m Cp / (3600 Qcap eps0)
    *
    [ Iext + Uref / (Rcell + Risc) ]
    """
    return (m * Cp / (3600.0 * Qcap * eps0)) * (
        Iext + Uref / (Rcell + Risc)
    )


def rhs(t, y, Risc, Ueq, Tamb):
    S, T = y
    U = U_OCV(S)

    I_isc = U / (Rcell + Risc)

    # SOC includes both internal short current and external current
    if include_Iext_in_SOC:
        dSdt = -(I_isc + Iext) / (3600.0 * Qcap)
    else:
        dSdt = -I_isc / (3600.0 * Qcap)

    Q_isc = zeta * U**2 / (Rcell + Risc)
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
def make_event_T_stop():
    def event_T_stop(t, y):
        return y[1] - T_stop

    event_T_stop.terminal = True
    event_T_stop.direction = 1
    return event_T_stop


def make_event_SOC_empty():
    def event_SOC_empty(t, y):
        return y[0]

    event_SOC_empty.terminal = True
    event_SOC_empty.direction = -1
    return event_SOC_empty


def get_initial_temperature(Tamb):
    if INITIAL_T_EQUALS_AMBIENT:
        return Tamb
    else:
        return FIXED_T0_C + 273.15


def simulate_one_point(Risc, Ueq, Tamb):
    """
    Simulate one parameter point under a given ambient temperature.

    Stop if:
    1. T reaches 120 C;
    2. SOC reaches 0;
    3. final time t_end is reached.
    """

    T0 = get_initial_temperature(Tamb)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, Risc, Ueq, Tamb),
        t_span=(0.0, t_end),
        y0=[S0, T0],
        method="DOP853",
        rtol=1e-6,
        atol=1e-8,
        max_step=20.0,
        events=[
            make_event_T_stop(),
            make_event_SOC_empty()
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
eps_grid = epsilon_value(R_grid, U_grid)


# ============================================================
# 6. Save/load simulation results
# ============================================================
def Iext_tag():
    return str(Iext).replace(".", "p")


def npz_path_for_Tamb(Tamb_C):
    if int(Tamb_C) == 40:
        return fixed_40C_path

    return output_dir / (
        f"Tmax_sweep_Iext{Iext_tag()}A_SOCincluded_"
        f"Tamb{int(Tamb_C)}C_NR{N_R}_NU{N_U}.npz"
    )


def load_npz_to_dict(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def grid_is_compatible(data):
    """
    Only check grid and required result fields.
    This is used for the fixed 40 C file because it may not store all metadata.
    """
    required_keys = [
        "Risc_array",
        "Ueq_array",
        "R_grid",
        "U_grid",
        "Tmax_map",
        "reached_stop_map",
        "final_time_map",
    ]

    for key in required_keys:
        if key not in data:
            print(f"Missing key in npz file: {key}")
            return False

    if data["Risc_array"].shape != Risc_array.shape:
        return False

    if data["Ueq_array"].shape != Ueq_array.shape:
        return False

    if not np.allclose(data["Risc_array"], Risc_array):
        return False

    if not np.allclose(data["Ueq_array"], Ueq_array):
        return False

    if data["R_grid"].shape != R_grid.shape:
        return False

    if data["U_grid"].shape != U_grid.shape:
        return False

    return True


def result_is_compatible(data, Tamb_C, Tamb_K):
    """
    Compatibility check for newly generated 20 C and 60 C files.
    """
    if not grid_is_compatible(data):
        return False

    if "Iext" not in data:
        return False

    if "include_Iext_in_SOC" not in data:
        return False

    if "Tamb_C" not in data:
        return False

    if "Tamb_K" not in data:
        return False

    if "eps_grid" not in data:
        return False

    if not np.isclose(float(data["Iext"]), Iext):
        return False

    if bool(data["include_Iext_in_SOC"]) != include_Iext_in_SOC:
        return False

    if not np.isclose(float(data["Tamb_C"]), Tamb_C):
        return False

    if not np.isclose(float(data["Tamb_K"]), Tamb_K):
        return False

    eps_check = epsilon_value(R_grid, U_grid)

    if not np.allclose(data["eps_grid"], eps_check, rtol=1e-10, atol=1e-12):
        return False

    return True


def run_sweep_for_Tamb(Tamb_C, Tamb):
    print("\n" + "=" * 70)
    print(f"Running simulation: Tamb = {Tamb_C:.0f} C, Iext = {Iext:.2f} A")
    print("=" * 70)

    Tmax_map = np.full_like(R_grid, np.nan, dtype=float)
    lambda2_map = np.full_like(R_grid, np.nan, dtype=float)
    margin_map = np.full_like(R_grid, np.nan, dtype=float)
    reached_stop_map = np.zeros_like(R_grid, dtype=bool)
    reached_empty_map = np.zeros_like(R_grid, dtype=bool)
    final_time_map = np.full_like(R_grid, np.nan, dtype=float)

    for i, Ueq in enumerate(Ueq_array):
        for j, Risc in enumerate(Risc_array):

            Tmax, reached_stop, reached_empty, final_time, lam2, margin = simulate_one_point(
                Risc, Ueq, Tamb
            )

            Tmax_map[i, j] = Tmax
            reached_stop_map[i, j] = reached_stop
            reached_empty_map[i, j] = reached_empty
            final_time_map[i, j] = final_time
            lambda2_map[i, j] = lam2
            margin_map[i, j] = margin

        print(f"Tamb = {Tamb_C:.0f} C: finished Ueq row {i + 1}/{N_U}")

    save_path = npz_path_for_Tamb(Tamb_C)

    np.savez(
        save_path,
        Tamb_C=Tamb_C,
        Tamb_K=Tamb,
        Iext=Iext,
        include_Iext_in_SOC=include_Iext_in_SOC,
        INITIAL_T_EQUALS_AMBIENT=INITIAL_T_EQUALS_AMBIENT,
        FIXED_T0_C=FIXED_T0_C,
        S0=S0,
        T0_K=get_initial_temperature(Tamb),
        T0_C=get_initial_temperature(Tamb) - 273.15,
        N_R=N_R,
        N_U=N_U,
        Risc_array=Risc_array,
        Ueq_array=Ueq_array,
        R_grid=R_grid,
        U_grid=U_grid,
        eps_grid=eps_grid,
        Tmax_map=Tmax_map,
        margin_map=margin_map,
        lambda2_map=lambda2_map,
        reached_stop_map=reached_stop_map,
        reached_empty_map=reached_empty_map,
        final_time_map=final_time_map
    )

    print(f"Saved simulation result to: {save_path}")

    return load_npz_to_dict(save_path)


def get_result_for_Tamb(Tamb_C, Tamb):
    save_path = npz_path_for_Tamb(Tamb_C)

    # 40 C: always load the fixed corrected result
    if int(Tamb_C) == 40:
        if not fixed_40C_path.exists():
            raise FileNotFoundError(f"Fixed 40 C result file not found: {fixed_40C_path}")

        data = load_npz_to_dict(fixed_40C_path)

        if not grid_is_compatible(data):
            raise ValueError(
                "The fixed 40 C file exists, but its grid is not compatible "
                "with the current Risc_array and Ueq_array."
            )

        print(f"Loaded fixed corrected 40 C result: {fixed_40C_path}")
        return data

    # 20 C and 60 C: load existing compatible file or rerun
    if (not FORCE_RERUN) and save_path.exists():
        data = load_npz_to_dict(save_path)

        if result_is_compatible(data, Tamb_C, Tamb):
            print(f"Loaded existing compatible result: {save_path}")
            return data
        else:
            print(f"Existing file is not compatible, rerunning: {save_path}")

    return run_sweep_for_Tamb(Tamb_C, Tamb)


# ============================================================
# 7. Boundary extraction and epsilon statistics
# ============================================================
def has_level(field, level):
    return np.nanmin(field) <= level <= np.nanmax(field)


def extract_contour_segments(ax, X, Y, field, level, color, linestyle, linewidth):
    """
    Plot one contour and return its segments.
    """
    if not has_level(field, level):
        print(
            f"No contour for level {level}: "
            f"data range is [{np.nanmin(field):.2f}, {np.nanmax(field):.2f}]"
        )
        return []

    cs = ax.contour(
        X,
        Y,
        field,
        levels=[level],
        colors=[color],
        linestyles=[linestyle],
        linewidths=[linewidth]
    )

    return cs.allsegs[0]


def extract_reached_stop_boundary(ax, X, Y, reached_stop_map, color, linestyle, linewidth):
    """
    Extract the 120 C boundary using reached_stop_map.
    """
    field = reached_stop_map.astype(float)

    if not (np.nanmin(field) < 0.5 < np.nanmax(field)):
        print("No reached-stop boundary: all points are either reached or not reached.")
        return []

    cs = ax.contour(
        X,
        Y,
        field,
        levels=[0.5],
        colors=[color],
        linestyles=[linestyle],
        linewidths=[linewidth]
    )

    return cs.allsegs[0]


def epsilon_stats_from_segments(segments):
    """
    segments: list of arrays with columns [Risc, Ueq]
    """
    if len(segments) == 0:
        return {
            "N": 0,
            "R_values": np.array([]),
            "U_values": np.array([]),
            "eps_values": np.array([]),
            "eps_mean": np.nan,
            "eps_median": np.nan,
            "eps_std": np.nan,
            "eps_var": np.nan,
            "eps_rel_var": np.nan,
            "eps_min": np.nan,
            "eps_max": np.nan
        }

    points = np.vstack(segments)
    R_points = points[:, 0]
    U_points = points[:, 1]

    valid = (
        np.isfinite(R_points)
        & np.isfinite(U_points)
        & (R_points > 0)
        & (U_points > 0)
    )

    R_points = R_points[valid]
    U_points = U_points[valid]

    eps_values = epsilon_value(R_points, U_points)

    order = np.argsort(R_points)
    R_points = R_points[order]
    U_points = U_points[order]
    eps_values = eps_values[order]

    eps_mean = np.mean(eps_values)
    eps_std = np.std(eps_values)
    eps_var = np.var(eps_values)

    return {
        "N": len(eps_values),
        "R_values": R_points,
        "U_values": U_points,
        "eps_values": eps_values,
        "eps_mean": eps_mean,
        "eps_median": np.median(eps_values),
        "eps_std": eps_std,
        "eps_var": eps_var,
        "eps_rel_var": eps_std / eps_mean if eps_mean != 0 else np.nan,
        "eps_min": np.min(eps_values),
        "eps_max": np.max(eps_values)
    }


# ============================================================
# 8. Run or load all ambient temperature cases
# ============================================================
results = {}

for Tamb_C, Tamb_K in zip(Tamb_cases_C, Tamb_cases_K):
    results[Tamb_C] = get_result_for_Tamb(Tamb_C, Tamb_K)


# ============================================================
# 9. Figure 1:
#    Compare simulated 70 C and 120 C thermal boundaries
# ============================================================
colors = {
    20.0: "tab:blue",
    40.0: "tab:green",
    60.0: "tab:red"
}

boundary_stats = {}

fig, ax = plt.subplots(figsize=(11, 7))

for Tamb_C in Tamb_cases_C:
    data = results[Tamb_C]

    Tmax_map = data["Tmax_map"]
    reached_stop_map = data["reached_stop_map"].astype(bool)

    Z_C = Tmax_map - 273.15
    color = colors[Tamb_C]

    # 70 C boundary
    segments_70 = extract_contour_segments(
        ax=ax,
        X=R_grid,
        Y=U_grid,
        field=Z_C,
        level=70.0,
        color=color,
        linestyle="-",
        linewidth=2.4
    )

    stats_70 = epsilon_stats_from_segments(segments_70)

    # 120 C stopping boundary
    segments_120 = extract_reached_stop_boundary(
        ax=ax,
        X=R_grid,
        Y=U_grid,
        reached_stop_map=reached_stop_map,
        color=color,
        linestyle="--",
        linewidth=2.4
    )

    stats_120 = epsilon_stats_from_segments(segments_120)

    boundary_stats[(Tamb_C, 70.0)] = stats_70
    boundary_stats[(Tamb_C, 120.0)] = stats_120

ax.set_xscale("log")
ax.set_xlabel(r"Internal Short Resistance $R_{isc}$ [$\Omega$]", fontsize=13)
ax.set_ylabel(r"Cooling Coefficient $U_{eq}$ [W/K]", fontsize=13)

ax.set_title(
    r"Simulated Thermal Boundaries under Different Ambient Temperatures",
    fontsize=14
)

ax.grid(True, which="both", linestyle="--", alpha=0.35)

legend_handles = []

for Tamb_C in Tamb_cases_C:
    color = colors[Tamb_C]

    legend_handles.append(
        Line2D(
            [0], [0],
            color=color,
            linestyle="-",
            linewidth=2.4,
            label=rf"$T_{{amb}}={Tamb_C:.0f}^\circ$C, $T_{{max}}$ reaches $70^\circ$C"
        )
    )

    legend_handles.append(
        Line2D(
            [0], [0],
            color=color,
            linestyle="--",
            linewidth=2.4,
            label=rf"$T_{{amb}}={Tamb_C:.0f}^\circ$C, $T_{{max}}$ reaches $120^\circ$C"
        )
    )

ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

plt.tight_layout()

fig1_pdf = fig_dir / "Tamb_comparison_thermal_boundaries.pdf"
fig1_png = fig_dir / "Tamb_comparison_thermal_boundaries.png"

plt.savefig(fig1_pdf, bbox_inches="tight")
plt.savefig(fig1_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved Figure 1 to:\n{fig1_pdf}\n{fig1_png}")


# ============================================================
# 10. Figure 2:
#     Combined mean time-scale ratio curves
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))

R_line = np.logspace(
    np.log10(Risc_array.min()),
    np.log10(Risc_array.max()),
    1200
)

for Tamb_C in Tamb_cases_C:
    color = colors[Tamb_C]

    stats_70 = boundary_stats[(Tamb_C, 70.0)]
    eps70 = stats_70["eps_mean"]

    if np.isfinite(eps70):
        U_line_70 = Ueq_for_epsilon(eps70, R_line)
        valid_70 = (
            (U_line_70 >= Ueq_array.min())
            & (U_line_70 <= Ueq_array.max())
        )

        ax.plot(
            R_line[valid_70],
            U_line_70[valid_70],
            color=color,
            linestyle="-",
            linewidth=2.4,
            label=rf"$T_{{amb}}={Tamb_C:.0f}^\circ$C, "
                  rf"$\bar\epsilon_{{70}}={eps70:.3f}$"
        )

    stats_120 = boundary_stats[(Tamb_C, 120.0)]
    eps120 = stats_120["eps_mean"]

    if np.isfinite(eps120):
        U_line_120 = Ueq_for_epsilon(eps120, R_line)
        valid_120 = (
            (U_line_120 >= Ueq_array.min())
            & (U_line_120 <= Ueq_array.max())
        )

        ax.plot(
            R_line[valid_120],
            U_line_120[valid_120],
            color=color,
            linestyle="--",
            linewidth=2.4,
            label=rf"$T_{{amb}}={Tamb_C:.0f}^\circ$C, "
                  rf"$\bar\epsilon_{{120}}={eps120:.3f}$"
        )

ax.set_xscale("log")
ax.set_xlabel(r"Internal Short Resistance $R_{isc}$ [$\Omega$]", fontsize=13)
ax.set_ylabel(r"Cooling Coefficient $U_{eq}$ [W/K]", fontsize=13)

ax.set_title(
    r"Mean Time-Scale Ratio Curves Corresponding to Thermal Boundaries",
    fontsize=14
)

ax.grid(True, which="both", linestyle="--", alpha=0.35)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()

fig2_pdf = fig_dir / "Tamb_comparison_mean_timescale_boundaries.pdf"
fig2_png = fig_dir / "Tamb_comparison_mean_timescale_boundaries.png"

plt.savefig(fig2_pdf, bbox_inches="tight")
plt.savefig(fig2_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved Figure 2 to:\n{fig2_pdf}\n{fig2_png}")


# ============================================================
# 11. Separate figures:
#     Same style as the 40 C mean-epsilon boundary figure
# ============================================================
soft_temp_cmap = LinearSegmentedColormap.from_list(
    "soft_temperature",
    [
        "#f7fbff",
        "#c6dbef",
        "#9ecae1",
        "#fee08b",
        "#fdae61",
        "#d73027",
    ],
    N=256
)


def plot_single_Tamb_mean_boundary_figure(Tamb_C):
    data = results[Tamb_C]

    Tmax_map = data["Tmax_map"]
    reached_stop_map = data["reached_stop_map"].astype(bool)
    Z_C = Tmax_map - 273.15

    stats_70 = boundary_stats[(Tamb_C, 70.0)]
    stats_120 = boundary_stats[(Tamb_C, 120.0)]

    eps70 = stats_70["eps_mean"]
    eps120 = stats_120["eps_mean"]

    fig, ax = plt.subplots(figsize=(11, 7))

    cf = ax.contourf(
        R_grid,
        U_grid,
        Z_C,
        levels=np.linspace(40, 120, 33),
        cmap=soft_temp_cmap,
        extend="max"
    )

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$T_{\max}$ [$^\circ$C]", fontsize=12)

    # 70 C thermal boundary
    if has_level(Z_C, 70.0):
        ax.contour(
            R_grid,
            U_grid,
            Z_C,
            levels=[70.0],
            colors=["black"],
            linewidths=2.4,
            linestyles="-"
        )

    # 120 C stopping boundary
    field_stop = reached_stop_map.astype(float)
    if np.nanmin(field_stop) < 0.5 < np.nanmax(field_stop):
        ax.contour(
            R_grid,
            U_grid,
            field_stop,
            levels=[0.5],
            colors=["#b00020"],
            linewidths=2.8,
            linestyles="-"
        )

    # Mean epsilon curve for 70 C
    if np.isfinite(eps70):
        U_line_70 = Ueq_for_epsilon(eps70, R_line)
        valid_70 = (
            (U_line_70 >= Ueq_array.min())
            & (U_line_70 <= Ueq_array.max())
        )

        ax.plot(
            R_line[valid_70],
            U_line_70[valid_70],
            color="#0077b6",
            linestyle="--",
            linewidth=2.4
        )

        if np.any(valid_70):
            idx = np.where(valid_70)[0]
            mid = idx[len(idx) // 2]
            ax.text(
                R_line[mid],
                U_line_70[mid] * 1.08,
                rf"$\bar{{\epsilon}}={eps70:.2f}$",
                color="#0077b6",
                fontsize=10,
                ha="center",
                va="bottom",
                rotation=-25
            )

    # Mean epsilon curve for 120 C
    if np.isfinite(eps120):
        U_line_120 = Ueq_for_epsilon(eps120, R_line)
        valid_120 = (
            (U_line_120 >= Ueq_array.min())
            & (U_line_120 <= Ueq_array.max())
        )

        ax.plot(
            R_line[valid_120],
            U_line_120[valid_120],
            color="#7b2cbf",
            linestyle="--",
            linewidth=2.6
        )

        if np.any(valid_120):
            idx = np.where(valid_120)[0]
            mid = idx[len(idx) // 2]
            ax.text(
                R_line[mid],
                U_line_120[mid] * 1.08,
                rf"$\bar{{\epsilon}}={eps120:.2f}$",
                color="#7b2cbf",
                fontsize=10,
                ha="center",
                va="bottom",
                rotation=-25
            )

    ax.set_xscale("log")
    ax.set_xlim(0.01, 10)
    ax.set_ylim(0.05, 10)

    ax.set_xlabel(r"Internal Short Resistance $R_{isc}$ [$\Omega$]", fontsize=13)
    ax.set_ylabel(r"Cooling Coefficient $U_{eq}$ [W/K]", fontsize=13)

    ax.set_title(
        r"Simulated Thermal Boundaries and Time-scale Ratio Boundaries",
        fontsize=14
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.30)

    legend_elements = [
        Line2D(
            [0], [0],
            color="black",
            lw=2.4,
            linestyle="-",
            label=r"$T_{\max}$ reaches $70^\circ$C"
        ),
        Line2D(
            [0], [0],
            color="#b00020",
            lw=2.8,
            linestyle="-",
            label=r"$T_{\max}$ reaches $120^\circ$C"
        )
    ]

    if np.isfinite(eps70):
        legend_elements.append(
            Line2D(
                [0], [0],
                color="#0077b6",
                lw=2.4,
                linestyle="--",
                label=rf"$\bar{{\epsilon}}={eps70:.2f}$"
            )
        )

    if np.isfinite(eps120):
        legend_elements.append(
            Line2D(
                [0], [0],
                color="#7b2cbf",
                lw=2.6,
                linestyle="--",
                label=rf"$\bar{{\epsilon}}={eps120:.2f}$"
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9.5,
        framealpha=0.95,
        title=rf"$T_{{amb}}={Tamb_C:.0f}^\circ C$"
    )

    plt.tight_layout()

    fig_pdf = fig_dir / f"Tamb{int(Tamb_C)}C_thermal_boundaries_mean_timescale.pdf"
    fig_png = fig_dir / f"Tamb{int(Tamb_C)}C_thermal_boundaries_mean_timescale.png"

    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved separated Tamb={Tamb_C:.0f} C figure to:")
    print(fig_pdf)
    print(fig_png)


for Tamb_C in Tamb_cases_C:
    plot_single_Tamb_mean_boundary_figure(Tamb_C)


# ============================================================
# 12. Save boundary statistics
# ============================================================
summary_rows = []

for Tamb_C in Tamb_cases_C:
    for boundary_T in [70.0, 120.0]:
        stats = boundary_stats[(Tamb_C, boundary_T)]

        row = [
            Tamb_C,
            boundary_T,
            stats["N"],
            stats["eps_mean"],
            stats["eps_median"],
            stats["eps_std"],
            stats["eps_var"],
            stats["eps_rel_var"],
            stats["eps_min"],
            stats["eps_max"]
        ]

        summary_rows.append(row)

summary_array = np.array(summary_rows, dtype=float)

summary_path = output_dir / "Tamb_boundary_epsilon_summary.npz"

np.savez(
    summary_path,
    summary_array=summary_array,
    columns=np.array([
        "Tamb_C",
        "Boundary_C",
        "N",
        "eps_mean",
        "eps_median",
        "eps_std",
        "eps_variance",
        "eps_relative_variation",
        "eps_min",
        "eps_max"
    ], dtype=object)
)

summary_txt_path = output_dir / "Tamb_boundary_epsilon_summary.txt"

with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("Boundary epsilon summary with Iext included in SOC and epsilon formula\n")
    f.write("=====================================================================\n\n")
    f.write(f"Iext = {Iext:.6f} A\n")
    f.write(f"include_Iext_in_SOC = {include_Iext_in_SOC}\n")
    f.write(f"INITIAL_T_EQUALS_AMBIENT = {INITIAL_T_EQUALS_AMBIENT}\n")
    f.write(f"N_R = {N_R}, N_U = {N_U}\n")
    f.write(f"Uref = {Uref:.10f} V\n\n")

    for row in summary_rows:
        Tamb_C, boundary_T, N, eps_mean, eps_median, eps_std, eps_var, eps_rel_var, eps_min, eps_max = row

        f.write(f"Tamb = {Tamb_C:.0f} C, boundary = {boundary_T:.0f} C\n")
        f.write("-" * 60 + "\n")
        f.write(f"N sampled points       = {int(N)}\n")
        f.write(f"epsilon mean           = {eps_mean:.10f}\n")
        f.write(f"epsilon median         = {eps_median:.10f}\n")
        f.write(f"epsilon variance       = {eps_var:.10e}\n")
        f.write(f"epsilon std            = {eps_std:.10f}\n")
        f.write(f"relative variation     = {100.0 * eps_rel_var:.6f}%\n")
        f.write(f"epsilon min            = {eps_min:.10f}\n")
        f.write(f"epsilon max            = {eps_max:.10f}\n\n")

print(f"\nSaved boundary epsilon summary to:")
print(summary_path)
print(summary_txt_path)


# ============================================================
# 13. Print summary
# ============================================================
print("\n" + "=" * 100)
print("Boundary epsilon summary")
print("=" * 100)

print(
    f"{'Tamb [C]':>10} | "
    f"{'Boundary [C]':>12} | "
    f"{'N':>6} | "
    f"{'mean eps':>10} | "
    f"{'median eps':>10} | "
    f"{'var eps':>12} | "
    f"{'rel var [%]':>12} | "
    f"{'min eps':>10} | "
    f"{'max eps':>10}"
)

print("-" * 100)

for row in summary_rows:
    Tamb_C, boundary_T, N, eps_mean, eps_median, eps_std, eps_var, eps_rel_var, eps_min, eps_max = row

    print(
        f"{Tamb_C:10.0f} | "
        f"{boundary_T:12.0f} | "
        f"{int(N):6d} | "
        f"{eps_mean:10.4f} | "
        f"{eps_median:10.4f} | "
        f"{eps_var:12.4e} | "
        f"{100.0 * eps_rel_var:12.2f} | "
        f"{eps_min:10.4f} | "
        f"{eps_max:10.4f}"
    )

print("=" * 100)


# ============================================================
# 14. Print simulation result summary
# ============================================================
print("\n" + "=" * 80)
print("Simulation summary")
print("=" * 80)

for Tamb_C in Tamb_cases_C:
    data = results[Tamb_C]

    Tmax_map = data["Tmax_map"]
    reached_stop_map = data["reached_stop_map"].astype(bool)
    reached_empty_map = (
        data["reached_empty_map"].astype(bool)
        if "reached_empty_map" in data
        else np.zeros_like(R_grid, dtype=bool)
    )
    margin_map = data["margin_map"]

    Z_C = Tmax_map - 273.15

    stable_thermal_direction = margin_map > 0
    unstable_thermal_direction = margin_map <= 0

    print(f"\nTamb = {Tamb_C:.0f} C")
    print(f"Grid size: {N_U} x {N_R} = {N_U * N_R}")
    print(f"min Tmax = {np.nanmin(Z_C):.2f} C")
    print(f"max Tmax = {np.nanmax(Z_C):.2f} C")
    print(f"Reached 120 C cutoff: {np.sum(reached_stop_map)} points")
    print(f"Reached SOC empty: {np.sum(reached_empty_map)} points")
    print(f"lambda2(Tmax) < 0: {np.sum(stable_thermal_direction)} points")
    print(f"lambda2(Tmax) >= 0: {np.sum(unstable_thermal_direction)} points")

print("\nAll done.")