import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 0. Global switches
# ============================================================
FORCE_RERUN = False
# False: if a compatible npz already exists, load it directly.
# True : always rerun newly generated current cases.

SAVE_FIGURES = False
# False: show figures only, do not save pdf/png figure files.
# True : save pdf/png figure files to fig_dir.

INITIAL_T_EQUALS_AMBIENT = True
# True : T0 = Tamb.
# False: T0 is fixed as FIXED_T0_C.

FIXED_T0_C = 40.0

# Whether to stop integration once SOC reaches 0.
# For the thesis model this should usually be True because the ODE is only
# physically meaningful for S in [0, 1]. The code also saves reached_empty_map
# so that high-current cases can be interpreted separately.
STOP_AT_SOC_EMPTY = True

# Boundary epsilon statistics exclude contour sampling points whose nearest
# grid point reaches SOC = 0 before reaching the 120 C cutoff.
# The figures are not given an extra SOC-depleted region, so their appearance
# stays consistent with the previous thermal-boundary figures.
EXCLUDE_SOC_DEPLETED_FROM_BOUNDARY_STATS = True


# ============================================================
# 1. Basic parameters
# ============================================================
m = 0.720
Cp = 1100.0
Qcap = 25.0          # Ah
Rcell = 0.002
Rgas = 8.314
zeta = 0.28

# External current is included in SOC evolution.
# This must be True to match the final thesis model.
include_Iext_in_SOC = True

# Fixed ambient temperature for the external current sensitivity study
Tamb_C = 40.0
Tamb = Tamb_C + 273.15

# External current cases
# Added 25 A as a high-current sensitivity case.
Iext_cases = [0.0, 2.5, 5.0, 7.5, 25.0]

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

# Default grid for all normal current cases
N_R_DEFAULT = 70
N_U_DEFAULT = 70

# Only 25 A uses a denser grid
N_R_25A = 120
N_U_25A = 120


# ============================================================
# 2. Output settings
# ============================================================
output_dir = Path(r"D:\files\毕业设计\Iext_sweep_results_Tamb40C_SOCincluded")
output_dir.mkdir(parents=True, exist_ok=True)

fig_dir = output_dir / "figures"
if SAVE_FIGURES:
    fig_dir.mkdir(parents=True, exist_ok=True)

# Existing corrected 5 A, 40 C result
fixed_5A_path = Path(
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


def epsilon_value(Risc, Ueq, Iext):
    """
    Final epsilon formula with external current included:

    epsilon(Risc, Ueq; Iext)
    =
    m Cp / (3600 Qcap Ueq)
    *
    [ Iext + Uref / (Rcell + Risc) ]
    """
    return (m * Cp / (3600.0 * Qcap * Ueq)) * (
        Iext + Uref / (Rcell + Risc)
    )


def Ueq_for_epsilon(eps0, Risc, Iext):
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


def rhs(t, y, Risc, Ueq, Iext):
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


def get_initial_temperature():
    if INITIAL_T_EQUALS_AMBIENT:
        return Tamb
    else:
        return FIXED_T0_C + 273.15


def simulate_one_point(Risc, Ueq, Iext):
    """
    Simulate one parameter point under a given external current.

    Stop if:
    1. T reaches 120 C;
    2. SOC reaches 0, if STOP_AT_SOC_EMPTY is True;
    3. final time t_end is reached.
    """

    T0 = get_initial_temperature()

    events = [make_event_T_stop()]
    if STOP_AT_SOC_EMPTY:
        events.append(make_event_SOC_empty())

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, Risc, Ueq, Iext),
        t_span=(0.0, t_end),
        y0=[S0, T0],
        method="DOP853",
        rtol=1e-6,
        atol=1e-8,
        max_step=20.0,
        events=events,
    )

    reached_stop = len(sol.t_events[0]) > 0

    if STOP_AT_SOC_EMPTY:
        reached_empty = len(sol.t_events[1]) > 0
    else:
        reached_empty = False

    if reached_stop:
        Tmax = T_stop
    else:
        Tmax = np.max(sol.y[1])

    final_S = sol.y[0, -1]
    final_T = sol.y[1, -1]

    lam2 = (dQ_chem_dT(Tmax) - Ueq) / (m * Cp)
    margin = Ueq - dQ_chem_dT(Tmax)

    # 0: reached t_end, 1: reached 120 C, 2: reached SOC empty
    if reached_stop:
        stop_reason = 1
    elif reached_empty:
        stop_reason = 2
    else:
        stop_reason = 0

    return Tmax, reached_stop, reached_empty, sol.t[-1], final_S, final_T, lam2, margin, stop_reason


# ============================================================
# 5. Parameter grid
# ============================================================
def grid_size_for_Iext(Iext):
    if np.isclose(Iext, 25.0):
        return N_R_25A, N_U_25A
    return N_R_DEFAULT, N_U_DEFAULT


def make_grid_for_Iext(Iext):
    N_R, N_U = grid_size_for_Iext(Iext)

    Risc_array = np.logspace(-2, 1, N_R)       # 0.01 ~ 10 Ohm
    Ueq_array = np.linspace(0.05, 10.0, N_U)   # 0.05 ~ 10 W/K

    R_grid, U_grid = np.meshgrid(Risc_array, Ueq_array)

    return Risc_array, Ueq_array, R_grid, U_grid


# ============================================================
# 6. Save/load simulation results
# ============================================================
def Iext_tag(Iext):
    return str(Iext).replace(".", "p")


def npz_path_for_Iext(Iext, N_R, N_U):
    if np.isclose(Iext, 5.0):
        return fixed_5A_path

    return output_dir / (
        f"Tmax_sweep_Iext{Iext_tag(Iext)}A_SOCincluded_"
        f"Tamb{int(Tamb_C)}C_NR{N_R}_NU{N_U}.npz"
    )


def load_npz_to_dict(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def grid_is_compatible(data, Risc_array, Ueq_array, R_grid, U_grid):
    """
    Only check grid and required result fields.
    This is used for the fixed 5 A file because it may not store all metadata.
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


def result_is_compatible(data, Iext, Risc_array, Ueq_array, R_grid, U_grid):
    """
    Compatibility check for newly generated current cases.
    """
    if not grid_is_compatible(data, Risc_array, Ueq_array, R_grid, U_grid):
        return False

    required_metadata = [
        "Iext",
        "include_Iext_in_SOC",
        "Tamb_C",
        "Tamb_K",
        "eps_grid",
        "STOP_AT_SOC_EMPTY",
        "final_SOC_map",
        "stop_reason_map",
    ]

    for key in required_metadata:
        if key not in data:
            return False

    if not np.isclose(float(data["Iext"]), Iext):
        return False

    if bool(data["include_Iext_in_SOC"]) != include_Iext_in_SOC:
        return False

    if bool(data["STOP_AT_SOC_EMPTY"]) != STOP_AT_SOC_EMPTY:
        return False

    if not np.isclose(float(data["Tamb_C"]), Tamb_C):
        return False

    if not np.isclose(float(data["Tamb_K"]), Tamb):
        return False

    eps_check = epsilon_value(R_grid, U_grid, Iext)

    if not np.allclose(data["eps_grid"], eps_check, rtol=1e-10, atol=1e-12):
        return False

    return True


def run_sweep_for_Iext(Iext):
    Risc_array, Ueq_array, R_grid, U_grid = make_grid_for_Iext(Iext)
    N_R = len(Risc_array)
    N_U = len(Ueq_array)

    print("\n" + "=" * 70)
    print(f"Running simulation: Iext = {Iext:.2f} A, Tamb = {Tamb_C:.0f} C")
    print(f"Grid size: N_R = {N_R}, N_U = {N_U}")
    print("=" * 70)

    eps_grid = epsilon_value(R_grid, U_grid, Iext)

    Tmax_map = np.full_like(R_grid, np.nan, dtype=float)
    lambda2_map = np.full_like(R_grid, np.nan, dtype=float)
    margin_map = np.full_like(R_grid, np.nan, dtype=float)
    reached_stop_map = np.zeros_like(R_grid, dtype=bool)
    reached_empty_map = np.zeros_like(R_grid, dtype=bool)
    final_time_map = np.full_like(R_grid, np.nan, dtype=float)
    final_SOC_map = np.full_like(R_grid, np.nan, dtype=float)
    final_T_map = np.full_like(R_grid, np.nan, dtype=float)
    stop_reason_map = np.full_like(R_grid, -1, dtype=int)

    for i, Ueq in enumerate(Ueq_array):
        for j, Risc in enumerate(Risc_array):

            (
                Tmax,
                reached_stop,
                reached_empty,
                final_time,
                final_S,
                final_T,
                lam2,
                margin,
                stop_reason,
            ) = simulate_one_point(Risc, Ueq, Iext)

            Tmax_map[i, j] = Tmax
            reached_stop_map[i, j] = reached_stop
            reached_empty_map[i, j] = reached_empty
            final_time_map[i, j] = final_time
            final_SOC_map[i, j] = final_S
            final_T_map[i, j] = final_T
            lambda2_map[i, j] = lam2
            margin_map[i, j] = margin
            stop_reason_map[i, j] = stop_reason

        print(f"Iext = {Iext:.2f} A: finished Ueq row {i + 1}/{N_U}")

    save_path = npz_path_for_Iext(Iext, N_R, N_U)

    np.savez(
        save_path,
        Tamb_C=Tamb_C,
        Tamb_K=Tamb,
        Iext=Iext,
        include_Iext_in_SOC=include_Iext_in_SOC,
        STOP_AT_SOC_EMPTY=STOP_AT_SOC_EMPTY,
        INITIAL_T_EQUALS_AMBIENT=INITIAL_T_EQUALS_AMBIENT,
        FIXED_T0_C=FIXED_T0_C,
        S0=S0,
        T0_K=get_initial_temperature(),
        T0_C=get_initial_temperature() - 273.15,
        T_alarm=T_alarm,
        T_limit=T_limit,
        T_stop=T_stop,
        t_end=t_end,
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
        final_time_map=final_time_map,
        final_SOC_map=final_SOC_map,
        final_T_map=final_T_map,
        stop_reason_map=stop_reason_map,
    )

    print(f"Saved simulation result to: {save_path}")

    return load_npz_to_dict(save_path)


def get_result_for_Iext(Iext):
    Risc_array, Ueq_array, R_grid, U_grid = make_grid_for_Iext(Iext)
    N_R = len(Risc_array)
    N_U = len(Ueq_array)

    save_path = npz_path_for_Iext(Iext, N_R, N_U)

    # 5 A: always load the fixed corrected result
    if np.isclose(Iext, 5.0):
        if not fixed_5A_path.exists():
            raise FileNotFoundError(f"Fixed 5 A result file not found: {fixed_5A_path}")

        data = load_npz_to_dict(fixed_5A_path)

        if not grid_is_compatible(data, Risc_array, Ueq_array, R_grid, U_grid):
            raise ValueError(
                "The fixed 5 A file exists, but its grid is not compatible "
                "with the current Risc_array and Ueq_array."
            )

        print(f"Loaded fixed corrected 5 A result: {fixed_5A_path}")
        return data

    # Other current cases: load existing compatible file or rerun
    if (not FORCE_RERUN) and save_path.exists():
        data = load_npz_to_dict(save_path)

        if result_is_compatible(data, Iext, Risc_array, Ueq_array, R_grid, U_grid):
            print(f"Loaded existing compatible result: {save_path}")
            return data
        else:
            print(f"Existing file is not compatible, rerunning: {save_path}")

    return run_sweep_for_Iext(Iext)


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


def empty_epsilon_stats(N_raw=0, N_excluded_soc_depleted=0):
    return {
        "N": 0,
        "N_raw": int(N_raw),
        "N_excluded_soc_depleted": int(N_excluded_soc_depleted),
        "R_values": np.array([]),
        "U_values": np.array([]),
        "eps_values": np.array([]),
        "eps_mean": np.nan,
        "eps_median": np.nan,
        "eps_std": np.nan,
        "eps_var": np.nan,
        "eps_rel_var": np.nan,
        "eps_min": np.nan,
        "eps_max": np.nan,
    }


def nearest_grid_soc_depleted_mask(R_points, U_points, reached_empty_map, Risc_array, Ueq_array):
    """
    Return True for contour points whose nearest simulation grid point belongs
    to the SOC-depleted set.

    This is used only for statistics. The plotted thermal boundaries keep the
    same visual format as before and do not show an additional SOC-depleted
    region.
    """
    if reached_empty_map is None:
        return np.zeros_like(R_points, dtype=bool)

    expected_shape = (len(Ueq_array), len(Risc_array))
    if reached_empty_map.shape != expected_shape:
        print("Warning: reached_empty_map shape is incompatible; SOC exclusion is skipped.")
        return np.zeros_like(R_points, dtype=bool)

    log_R_grid_1d = np.log(Risc_array)
    log_R_points = np.log(R_points)

    j_right = np.searchsorted(log_R_grid_1d, log_R_points)
    j_right = np.clip(j_right, 0, len(Risc_array) - 1)
    j_left = np.clip(j_right - 1, 0, len(Risc_array) - 1)

    dist_left = np.abs(log_R_points - log_R_grid_1d[j_left])
    dist_right = np.abs(log_R_points - log_R_grid_1d[j_right])
    j_near = np.where(dist_left <= dist_right, j_left, j_right)

    i_right = np.searchsorted(Ueq_array, U_points)
    i_right = np.clip(i_right, 0, len(Ueq_array) - 1)
    i_left = np.clip(i_right - 1, 0, len(Ueq_array) - 1)

    dist_left = np.abs(U_points - Ueq_array[i_left])
    dist_right = np.abs(U_points - Ueq_array[i_right])
    i_near = np.where(dist_left <= dist_right, i_left, i_right)

    return reached_empty_map[i_near, j_near].astype(bool)


def epsilon_stats_from_segments(
    segments,
    Iext,
    Risc_array,
    Ueq_array,
    R_grid,
    U_grid,
    reached_empty_map=None,
    exclude_soc_depleted=None
):
    """
    segments: list of arrays with columns [Risc, Ueq]

    If exclude_soc_depleted is True, the returned epsilon statistics exclude
    contour sampling points whose nearest grid point reached SOC = 0 before
    reaching 120 C. If exclude_soc_depleted is False, all contour sampling
    points are used. If it is None, the global switch
    EXCLUDE_SOC_DEPLETED_FROM_BOUNDARY_STATS is used.
    """
    if len(segments) == 0:
        return empty_epsilon_stats()

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

    N_raw = len(R_points)
    N_excluded_soc_depleted = 0

    if exclude_soc_depleted is None:
        exclude_soc_depleted = EXCLUDE_SOC_DEPLETED_FROM_BOUNDARY_STATS

    if exclude_soc_depleted and reached_empty_map is not None:
        soc_depleted_nearest = nearest_grid_soc_depleted_mask(
            R_points,
            U_points,
            reached_empty_map,
            Risc_array,
            Ueq_array,
        )

        keep_for_stats = ~soc_depleted_nearest
        N_excluded_soc_depleted = int(np.sum(~keep_for_stats))

        R_points = R_points[keep_for_stats]
        U_points = U_points[keep_for_stats]

    if len(R_points) == 0:
        return empty_epsilon_stats(
            N_raw=N_raw,
            N_excluded_soc_depleted=N_excluded_soc_depleted,
        )

    # Use the same epsilon sampling method as the Chapter 3 plotting code.
    # First evaluate epsilon on the saved parameter grid, then interpolate
    # the grid values at the contour points. This keeps the mean epsilon
    # values consistent with the previous 40 C, 5 A figures.
    eps_grid_current = epsilon_value(R_grid, U_grid, Iext)
    eps_interp = RegularGridInterpolator(
        points=(Ueq_array, Risc_array),
        values=eps_grid_current,
        bounds_error=False,
        fill_value=np.nan,
    )

    pts = np.column_stack([U_points, R_points])
    eps_values = eps_interp(pts)

    finite_eps = np.isfinite(eps_values)
    R_points = R_points[finite_eps]
    U_points = U_points[finite_eps]
    eps_values = eps_values[finite_eps]

    if len(R_points) == 0:
        return empty_epsilon_stats(
            N_raw=N_raw,
            N_excluded_soc_depleted=N_excluded_soc_depleted,
        )

    order = np.argsort(R_points)
    R_points = R_points[order]
    U_points = U_points[order]
    eps_values = eps_values[order]

    eps_mean = np.mean(eps_values)
    eps_std = np.std(eps_values)
    eps_var = np.var(eps_values)

    return {
        "N": len(eps_values),
        "N_raw": int(N_raw),
        "N_excluded_soc_depleted": int(N_excluded_soc_depleted),
        "R_values": R_points,
        "U_values": U_points,
        "eps_values": eps_values,
        "eps_mean": eps_mean,
        "eps_median": np.median(eps_values),
        "eps_std": eps_std,
        "eps_var": eps_var,
        "eps_rel_var": eps_std / eps_mean if eps_mean != 0 else np.nan,
        "eps_min": np.min(eps_values),
        "eps_max": np.max(eps_values),
    }


def representative_epsilon_for_plot(stats, Iext, trim_ratio=0.10):
    """
    For normal current cases, use the original arithmetic mean epsilon.
    For 25 A, use a trimmed geometric mean to reduce endpoint and outlier effects.
    This only changes the plotted representative epsilon curve for 25 A.
    """

    eps_values = stats["eps_values"]

    eps_values = eps_values[
        np.isfinite(eps_values)
        & (eps_values > 0)
    ]

    if len(eps_values) == 0:
        return np.nan

    if np.isclose(Iext, 25.0):
        if len(eps_values) >= 10:
            q_low, q_high = np.quantile(
                eps_values,
                [trim_ratio, 1.0 - trim_ratio]
            )
            eps_values = eps_values[
                (eps_values >= q_low)
                & (eps_values <= q_high)
            ]

        if len(eps_values) == 0:
            return np.nan

        return np.exp(np.mean(np.log(eps_values)))

    return stats["eps_mean"]


def epsilon_label_for_plot(Iext, boundary_T, eps_value):
    if np.isclose(Iext, 25.0):
        return rf"$\epsilon_{{\mathrm{{fit}},{int(boundary_T)}}}={eps_value:.3f}$"
    else:
        return rf"$\bar\epsilon_{{{int(boundary_T)}}}={eps_value:.3f}$"


def epsilon_text_for_plot(Iext, eps_value):
    if np.isclose(Iext, 25.0):
        return rf"$\epsilon_{{\mathrm{{fit}}}}={eps_value:.2f}$"
    else:
        return rf"$\bar{{\epsilon}}={eps_value:.2f}$"


def get_reached_empty_map(data):
    if "reached_empty_map" in data:
        return data["reached_empty_map"].astype(bool)
    return np.zeros_like(data["R_grid"], dtype=bool)


def get_final_SOC_map(data):
    if "final_SOC_map" in data:
        return data["final_SOC_map"]
    return np.full_like(data["R_grid"], np.nan, dtype=float)


def get_stop_reason_map(data):
    if "stop_reason_map" in data:
        return data["stop_reason_map"]
    out = np.zeros_like(data["R_grid"], dtype=int)
    out[data["reached_stop_map"].astype(bool)] = 1
    return out


# ============================================================
# 8. Run or load all external current cases
# ============================================================
results = {}

for Iext in Iext_cases:
    results[Iext] = get_result_for_Iext(Iext)


# ============================================================
# 9. Figure 1:
#    Compare simulated 70 C and 120 C thermal boundaries
# ============================================================
colors = {
    0.0: "tab:blue",
    2.5: "tab:orange",
    5.0: "tab:green",
    7.5: "tab:red",
    25.0: "tab:purple"
}

boundary_stats = {}
# boundary_stats_for_plot uses all contour sampling points, so the figures keep
# the same visual format as before. boundary_stats uses SOC-depleted exclusion
# and is saved in the summary table for model-valid statistics.
boundary_stats_for_plot = {}

fig, ax = plt.subplots(figsize=(11, 7))

for Iext in Iext_cases:
    data = results[Iext]

    Risc_array = data["Risc_array"]
    Ueq_array = data["Ueq_array"]
    R_grid = data["R_grid"]
    U_grid = data["U_grid"]

    Tmax_map = data["Tmax_map"]
    reached_stop_map = data["reached_stop_map"].astype(bool)
    reached_empty_map = get_reached_empty_map(data)

    Z_C = Tmax_map - 273.15
    color = colors[Iext]

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

    stats_70_plot = epsilon_stats_from_segments(
        segments_70,
        Iext,
        Risc_array,
        Ueq_array,
        R_grid,
        U_grid,
        reached_empty_map=None,
        exclude_soc_depleted=False,
    )

    stats_70_valid = epsilon_stats_from_segments(
        segments_70,
        Iext,
        Risc_array,
        Ueq_array,
        R_grid,
        U_grid,
        reached_empty_map=reached_empty_map,
        exclude_soc_depleted=EXCLUDE_SOC_DEPLETED_FROM_BOUNDARY_STATS,
    )

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

    stats_120_plot = epsilon_stats_from_segments(
        segments_120,
        Iext,
        Risc_array,
        Ueq_array,
        R_grid,
        U_grid,
        reached_empty_map=None,
        exclude_soc_depleted=False,
    )

    stats_120_valid = epsilon_stats_from_segments(
        segments_120,
        Iext,
        Risc_array,
        Ueq_array,
        R_grid,
        U_grid,
        reached_empty_map=reached_empty_map,
        exclude_soc_depleted=EXCLUDE_SOC_DEPLETED_FROM_BOUNDARY_STATS,
    )

    boundary_stats_for_plot[(Iext, 70.0)] = stats_70_plot
    boundary_stats_for_plot[(Iext, 120.0)] = stats_120_plot
    boundary_stats[(Iext, 70.0)] = stats_70_valid
    boundary_stats[(Iext, 120.0)] = stats_120_valid

ax.set_xscale("log")
ax.set_xlabel(r"Internal Short Resistance $R_{isc}$ [$\Omega$]", fontsize=13)
ax.set_ylabel(r"Cooling Coefficient $U_{eq}$ [W/K]", fontsize=13)

ax.set_title(
    r"Simulated Thermal Boundaries under Different External Currents",
    fontsize=14
)

ax.grid(True, which="both", linestyle="--", alpha=0.35)

legend_handles = []

for Iext in Iext_cases:
    color = colors[Iext]

    legend_handles.append(
        Line2D(
            [0], [0],
            color=color,
            linestyle="-",
            linewidth=2.4,
            label=rf"$I_{{ext}}={Iext:g}\,\mathrm{{A}}$, $T_{{max}}$ reaches $70^\circ$C"
        )
    )

    legend_handles.append(
        Line2D(
            [0], [0],
            color=color,
            linestyle="--",
            linewidth=2.4,
            label=rf"$I_{{ext}}={Iext:g}\,\mathrm{{A}}$, $T_{{max}}$ reaches $120^\circ$C"
        )
    )

ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right")

plt.tight_layout()

if SAVE_FIGURES:
    fig1_pdf = fig_dir / "Iext_comparison_thermal_boundaries_Tamb40C.pdf"
    fig1_png = fig_dir / "Iext_comparison_thermal_boundaries_Tamb40C.png"

    plt.savefig(fig1_pdf, bbox_inches="tight")
    plt.savefig(fig1_png, dpi=300, bbox_inches="tight")

    print(f"Saved Figure 1 to:\n{fig1_pdf}\n{fig1_png}")
else:
    print("Figure 1 displayed only; no figure file was saved.")

plt.show()


# ============================================================
# 10. Figure 2:
#     Combined mean time-scale ratio curves
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))

R_min = min(np.min(results[Iext]["Risc_array"]) for Iext in Iext_cases)
R_max = max(np.max(results[Iext]["Risc_array"]) for Iext in Iext_cases)

R_line = np.logspace(
    np.log10(R_min),
    np.log10(R_max),
    1200
)

for Iext in Iext_cases:
    data = results[Iext]
    Ueq_array = data["Ueq_array"]

    color = colors[Iext]

    # Keep the original plotting statistics for all curves.
    # Only the representative epsilon value is treated differently for 25 A.
    stats_70 = boundary_stats_for_plot[(Iext, 70.0)]
    eps70 = representative_epsilon_for_plot(stats_70, Iext)

    if np.isfinite(eps70):
        U_line_70 = Ueq_for_epsilon(eps70, R_line, Iext)
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
            label=rf"$I_{{ext}}={Iext:g}\,\mathrm{{A}}$, "
                  + epsilon_label_for_plot(Iext, 70.0, eps70)
        )

    # Keep the original plotting statistics for all curves.
    # Only the representative epsilon value is treated differently for 25 A.
    stats_120 = boundary_stats_for_plot[(Iext, 120.0)]
    eps120 = representative_epsilon_for_plot(stats_120, Iext)

    if np.isfinite(eps120):
        U_line_120 = Ueq_for_epsilon(eps120, R_line, Iext)
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
            label=rf"$I_{{ext}}={Iext:g}\,\mathrm{{A}}$, "
                  + epsilon_label_for_plot(Iext, 120.0, eps120)
        )

ax.set_xscale("log")
ax.set_xlabel(r"Internal Short Resistance $R_{isc}$ [$\Omega$]", fontsize=13)
ax.set_ylabel(r"Cooling Coefficient $U_{eq}$ [W/K]", fontsize=13)

ax.set_title(
    r"Mean Time-Scale Ratio Curves Corresponding to Thermal Boundaries",
    fontsize=14
)

ax.grid(True, which="both", linestyle="--", alpha=0.35)
ax.legend(fontsize=8.5, loc="upper right")

plt.tight_layout()

if SAVE_FIGURES:
    fig2_pdf = fig_dir / "Iext_comparison_mean_timescale_boundaries_Tamb40C.pdf"
    fig2_png = fig_dir / "Iext_comparison_mean_timescale_boundaries_Tamb40C.png"

    plt.savefig(fig2_pdf, bbox_inches="tight")
    plt.savefig(fig2_png, dpi=300, bbox_inches="tight")

    print(f"Saved Figure 2 to:\n{fig2_pdf}\n{fig2_png}")
else:
    print("Figure 2 displayed only; no figure file was saved.")

plt.show()


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


def plot_single_Iext_mean_boundary_figure(Iext):
    data = results[Iext]

    R_grid = data["R_grid"]
    U_grid = data["U_grid"]
    Ueq_array = data["Ueq_array"]

    Tmax_map = data["Tmax_map"]
    reached_stop_map = data["reached_stop_map"].astype(bool)
    Z_C = Tmax_map - 273.15

    # Keep the original plotting statistics for all curves.
    # Only the representative epsilon value is treated differently for 25 A.
    stats_70 = boundary_stats_for_plot[(Iext, 70.0)]
    stats_120 = boundary_stats_for_plot[(Iext, 120.0)]

    eps70 = representative_epsilon_for_plot(stats_70, Iext)
    eps120 = representative_epsilon_for_plot(stats_120, Iext)

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
        U_line_70 = Ueq_for_epsilon(eps70, R_line, Iext)
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
                epsilon_text_for_plot(Iext, eps70),
                color="#0077b6",
                fontsize=10,
                ha="center",
                va="bottom",
                rotation=-25
            )

    # Mean epsilon curve for 120 C
    if np.isfinite(eps120):
        U_line_120 = Ueq_for_epsilon(eps120, R_line, Iext)
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
                epsilon_text_for_plot(Iext, eps120),
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
                label=epsilon_label_for_plot(Iext, 70.0, eps70)
            )
        )

    if np.isfinite(eps120):
        legend_elements.append(
            Line2D(
                [0], [0],
                color="#7b2cbf",
                lw=2.6,
                linestyle="--",
                label=epsilon_label_for_plot(Iext, 120.0, eps120)
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9.5,
        framealpha=0.95,
        title=rf"$I_{{ext}}={Iext:g}\,\mathrm{{A}}$, $T_{{amb}}=40^\circ C$"
    )

    plt.tight_layout()

    if SAVE_FIGURES:
        fig_pdf = fig_dir / f"Iext{Iext_tag(Iext)}A_Tamb40C_thermal_boundaries_mean_timescale.pdf"
        fig_png = fig_dir / f"Iext{Iext_tag(Iext)}A_Tamb40C_thermal_boundaries_mean_timescale.png"

        plt.savefig(fig_pdf, bbox_inches="tight")
        plt.savefig(fig_png, dpi=300, bbox_inches="tight")

        print(f"Saved separated Iext={Iext:.2f} A figure to:")
        print(fig_pdf)
        print(fig_png)
    else:
        print(f"Separated Iext={Iext:.2f} A figure displayed only; no figure file was saved.")

    plt.show()


for Iext in Iext_cases:
    plot_single_Iext_mean_boundary_figure(Iext)


# ============================================================
# 12. Save boundary statistics
# ============================================================
summary_rows = []

for Iext in Iext_cases:
    for boundary_T in [70.0, 120.0]:
        stats = boundary_stats[(Iext, boundary_T)]

        row = [
            Iext,
            Tamb_C,
            boundary_T,
            stats["N"],
            stats["N_raw"],
            stats["N_excluded_soc_depleted"],
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

summary_path = output_dir / "Iext_boundary_epsilon_summary_Tamb40C.npz"

np.savez(
    summary_path,
    summary_array=summary_array,
    columns=np.array([
        "Iext_A",
        "Tamb_C",
        "Boundary_C",
        "N_after_SOC_exclusion",
        "N_raw_before_SOC_exclusion",
        "N_excluded_SOC_depleted",
        "eps_mean",
        "eps_median",
        "eps_std",
        "eps_variance",
        "eps_relative_variation",
        "eps_min",
        "eps_max"
    ], dtype=object)
)

summary_txt_path = output_dir / "Iext_boundary_epsilon_summary_Tamb40C.txt"

with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("Boundary epsilon summary with Iext included in SOC and epsilon formula\n")
    f.write("=====================================================================\n\n")
    f.write(f"Tamb = {Tamb_C:.6f} C\n")
    f.write(f"include_Iext_in_SOC = {include_Iext_in_SOC}\n")
    f.write(f"STOP_AT_SOC_EMPTY = {STOP_AT_SOC_EMPTY}\n")
    f.write(f"INITIAL_T_EQUALS_AMBIENT = {INITIAL_T_EQUALS_AMBIENT}\n")
    f.write(f"default grid: N_R = {N_R_DEFAULT}, N_U = {N_U_DEFAULT}\n")
    f.write(f"25 A grid: N_R = {N_R_25A}, N_U = {N_U_25A}\n")
    f.write(f"Uref = {Uref:.10f} V\n")
    f.write(f"EXCLUDE_SOC_DEPLETED_FROM_BOUNDARY_STATS = {EXCLUDE_SOC_DEPLETED_FROM_BOUNDARY_STATS}\n")
    f.write("Note: boundary epsilon statistics exclude contour sampling points whose nearest grid point reached SOC = 0.\n")
    f.write("These SOC-depleted points are counted separately in the SOC depletion summary.\n")
    f.write("For plotted curves, the original all-point boundary statistics are kept. Only the 25 A representative epsilon uses a trimmed geometric mean.\n\n")

    for row in summary_rows:
        (
            Iext,
            Tamb_C_row,
            boundary_T,
            N,
            N_raw,
            N_excluded_soc,
            eps_mean,
            eps_median,
            eps_std,
            eps_var,
            eps_rel_var,
            eps_min,
            eps_max,
        ) = row

        f.write(f"Iext = {Iext:.2f} A, Tamb = {Tamb_C_row:.0f} C, boundary = {boundary_T:.0f} C\n")
        f.write("-" * 60 + "\n")
        f.write(f"N sampled points after SOC exclusion = {int(N)}\n")
        f.write(f"N sampled points before exclusion    = {int(N_raw)}\n")
        f.write(f"N excluded as SOC-depleted           = {int(N_excluded_soc)}\n")
        f.write(f"epsilon mean           = {eps_mean:.10f}\n")
        f.write(f"epsilon median         = {eps_median:.10f}\n")
        f.write(f"epsilon variance       = {eps_var:.10e}\n")
        f.write(f"epsilon std            = {eps_std:.10f}\n")
        f.write(f"relative variation     = {100.0 * eps_rel_var:.6f}%\n")
        f.write(f"epsilon min            = {eps_min:.10f}\n")
        f.write(f"epsilon max            = {eps_max:.10f}\n\n")

print("\nSaved boundary epsilon summary to:")
print(summary_path)
print(summary_txt_path)


# ============================================================
# 12b. Save SOC depletion summary
# ============================================================
soc_summary_rows = []

for Iext in Iext_cases:
    data = results[Iext]

    reached_stop_map = data["reached_stop_map"].astype(bool)
    has_soc_empty_info = "reached_empty_map" in data
    has_final_soc_info = "final_SOC_map" in data

    N_total = data["R_grid"].size

    if has_soc_empty_info:
        reached_empty_map = data["reached_empty_map"].astype(bool)
        soc_depleted_before_120 = reached_empty_map & (~reached_stop_map)
        N_soc_depleted = float(np.sum(soc_depleted_before_120))
        N_finished_no_120_no_soc0 = float(np.sum((~reached_stop_map) & (~reached_empty_map)))
    else:
        reached_empty_map = np.zeros_like(data["R_grid"], dtype=bool)
        N_soc_depleted = np.nan
        N_finished_no_120_no_soc0 = np.nan

    if has_final_soc_info:
        final_SOC_map = data["final_SOC_map"]
        min_final_SOC = float(np.nanmin(final_SOC_map))
        max_final_SOC = float(np.nanmax(final_SOC_map))
    else:
        min_final_SOC = np.nan
        max_final_SOC = np.nan

    row = [
        Iext,
        Tamb_C,
        N_total,
        float(np.sum(reached_stop_map)),
        N_soc_depleted,
        N_finished_no_120_no_soc0,
        float(has_soc_empty_info),
        float(has_final_soc_info),
        min_final_SOC,
        max_final_SOC,
    ]

    soc_summary_rows.append(row)

soc_summary_array = np.array(soc_summary_rows, dtype=float)

soc_summary_path = output_dir / "Iext_SOC_depletion_summary_Tamb40C.npz"

np.savez(
    soc_summary_path,
    summary_array=soc_summary_array,
    columns=np.array([
        "Iext_A",
        "Tamb_C",
        "N_total",
        "N_reached_120C",
        "N_SOC_depleted_before_120C",
        "N_finished_without_120C_or_SOC0",
        "has_reached_empty_map",
        "has_final_SOC_map",
        "min_final_SOC",
        "max_final_SOC",
    ], dtype=object)
)

soc_summary_txt_path = output_dir / "Iext_SOC_depletion_summary_Tamb40C.txt"

with open(soc_summary_txt_path, "w", encoding="utf-8") as f:
    f.write("SOC depletion summary for current sensitivity study\n")
    f.write("=================================================\n\n")
    f.write(f"Tamb = {Tamb_C:.6f} C\n")
    f.write(f"STOP_AT_SOC_EMPTY = {STOP_AT_SOC_EMPTY}\n")
    f.write("SOC-depleted points are treated as model-limited points, not as thermally safe points.\n")
    f.write("They are not separately marked in the final figures.\n")
    f.write("They are excluded from the boundary epsilon statistics when SOC information is available.\n\n")

    for row in soc_summary_rows:
        (
            Iext,
            Tamb_C_row,
            N_total,
            N_reached_120,
            N_soc_depleted,
            N_finished,
            has_soc_empty_info,
            has_final_soc_info,
            min_final_SOC,
            max_final_SOC,
        ) = row

        f.write(f"Iext = {Iext:.2f} A, Tamb = {Tamb_C_row:.0f} C\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total grid points                         = {int(N_total)}\n")
        f.write(f"Reached 120 C cutoff                      = {int(N_reached_120)}\n")

        if bool(has_soc_empty_info):
            f.write(f"Reached SOC = 0 before 120 C              = {int(N_soc_depleted)}\n")
            f.write(f"Finished without 120 C or SOC depletion   = {int(N_finished)}\n")
        else:
            f.write("Reached SOC = 0 before 120 C              = not available in this npz file\n")
            f.write("Finished without 120 C or SOC depletion   = not available in this npz file\n")

        if bool(has_final_soc_info):
            f.write(f"Minimum final SOC                         = {min_final_SOC:.6f}\n")
            f.write(f"Maximum final SOC                         = {max_final_SOC:.6f}\n")
        else:
            f.write("Final SOC map                             = not available in this npz file\n")

        f.write("\n")

print("\nSaved SOC depletion summary to:")
print(soc_summary_path)
print(soc_summary_txt_path)


# ============================================================
# 13. Print boundary epsilon summary
# ============================================================
print("\n" + "=" * 110)
print("Boundary epsilon summary")
print("=" * 110)

print(
    f"{'Iext [A]':>10} | "
    f"{'Tamb [C]':>10} | "
    f"{'Boundary [C]':>12} | "
    f"{'N used':>6} | "
    f"{'N excl.':>7} | "
    f"{'mean eps':>10} | "
    f"{'median eps':>10} | "
    f"{'var eps':>12} | "
    f"{'rel var [%]':>12} | "
    f"{'min eps':>10} | "
    f"{'max eps':>10}"
)

print("-" * 118)

for row in summary_rows:
    (
        Iext,
        Tamb_C_row,
        boundary_T,
        N,
        N_raw,
        N_excluded_soc,
        eps_mean,
        eps_median,
        eps_std,
        eps_var,
        eps_rel_var,
        eps_min,
        eps_max,
    ) = row

    print(
        f"{Iext:10.2f} | "
        f"{Tamb_C_row:10.0f} | "
        f"{boundary_T:12.0f} | "
        f"{int(N):6d} | "
        f"{int(N_excluded_soc):7d} | "
        f"{eps_mean:10.4f} | "
        f"{eps_median:10.4f} | "
        f"{eps_var:12.4e} | "
        f"{100.0 * eps_rel_var:12.2f} | "
        f"{eps_min:10.4f} | "
        f"{eps_max:10.4f}"
    )

print("=" * 110)


# ============================================================
# 14. Print simulation result summary
# ============================================================
print("\n" + "=" * 90)
print("Simulation summary")
print("=" * 90)

for Iext in Iext_cases:
    data = results[Iext]

    Tmax_map = data["Tmax_map"]
    reached_stop_map = data["reached_stop_map"].astype(bool)
    has_soc_empty_info = "reached_empty_map" in data
    has_final_soc_info = "final_SOC_map" in data
    reached_empty_map = get_reached_empty_map(data)
    margin_map = data["margin_map"]
    final_SOC_map = get_final_SOC_map(data)
    stop_reason_map = get_stop_reason_map(data)

    Z_C = Tmax_map - 273.15

    stable_thermal_direction = margin_map > 0
    unstable_thermal_direction = margin_map <= 0

    print(f"\nIext = {Iext:.2f} A, Tamb = {Tamb_C:.0f} C")
    print(f"Grid size: {data['R_grid'].shape[0]} x {data['R_grid'].shape[1]} = {data['R_grid'].size}")
    print(f"min Tmax = {np.nanmin(Z_C):.2f} C")
    print(f"max Tmax = {np.nanmax(Z_C):.2f} C")
    print(f"Reached 120 C cutoff: {np.sum(reached_stop_map)} points")

    if has_soc_empty_info:
        soc_depleted_before_120 = reached_empty_map & (~reached_stop_map)
        print(f"Reached SOC empty before 120 C: {np.sum(soc_depleted_before_120)} points")
        print(f"Finished at t_end: {np.sum((~reached_stop_map) & (~reached_empty_map))} points")
    else:
        print("Reached SOC empty before 120 C: not available in this loaded npz file")
        print("Finished at t_end without SOC depletion: not available in this loaded npz file")
    print(f"lambda2(Tmax) < 0: {np.sum(stable_thermal_direction)} points")
    print(f"lambda2(Tmax) >= 0: {np.sum(unstable_thermal_direction)} points")

    if has_final_soc_info and np.any(np.isfinite(final_SOC_map)):
        print(f"min final SOC = {np.nanmin(final_SOC_map):.4f}")
        print(f"max final SOC = {np.nanmax(final_SOC_map):.4f}")
    else:
        print("final SOC map is not available for this loaded file.")

print("\nAll done.")