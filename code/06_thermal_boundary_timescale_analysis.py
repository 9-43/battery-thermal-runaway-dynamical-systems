import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


# ============================================================
# 1. Load saved simulation results
# ============================================================
script_dir = Path(__file__).resolve().parent

data_path = Path(
    r"D:\files\毕业设计\corrected_Iext_SOCincluded\Tmax_sweep_Iext5A_SOCincluded_Tamb40C_NR70_NU70.npz"
)

data = np.load(data_path)

Risc_array = data["Risc_array"]
Ueq_array = data["Ueq_array"]
R_grid = data["R_grid"]
U_grid = data["U_grid"]
Tmax_map = data["Tmax_map"]

has_reached_map = "reached_stop_map" in data.files
if has_reached_map:
    reached_stop_map = data["reached_stop_map"]

Tmax_C = Tmax_map - 273.15

output_dir = data_path.parent


# ============================================================
# 2. Parameters needed for analytical epsilon contours
# ============================================================
m = 0.720
Cp = 1100.0
Qcap = 25.0
Rcell = 0.002
Iext = 5.0

# OCV polynomial
p = [-2.622, 5.761, -4.145, 2.062, 3.131]

def U_OCV(S):
    S = np.clip(S, 0.0, 1.0)
    return np.polyval(p, S)

S_dense = np.linspace(0.0, 1.0, 3001)
U_dense = U_OCV(S_dense)
Uref = np.max(U_dense)


# ============================================================
# 3. Correct epsilon formula with external current
# ============================================================
def epsilon_from_formula(Risc, Ueq):
    return (m * Cp / (3600.0 * Qcap * Ueq)) * (
        Iext + Uref / (Rcell + Risc)
    )

def Ueq_from_epsilon(eps_value, Risc):
    return (m * Cp / (3600.0 * Qcap * eps_value)) * (
        Iext + Uref / (Rcell + Risc)
    )

# 不再直接使用 npz 里面可能保存过的旧 eps_grid
eps_grid = epsilon_from_formula(R_grid, U_grid)


# ============================================================
# 4. Helper: sample epsilon along a contour
# ============================================================
eps_interp = RegularGridInterpolator(
    points=(Ueq_array, Risc_array),
    values=eps_grid,
    bounds_error=False,
    fill_value=np.nan
)

def sample_epsilon_on_contour(contour_set):
    R_values = []
    U_values = []
    eps_values = []

    if len(contour_set.allsegs) == 0:
        return np.array([]), np.array([]), np.array([])

    for seg in contour_set.allsegs[0]:
        if len(seg) < 2:
            continue

        R_seg = seg[:, 0]
        U_seg = seg[:, 1]

        pts = np.column_stack([U_seg, R_seg])
        eps_seg = eps_interp(pts)

        mask = (
            np.isfinite(R_seg)
            & np.isfinite(U_seg)
            & np.isfinite(eps_seg)
        )

        R_values.extend(R_seg[mask])
        U_values.extend(U_seg[mask])
        eps_values.extend(eps_seg[mask])

    R_values = np.array(R_values)
    U_values = np.array(U_values)
    eps_values = np.array(eps_values)

    # 按 Risc 排序，后面采样图横轴就是 Risc
    if len(R_values) > 0:
        order = np.argsort(R_values)
        R_values = R_values[order]
        U_values = U_values[order]
        eps_values = eps_values[order]

    return R_values, U_values, eps_values


def epsilon_statistics(eps_values):
    if len(eps_values) == 0:
        return {
            "N": 0,
            "mean": np.nan,
            "median": np.nan,
            "variance": np.nan,
            "std": np.nan,
            "relative_variation": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    eps_mean = np.mean(eps_values)
    eps_std = np.std(eps_values)
    eps_var = np.var(eps_values)

    return {
        "N": len(eps_values),
        "mean": eps_mean,
        "median": np.median(eps_values),
        "variance": eps_var,
        "std": eps_std,
        "relative_variation": 100.0 * eps_std / eps_mean,
        "min": np.min(eps_values),
        "max": np.max(eps_values),
    }


def print_stats(label, stats, R_values, U_values):
    print("\n" + "=" * 70)
    print(f"{label} boundary time-scale ratio statistics")
    print("=" * 70)
    print(f"N sampled points        = {stats['N']}")
    print(f"epsilon mean            = {stats['mean']:.6f}")
    print(f"epsilon median          = {stats['median']:.6f}")
    print(f"epsilon variance        = {stats['variance']:.6e}")
    print(f"epsilon std             = {stats['std']:.6f}")
    print(f"relative variation      = {stats['relative_variation']:.2f}%")
    print(f"epsilon min             = {stats['min']:.6f}")
    print(f"epsilon max             = {stats['max']:.6f}")

    if len(R_values) > 0:
        print(f"Risc range              = [{np.min(R_values):.6g}, {np.max(R_values):.6g}] Ohm")
        print(f"Ueq range               = [{np.min(U_values):.6g}, {np.max(U_values):.6g}] W/K")


def save_samples_csv(path, R_values, U_values, eps_values):
    if len(eps_values) == 0:
        return

    out = np.column_stack([R_values, U_values, eps_values])
    np.savetxt(
        path,
        out,
        delimiter=",",
        header="Risc_Ohm,Ueq_W_per_K,epsilon",
        comments="",
        fmt="%.12e"
    )


# ============================================================
# 5. Soft but clearer colormap
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


# ============================================================
# 6. Plot background and temperature boundaries
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))

cf = ax.contourf(
    R_grid,
    U_grid,
    Tmax_C,
    levels=np.linspace(40, 120, 33),
    cmap=soft_temp_cmap,
    extend="max"
)

cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r"$T_{\max}$ [$^\circ$C]", fontsize=12)


# ------------------------------------------------------------
# Temperature boundary: Tmax = 70 C
# ------------------------------------------------------------
cs70 = ax.contour(
    R_grid,
    U_grid,
    Tmax_C,
    levels=[70],
    colors=["black"],
    linewidths=2.4,
    linestyles="-"
)

# 不用 inline=True，避免文字把黑线切断
ax.clabel(
    cs70,
    fmt={70: r"$T_{\max}=70^\circ C$"},
    fontsize=10,
    inline=False
)

R70, U70, eps70_values = sample_epsilon_on_contour(cs70)
stats70 = epsilon_statistics(eps70_values)
eps70_mean = stats70["mean"]


# ------------------------------------------------------------
# Temperature boundary: Tmax = 120 C
# Use reached_stop_map if available because Tmax may be capped at 120 C
# ------------------------------------------------------------
if has_reached_map:
    cs120 = ax.contour(
        R_grid,
        U_grid,
        reached_stop_map.astype(float),
        levels=[0.5],
        colors=["#b00020"],
        linewidths=2.8,
        linestyles="-"
    )

    ax.clabel(
        cs120,
        fmt={0.5: r"$T_{\max}=120^\circ C$"},
        fontsize=10,
        inline=False
    )
else:
    cs120 = ax.contour(
        R_grid,
        U_grid,
        Tmax_C,
        levels=[120],
        colors=["#b00020"],
        linewidths=2.8,
        linestyles="-"
    )

    ax.clabel(
        cs120,
        fmt={120: r"$T_{\max}=120^\circ C$"},
        fontsize=10,
        inline=False
    )

R120, U120, eps120_values = sample_epsilon_on_contour(cs120)
stats120 = epsilon_statistics(eps120_values)
eps120_mean = stats120["mean"]


# ============================================================
# 7. Draw corresponding mean epsilon contours analytically
# ============================================================
R_line = np.logspace(
    np.log10(np.min(Risc_array)),
    np.log10(np.max(Risc_array)),
    2000
)

ymin, ymax = 0.05, 10.0

def plot_epsilon_line(eps_value, color, linestyle, linewidth):
    if not np.isfinite(eps_value):
        return

    U_line = Ueq_from_epsilon(eps_value, R_line)

    valid = (U_line >= ymin) & (U_line <= ymax)

    ax.plot(
        R_line[valid],
        U_line[valid],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth
    )

    if np.any(valid):
        valid_idx = np.where(valid)[0]
        mid = valid_idx[len(valid_idx) // 2]

        ax.text(
            R_line[mid],
            U_line[mid] * 1.08,
            rf"$\bar{{\epsilon}}={eps_value:.2f}$",
            color=color,
            fontsize=10,
            ha="center",
            va="bottom",
            rotation=-25
        )


if np.isfinite(eps70_mean):
    plot_epsilon_line(
        eps70_mean,
        color="#0077b6",
        linestyle="--",
        linewidth=2.4
    )

if np.isfinite(eps120_mean):
    plot_epsilon_line(
        eps120_mean,
        color="#7b2cbf",
        linestyle="--",
        linewidth=2.6
    )


# ============================================================
# 8. Figure settings
# ============================================================
ax.set_xscale("log")
ax.set_xlim(0.01, 10)
ax.set_ylim(ymin, ymax)

ax.set_xlabel(r"Internal Short Resistance $R_{isc}$ [$\Omega$]", fontsize=13)
ax.set_ylabel(r"Cooling Coefficient $U_{eq}$ [W/K]", fontsize=13)

# 保持原来的标题，不加外电流
ax.set_title(
    r"Simulated Thermal Boundaries and Time-scale Ratio Boundaries",
    fontsize=14
)

ax.grid(True, which="both", linestyle="--", alpha=0.30)

legend_elements = [
    Line2D([0], [0], color="black", lw=2.4, linestyle="-",
           label=r"$T_{\max}=70^\circ C$"),
    Line2D([0], [0], color="#b00020", lw=2.8, linestyle="-",
           label=r"$T_{\max}=120^\circ C$"),
]

if np.isfinite(eps70_mean):
    legend_elements.append(
        Line2D([0], [0], color="#0077b6", lw=2.4, linestyle="--",
               label=rf"$\bar{{\epsilon}}={eps70_mean:.2f}$")
    )

if np.isfinite(eps120_mean):
    legend_elements.append(
        Line2D([0], [0], color="#7b2cbf", lw=2.6, linestyle="--",
               label=rf"$\bar{{\epsilon}}={eps120_mean:.2f}$")
    )

ax.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=9.5,
    framealpha=0.95
)

plt.tight_layout()


# ============================================================
# 9. Save main figure
# ============================================================
fig_path_pdf = output_dir / "Simulated_Thermal_Boundaries_and_Time_scale_Ratio_Boundaries.pdf"
fig_path_png = output_dir / "Simulated_Thermal_Boundaries_and_Time_scale_Ratio_Boundaries.png"

plt.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# 10. Plot epsilon values sampled along 70 C and 120 C boundaries
# ============================================================
def plot_epsilon_samples_vs_Risc(R_values, eps_values, stats, temp_text, color, save_name):
    if len(eps_values) == 0:
        print(f"No epsilon samples found for {temp_text} boundary.")
        return

    fig_s, ax_s = plt.subplots(figsize=(9, 5))

    ax_s.plot(
        R_values,
        eps_values,
        color=color,
        linewidth=1.8,
        marker="o",
        markersize=3.0
    )

    ax_s.axhline(
        stats["mean"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=rf"mean $\bar{{\epsilon}}={stats['mean']:.4f}$"
    )

    ax_s.set_xscale("log")
    ax_s.set_xlabel(r"Internal Short Resistance $R_{isc}$ [$\Omega$]", fontsize=12)
    ax_s.set_ylabel(r"Time-scale ratio $\epsilon$", fontsize=12)

    ax_s.set_title(
        rf"Time scale ratio values sampled along the simulated {temp_text} thermal boundary.",
        fontsize=12
    )

    ax_s.grid(True, which="both", linestyle="--", alpha=0.35)

    # 图例只写平均值
    ax_s.legend(loc="best", fontsize=10)

    plt.tight_layout()

    fig_pdf = output_dir / f"{save_name}.pdf"
    fig_png = output_dir / f"{save_name}.png"

    plt.savefig(fig_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"\n{temp_text} sampled epsilon figure saved to:")
    print(fig_pdf)
    print(fig_png)


plot_epsilon_samples_vs_Risc(
    R_values=R70,
    eps_values=eps70_values,
    stats=stats70,
    temp_text=r"$70^\circ\mathrm{C}$",
    color="#0077b6",
    save_name="Time_scale_ratio_values_sampled_along_70C_thermal_boundary"
)

plot_epsilon_samples_vs_Risc(
    R_values=R120,
    eps_values=eps120_values,
    stats=stats120,
    temp_text=r"$120^\circ\mathrm{C}$",
    color="#7b2cbf",
    save_name="Time_scale_ratio_values_sampled_along_120C_thermal_boundary"
)


# ============================================================
# 11. Save sampled data
# ============================================================
csv70_path = output_dir / "epsilon_samples_along_70C_boundary.csv"
csv120_path = output_dir / "epsilon_samples_along_120C_boundary.csv"

save_samples_csv(csv70_path, R70, U70, eps70_values)
save_samples_csv(csv120_path, R120, U120, eps120_values)


# ============================================================
# 12. Print and save statistics
# ============================================================
print("\nBoundary epsilon summary:")
print(f"Loaded file: {data_path}")
print(f"Iext = {Iext:.4f} A")
print(f"Uref = {Uref:.6f} V")

print_stats("70 C", stats70, R70, U70)
print_stats("120 C", stats120, R120, U120)

summary_path = output_dir / "epsilon_boundary_statistics.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Boundary epsilon statistics\n")
    f.write("===========================\n\n")
    f.write(f"Loaded file: {data_path}\n")
    f.write(f"Iext = {Iext:.6f} A\n")
    f.write(f"Uref = {Uref:.10f} V\n\n")

    for label, stats, R_values, U_values in [
        ("70 C", stats70, R70, U70),
        ("120 C", stats120, R120, U120),
    ]:
        f.write(f"{label} boundary time-scale ratio statistics\n")
        f.write("-" * 60 + "\n")
        f.write(f"N sampled points        = {stats['N']}\n")
        f.write(f"epsilon mean            = {stats['mean']:.10f}\n")
        f.write(f"epsilon median          = {stats['median']:.10f}\n")
        f.write(f"epsilon variance        = {stats['variance']:.10e}\n")
        f.write(f"epsilon std             = {stats['std']:.10f}\n")
        f.write(f"relative variation      = {stats['relative_variation']:.6f}%\n")
        f.write(f"epsilon min             = {stats['min']:.10f}\n")
        f.write(f"epsilon max             = {stats['max']:.10f}\n")

        if len(R_values) > 0:
            f.write(f"Risc range              = [{np.min(R_values):.10e}, {np.max(R_values):.10e}] Ohm\n")
            f.write(f"Ueq range               = [{np.min(U_values):.10e}, {np.max(U_values):.10e}] W/K\n")

        f.write("\n")

print("\nMain figure saved to:")
print(fig_path_pdf)
print(fig_path_png)

print("\nSampled epsilon data saved to:")
print(csv70_path)
print(csv120_path)

print("\nStatistics saved to:")
print(summary_path)