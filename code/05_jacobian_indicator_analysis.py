import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# 1. Basic parameters
# ============================================================
m = 0.720
Cp = 1100.0
Qcap = 25.0          # Ah
Rcell = 0.002
Rgas = 8.314
Tamb = 313.15        # K
Iext = 5.0
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

# anode reaction
c_an = 1.0
m_an = 100.58
H_an = 1714.0
A_an = 0.038
Ea_an = 33000.0

g_z_anode = np.exp(-c_sei / c_sei_0_ref)

# Temperature thresholds
T_alarm = 343.15     # 70 C
T_limit = 393.15     # 120 C

# ============================================================
# 2. Model functions
# ============================================================
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


def lambda_2(T, Ueq):
    """
    Thermal-direction Jacobian eigenvalue.
    lambda_2 < 0: local temperature perturbation decays.
    lambda_2 > 0: local temperature perturbation grows.
    """
    return (dQ_chem_dT(T) - Ueq) / (m * Cp)


# ============================================================
# 3. Time-scale ratio
# ============================================================
S_dense = np.linspace(0.0, 1.0, 3001)
U_dense = U_OCV(S_dense)
Uref = np.max(U_dense)

def epsilon_value(Risc, Ueq):
    return (m * Cp * Uref) / (
        Ueq * 3600.0 * Qcap * (Rcell + Risc)
    )


# ============================================================
# 4. Full two-dimensional ODE system
# ============================================================
def rhs(t, y, Risc, Ueq):
    S, T = y
    U = U_OCV(S)

    I_isc = U / (Rcell + Risc)

    # SOC equation
    dSdt = -(Iext + I_isc) / (3600.0 * Qcap)

    # Thermal equation
    Q_isc = zeta * U**2 / (Rcell + Risc)
    Q_ext = Iext**2 * Rcell

    dTdt = (
        Q_ext
        + Q_isc
        + Q_chem(T)
        - Ueq * (T - Tamb)
    ) / (m * Cp)

    return [dSdt, dTdt]


# ============================================================
# 5. Choose one epsilon > 0.1 parameter point
# ============================================================
Risc = 0.10
Ueq = 0.2

eps = epsilon_value(Risc, Ueq)
print(f"Selected point:")
print(f"Risc = {Risc:.4f} Ohm")
print(f"Ueq  = {Ueq:.4f} W/K")
print(f"epsilon = {eps:.4f}")

if eps <= 0.1:
    print("Warning: this point does not satisfy epsilon > 0.1.")


# ============================================================
# 6. Simulation
# ============================================================
S0 = 1.0
T0 = Tamb
t_end = 20000.0

def event_T_limit(t, y):
    return y[1] - T_limit

event_T_limit.terminal = True
event_T_limit.direction = 1

def event_SOC_empty(t, y):
    return y[0]

event_SOC_empty.terminal = True
event_SOC_empty.direction = -1

sol = solve_ivp(
    fun=lambda t, y: rhs(t, y, Risc, Ueq),
    t_span=(0.0, t_end),
    y0=[S0, T0],
    method="DOP853",
    rtol=1e-7,
    atol=1e-9,
    max_step=20.0,
    dense_output=False,
    events=[event_T_limit, event_SOC_empty]
)

t = sol.t
S = sol.y[0]
T = sol.y[1]

T_C = T - 273.15

lam2 = lambda_2(T, Ueq)

stable_mask = lam2 < 0
unstable_mask = lam2 >= 0

print("\nSimulation result:")
print(f"final time = {t[-1]:.2f} s")
print(f"max T = {np.max(T_C):.2f} C")
print(f"min lambda_2 = {np.min(lam2):.4e} 1/s")
print(f"max lambda_2 = {np.max(lam2):.4e} 1/s")
print(f"lambda_2 < 0 for all time? {np.all(stable_mask)}")

if len(sol.t_events[0]) > 0:
    print("The trajectory reached 120 C and the simulation stopped.")
elif len(sol.t_events[1]) > 0:
    print("SOC reached zero and the simulation stopped.")
else:
    print("The simulation reached t_end.")


# ============================================================
# 7. Plot S-T vector field and lambda_2 curve
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(10.5, 8))

# ------------------------------------------------------------
# S-T phase plane vector field
# ------------------------------------------------------------
S_min = 0.0
S_max = 1.0

T_plot_min_C = min(Tamb - 273.15, np.min(T_C)) - 5.0
T_plot_max_C = max(125.0, np.max(T_C) + 5.0)

S_vec = np.linspace(S_min + 0.02, S_max - 0.02, 24)
T_vec_C = np.linspace(T_plot_min_C, T_plot_max_C, 24)

S_grid, T_grid_C = np.meshgrid(S_vec, T_vec_C)
T_grid = T_grid_C + 273.15

dS_grid, dT_grid = rhs(0.0, [S_grid, T_grid], Risc, Ueq)

# Normalize arrows for visualization only.
# The physical direction still comes from the ODE system.
dS_plot = dS_grid / (S_max - S_min)
dT_plot = dT_grid / (T_plot_max_C - T_plot_min_C)

arrow_norm = np.sqrt(dS_plot**2 + dT_plot**2)
arrow_norm[arrow_norm == 0] = 1.0

arrow_len_S = 0.035 * (S_max - S_min)
arrow_len_T = 0.035 * (T_plot_max_C - T_plot_min_C)

dS_arrow = dS_plot / arrow_norm * arrow_len_S
dT_arrow = dT_plot / arrow_norm * arrow_len_T

axes[0].quiver(
    S_grid,
    T_grid_C,
    dS_arrow,
    dT_arrow,
    angles="xy",
    scale_units="xy",
    scale=1.0,
    width=0.0028,
    color="gray",
    alpha=0.75
)

axes[0].plot(
    S,
    T_C,
    color="black",
    linewidth=2.2,
    label=r"trajectory from $(S_0,T_0)$"
)

axes[0].scatter(
    S0,
    T0 - 273.15,
    s=65,
    color="#2a9d8f",
    edgecolor="black",
    zorder=5,
    label=r"initial condition"
)

axes[0].scatter(
    S[-1],
    T_C[-1],
    s=65,
    color="#e76f51",
    edgecolor="black",
    zorder=5,
    label=r"final point"
)

axes[0].axhline(
    70,
    color="orange",
    linestyle="--",
    linewidth=1.6,
    label=r"$70^\circ C$"
)

axes[0].axhline(
    120,
    color="red",
    linestyle="--",
    linewidth=1.6,
    label=r"$120^\circ C$"
)

axes[0].set_xlim(S_min, S_max)
axes[0].set_ylim(T_plot_min_C, T_plot_max_C)

axes[0].set_xlabel(r"SOC $S$", fontsize=12)
axes[0].set_ylabel(r"Temperature [$^\circ$C]", fontsize=12)

axes[0].set_title(
    rf"Vector field in the $S$-$T$ plane, "
    rf"$R_{{isc}}={Risc:.2f}\Omega$, $U_{{eq}}={Ueq:.2f}$ W/K, $\epsilon={eps:.3f}$",
    fontsize=13
)

axes[0].grid(True, linestyle="--", alpha=0.35)
axes[0].legend(fontsize=9, loc="upper right")


# ------------------------------------------------------------
# lambda_2 curve
# ------------------------------------------------------------
axes[1].plot(
    t,
    lam2,
    color="black",
    linewidth=2.0,
    label=r"$\lambda_2(T(t))$"
)

axes[1].axhline(
    0,
    color="red",
    linestyle="--",
    linewidth=1.8,
    label=r"$\lambda_2=0$"
)

axes[1].fill_between(
    t,
    lam2,
    0,
    where=(lam2 < 0),
    alpha=0.25,
    color="#2a9d8f",
    label=r"local damping"
)

axes[1].fill_between(
    t,
    lam2,
    0,
    where=(lam2 >= 0),
    alpha=0.25,
    color="#e76f51",
    label=r"local amplification"
)

axes[1].set_xlabel(r"Time [s]", fontsize=12)
axes[1].set_ylabel(r"$\lambda_2$ [1/s]", fontsize=12)
axes[1].grid(True, linestyle="--", alpha=0.35)
axes[1].legend(fontsize=9, loc="best")

plt.tight_layout()
plt.show()