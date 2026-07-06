import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. Parameters
# ============================================================
m = 0.720
Cp = 1100.0
Qcap = 25.0          # Ah
Rcell = 0.002
Rgas = 8.314
Tamb = 313.15        # K, 40 C
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

# Output folder
out_dir = Path("Qchem_time_scale_check")
out_dir.mkdir(exist_ok=True)


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


def Q_isc_at_Risc(Risc, S=1.0):
    U = U_OCV(S)
    return zeta * U**2 / (Rcell + Risc)


# ============================================================
# 3. Temperature range
# ============================================================
T_C = np.linspace(40.01, 120.0, 800)
T_K = T_C + 273.15

Qchem_vals = Q_chem(T_K)
dQchem_vals = dQ_chem_dT(T_K)

Qext = Iext**2 * Rcell

# Representative values
Risc_values = [0.10, 0.20, 1.00]
Ueq_values = [0.20, 0.60, 1.50]


# ============================================================
# 4. Print useful numerical values
# ============================================================
print("Chemical heat generation and its temperature sensitivity:")
for TC in [40, 70, 120]:
    TK = TC + 273.15
    print(
        f"T = {TC:>3.0f} C: "
        f"Q_chem = {Q_chem(TK):.6f} W, "
        f"dQ_chem/dT = {dQ_chem_dT(TK):.6f} W/K"
    )

print("\nRepresentative internal short circuit heat at S = 1:")
for Risc in Risc_values:
    print(
        f"Risc = {Risc:.2f} Ohm: "
        f"Q_isc = {Q_isc_at_Risc(Risc, S=1.0):.6f} W"
    )

print(f"\nExternal current heat:")
print(f"Iext = {Iext:.1f} A: Q_ext = {Qext:.6f} W")


# ============================================================
# 5. Figure 1: heat power comparison
#    Q_chem should be compared with heat power terms, not U_eq directly.
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5.8))

ax.plot(
    T_C,
    Qchem_vals,
    linewidth=2.2,
    label=r"$Q_{chem}(T)$"
)

# Internal short circuit heat levels
for Risc in Risc_values:
    ax.axhline(
        Q_isc_at_Risc(Risc, S=1.0),
        linestyle="--",
        linewidth=1.6,
        label=rf"$Q_{{isc}}$, $R_{{isc}}={Risc:.2f}\,\Omega$, $S=1$"
    )

# External current heat
ax.axhline(
    Qext,
    linestyle=":",
    linewidth=1.8,
    label=rf"$Q_{{ext}}=I_{{ext}}^2R_{{cell}}$, $I_{{ext}}={Iext:.1f}\,\mathrm{{A}}$"
)

# Cooling power curves Ueq(T-Tamb)
for Ueq in [0.20, 0.60]:
    Qdiss = Ueq * (T_K - Tamb)
    Qdiss_plot = np.maximum(Qdiss, 1e-8)

    ax.plot(
        T_C,
        Qdiss_plot,
        linestyle="-.",
        linewidth=1.8,
        label=rf"$U_{{eq}}(T-T_{{amb}})$, $U_{{eq}}={Ueq:.2f}\,\mathrm{{W/K}}$"
    )

ax.axvline(
    70,
    linestyle=":",
    linewidth=1.5,
    label=r"$70^\circ\mathrm{C}$"
)

ax.axvline(
    120,
    linestyle=":",
    linewidth=1.5,
    label=r"$120^\circ\mathrm{C}$"
)

ax.set_yscale("log")
ax.set_xlabel(r"Temperature $T$ [$^\circ$C]", fontsize=12)
ax.set_ylabel(r"Heat power [W]", fontsize=12)
ax.set_title(
    r"Comparison of $Q_{chem}(T)$ with electrical heat and cooling power",
    fontsize=13
)
ax.grid(True, which="both", linestyle="--", alpha=0.35)
ax.legend(fontsize=8.5, loc="best")

plt.tight_layout()
plt.savefig(out_dir / "Qchem_heat_power_comparison.pdf", bbox_inches="tight")
plt.savefig(out_dir / "Qchem_heat_power_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 6. Figure 2: local thermal stability comparison
#    dQ_chem/dT can be compared directly with U_eq because both are W/K.
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5.8))

ax.plot(
    T_C,
    dQchem_vals,
    linewidth=2.2,
    label=r"$Q'_{chem}(T)$"
)

for Ueq in Ueq_values:
    ax.axhline(
        Ueq,
        linestyle="--",
        linewidth=1.7,
        label=rf"$U_{{eq}}={Ueq:.2f}\,\mathrm{{W/K}}$"
    )

ax.axvline(
    70,
    linestyle=":",
    linewidth=1.5,
    label=r"$70^\circ\mathrm{C}$"
)

ax.axvline(
    120,
    linestyle=":",
    linewidth=1.5,
    label=r"$120^\circ\mathrm{C}$"
)

ax.set_yscale("log")
ax.set_xlabel(r"Temperature $T$ [$^\circ$C]", fontsize=12)
ax.set_ylabel(r"$Q'_{chem}(T)$ and $U_{eq}$ [W/K]", fontsize=12)
ax.set_title(
    r"Temperature sensitivity of chemical heat compared with $U_{eq}$",
    fontsize=13
)
ax.grid(True, which="both", linestyle="--", alpha=0.35)
ax.legend(fontsize=9, loc="best")

plt.tight_layout()
plt.savefig(out_dir / "dQchem_dT_vs_Ueq.pdf", bbox_inches="tight")
plt.savefig(out_dir / "dQchem_dT_vs_Ueq.png", dpi=300, bbox_inches="tight")
plt.show()