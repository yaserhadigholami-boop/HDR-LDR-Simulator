import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(layout="wide")
st.title("Dose Rate Comparison (PERT Concept)")

# -----------------------------
# Time axis
# -----------------------------
t_max = 200
n_points = 2000
t_global = np.linspace(0, t_max, n_points)

# -----------------------------
# Physical half-lives (hours)
# -----------------------------
Tphys_Lu = 160
Tphys_Cu64 = 12.7
Tphys_Cu67 = 62.0  # approximate

# -----------------------------
# Trapezoid integration
# -----------------------------
def trapz_manual(y, x):
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)

# -----------------------------
# Activity curve (rise + tail)
# -----------------------------
def activity_curve(A0, Tphys, Tbio, t):
    lambda_p = np.log(2)/Tphys
    lambda_b = np.log(2)/Tbio
    return A0 * (lambda_b / (lambda_b - lambda_p)) * (
        np.exp(-lambda_p*t) - np.exp(-lambda_b*t)
    )

# -----------------------------
# Dose computation
# -----------------------------
def compute_dose(A0, Tphys, Tbio, S, alpha, Tav):
    A = activity_curve(A0, Tphys, Tbio, t_global)
    Ddot = S * A
    Rcrit = 0.693/(alpha*Tav)

    effective_mask = Ddot > Rcrit
    effective = trapz_manual(Ddot[effective_mask], t_global[effective_mask]) if effective_mask.any() else 0
    wasted = trapz_manual(np.minimum(Ddot, Rcrit), t_global)
    total = trapz_manual(Ddot, t_global)
    efficiency = effective / total if total > 0 else 0

    return Ddot, A, Rcrit, total, effective, wasted, efficiency

# -----------------------------
# Compute A0 for target dose
# -----------------------------
def compute_A0_for_target(D_target, Tphys, Tbio, S):
    A_norm = activity_curve(1.0, Tphys, Tbio, t_global)
    integral = trapz_manual(S * A_norm, t_global)
    return D_target / integral

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Global Control")
D_target = st.sidebar.slider("Target Dose (Gy)", 10.0, 200.0, 50.0)

st.sidebar.header("Biology (Shared)")
alpha = st.sidebar.slider("alpha (Gy⁻¹)", 0.05, 1.0, 0.3)
Tav = st.sidebar.slider("Repair Half-Time Tav (h)", 10.0, 200.0, 72.0)

st.sidebar.header("177Lu")
Tbio_Lu = st.sidebar.slider("Lu Tbio (h)", 1.0, 300.0, 200.0)

st.sidebar.header("64Cu")
Tbio_Cu64 = st.sidebar.slider("64Cu Tbio (h)", 1.0, 200.0, 50.0)

st.sidebar.header("67Cu")
Tbio_Cu67 = st.sidebar.slider("67Cu Tbio (h)", 1.0, 200.0, 50.0)

S = 0.05

# -----------------------------
# Compute A0
# -----------------------------
A0_Lu = compute_A0_for_target(D_target, Tphys_Lu, Tbio_Lu, S)
A0_Cu64 = compute_A0_for_target(D_target, Tphys_Cu64, Tbio_Cu64, S)
A0_Cu67 = compute_A0_for_target(D_target, Tphys_Cu67, Tbio_Cu67, S)

# -----------------------------
# Dose calculations
# -----------------------------
Ddot_Lu, A_Lu, Rcrit, total_Lu, eff_Lu, waste_Lu, eff_ratio_Lu = compute_dose(
    A0_Lu, Tphys_Lu, Tbio_Lu, S, alpha, Tav
)
Ddot_Cu64, A_Cu64, _, total_Cu64, eff_Cu64, waste_Cu64, eff_ratio_Cu64 = compute_dose(
    A0_Cu64, Tphys_Cu64, Tbio_Cu64, S, alpha, Tav
)
Ddot_Cu67, A_Cu67, _, total_Cu67, eff_Cu67, waste_Cu67, eff_ratio_Cu67 = compute_dose(
    A0_Cu67, Tphys_Cu67, Tbio_Cu67, S, alpha, Tav
)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Dose rate curves
ax.plot(t_global, Ddot_Lu, label="177Lu", linewidth=2)
ax.plot(t_global, Ddot_Cu64, label="64Cu", linewidth=2)
ax.plot(t_global, Ddot_Cu67, label="67Cu", linewidth=2)

# Rcrit
ax.axhline(Rcrit, linestyle='--', color='red', linewidth=2, label="Rcrit")

# Shading effective and wasted
for Ddot, color in zip([Ddot_Lu, Ddot_Cu64, Ddot_Cu67], ['green','orange','blue']):
    effective_mask = Ddot > Rcrit
    ax.fill_between(t_global, Rcrit, Ddot, where=effective_mask, color=color, alpha=0.25)
    ax.fill_between(t_global, 0, np.minimum(Ddot, Rcrit), color=color, alpha=0.08)

# -----------------------------
# Text boxes
# -----------------------------
texts = [
    f"177Lu\nA0: {A0_Lu:.2f} MBq\nWasted: {waste_Lu:.2f} Gy\nEfficiency: {eff_ratio_Lu*100:.1f}%",
    f"64Cu\nA0: {A0_Cu64:.2f} MBq\nWasted: {waste_Cu64:.2f} Gy\nEfficiency: {eff_ratio_Cu64*100:.1f}%",
    f"67Cu\nA0: {A0_Cu67:.2f} MBq\nWasted: {waste_Cu67:.2f} Gy\nEfficiency: {eff_ratio_Cu67*100:.1f}%"
]
for i, txt in enumerate(texts):
    ax.text(0.02, 0.95-0.25*i, txt, transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle="round", alpha=0.2))

# Peak ratio Cu64 / Cu67
peak_ratio = Ddot_Cu64.max() / Ddot_Cu67.max()
ax.text(0.65, 0.95, f"Peak Ratio (Cu64/Cu67): {peak_ratio:.2f}×", transform=ax.transAxes,
        fontsize=11, va='top', bbox=dict(boxstyle="round", alpha=0.3))

# -----------------------------
# Axis
# -----------------------------
ax.set_xlim(0, t_max)
ax.set_ylim(0, max(Ddot_Lu.max(), Ddot_Cu64.max(), Ddot_Cu67.max())*1.2)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Dose Rate (Gy/h)")
ax.set_title("Dose Rate Comparison (Same Total Dose)")
ax.legend()
ax.grid(True)

st.pyplot(fig)
