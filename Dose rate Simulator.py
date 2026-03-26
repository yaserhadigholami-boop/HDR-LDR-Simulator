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
Tphys_Cu = 12.7

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
# Find crossing point
# -----------------------------
def find_crossing(t, curve, threshold):
    diff = curve - threshold
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        return None, None
    i = idx[0]
    return t[i], i

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
Tbio_Cu = st.sidebar.slider("Cu Tbio (h)", 1.0, 200.0, 50.0)

S = 0.05

# -----------------------------
# Compute A0
# -----------------------------
A0_Lu = compute_A0_for_target(D_target, Tphys_Lu, Tbio_Lu, S)
A0_Cu = compute_A0_for_target(D_target, Tphys_Cu, Tbio_Cu, S)

# -----------------------------
# Dose calculations
# -----------------------------
Ddot_Lu, A_Lu, Rcrit, total_Lu, eff_Lu, waste_Lu, eff_ratio_Lu = compute_dose(
    A0_Lu, Tphys_Lu, Tbio_Lu, S, alpha, Tav
)

Ddot_Cu, A_Cu, Rcrit, total_Cu, eff_Cu, waste_Cu, eff_ratio_Cu = compute_dose(
    A0_Cu, Tphys_Cu, Tbio_Cu, S, alpha, Tav
)

# -----------------------------
# Normalisation for plotting only
# -----------------------------
max_val = max(Ddot_Lu.max(), Ddot_Cu.max())
Ddot_Lu_n = Ddot_Lu / max_val
Ddot_Cu_n = Ddot_Cu / max_val
Rcrit_n = Rcrit / max_val

# -----------------------------
# Crossing points
# -----------------------------
t_Lu, idx_Lu = find_crossing(t_global, Ddot_Lu, Rcrit)
t_Cu, idx_Cu = find_crossing(t_global, Ddot_Cu, Rcrit)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t_global, Ddot_Lu_n, label="177Lu", linewidth=2, color='green')
ax.plot(t_global, Ddot_Cu_n, label="64Cu", linewidth=2, color='red')

ax.axhline(Rcrit_n, linestyle='--', linewidth=2, label="Rcrit")

# Shading
mask_Lu = Ddot_Lu_n > Rcrit_n
mask_Cu = Ddot_Cu_n > Rcrit_n

# Not effective (below Rcrit)
ax.fill_between(t_global, 0, np.minimum(Ddot_Lu_n, Rcrit_n), alpha=0.08)
ax.fill_between(t_global, 0, np.minimum(Ddot_Cu_n, Rcrit_n), alpha=0.08)

# Effective (above Rcrit)
ax.fill_between(t_global, Rcrit_n, Ddot_Lu_n, where=mask_Lu, alpha=0.25)
ax.fill_between(t_global, Rcrit_n, Ddot_Cu_n, where=mask_Cu, alpha=0.25)

# Crossing markers
if idx_Lu is not None:
    ax.scatter(t_Lu, Rcrit_n)
    ax.annotate("Lu", (t_Lu, Rcrit_n), xytext=(10, 10), textcoords='offset points')

if idx_Cu is not None:
    ax.scatter(t_Cu, Rcrit_n)
    ax.annotate("Cu", (t_Cu, Rcrit_n), xytext=(10, -15), textcoords='offset points')

# -----------------------------
# Text boxes
# -----------------------------
text_Lu = (
    f"177Lu\n"
    f"A0: {A0_Lu:.2f} MBq\n"
    f"Wasted: {waste_Lu:.2f} Gy\n"
    f"Efficiency: {eff_ratio_Lu*100:.1f}%"
)

text_Cu = (
    f"64Cu\n"
    f"A0: {A0_Cu:.2f} MBq\n"
    f"Wasted: {waste_Cu:.2f} Gy\n"
    f"Efficiency: {eff_ratio_Cu*100:.1f}%"
)

peak_ratio = Ddot_Cu.max() / Ddot_Lu.max()
text_global = f"Peak Ratio (Cu/Lu): {peak_ratio:.2f}×"

ax.text(0.02, 0.95, text_Lu, transform=ax.transAxes,
        fontsize=10, va='top', bbox=dict(boxstyle="round", alpha=0.2))

ax.text(0.02, 0.65, text_Cu, transform=ax.transAxes,
        fontsize=10, va='top', bbox=dict(boxstyle="round", alpha=0.2))

ax.text(0.65, 0.95, text_global, transform=ax.transAxes,
        fontsize=11, va='top', bbox=dict(boxstyle="round", alpha=0.3))

# -----------------------------
# Axis
# -----------------------------
ax.set_xlim(0, t_max)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Normalised Dose Rate")
ax.set_title("Dose Rate Comparison (Same Total Dose)")
ax.legend()
ax.grid(True)

st.pyplot(fig)
