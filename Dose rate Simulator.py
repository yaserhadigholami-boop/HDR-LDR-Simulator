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
# Bi-exponential activity (rise + tail)
# -----------------------------
def activity_curve(A0, Tphys, Tbio, t_array):
    lambda_p = np.log(2)/Tphys
    lambda_b = np.log(2)/Tbio
    return A0 * (lambda_b / (lambda_b - lambda_p)) * (
        np.exp(-lambda_p*t_array) - np.exp(-lambda_b*t_array)
    )

# -----------------------------
# Compute dose (FULL version)
# -----------------------------
def compute_dose(A0, Tphys, Tbio, S, alpha, Tav, t_array=None):
    if t_array is None:
        t_array = t_global

    A = activity_curve(A0, Tphys, Tbio, t_array)
    Ddot = S * A
    Rcrit = 0.693/(alpha*Tav)

    effective_mask = Ddot > Rcrit
    effective = trapz_manual(Ddot[effective_mask], t_array[effective_mask]) if effective_mask.any() else 0
    wasted = trapz_manual(np.minimum(Ddot, Rcrit), t_array)
    total = trapz_manual(Ddot, t_array)
    efficiency = effective/total if total>0 else 0

    return Ddot, A, Rcrit, total, effective, wasted, efficiency

# -----------------------------
# Compute required A0
# -----------------------------
def compute_A0_for_target(D_target, Tphys, Tbio, S, t_array=None):
    if t_array is None:
        t_array = t_global

    A_norm = activity_curve(1.0, Tphys, Tbio, t_array)
    integral = trapz_manual(S*A_norm, t_array)
    return D_target / integral

# -----------------------------
# Find crossing
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
# Correct A0 (THIS is the key)
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
# Normalisation (visual ONLY)
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

A_cross_Lu = A_Lu[idx_Lu] if idx_Lu is not None else None
A_cross_Cu = A_Cu[idx_Cu] if idx_Cu is not None else None

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t_global, Ddot_Lu_n, label="177Lu", linewidth=2)
ax.plot(t_global, Ddot_Cu_n, label="64Cu", linewidth=2)

ax.axhline(Rcrit_n, linestyle='--', linewidth=2, label="Rcrit")

# NOT effective (below Rcrit)
ax.fill_between(
    t_global,
    0,
    np.minimum(Ddot_Lu_n, Rcrit_n),
    alpha=0.08
)

ax.fill_between(
    t_global,
    0,
    np.minimum(Ddot_Cu_n, Rcrit_n),
    alpha=0.08
)

# Effective
mask_Lu = Ddot_Lu_n > Rcrit_n
mask_Cu = Ddot_Cu_n > Rcrit_n

ax.fill_between(t_global, Rcrit_n, Ddot_Lu_n, where=mask_Lu, alpha=0.25)
ax.fill_between(t_global, Rcrit_n, Ddot_Cu_n, where=mask_Cu, alpha=0.25)

# Crossing markers
if idx_Lu is not None:
    ax.scatter(t_Lu, Rcrit_n)
    ax.annotate("Lu", (t_Lu, Rcrit_n), xytext=(10, 10), textcoords='offset points')

if idx_Cu is not None:
    ax.scatter(t_Cu, Rcrit_n)
    ax.annotate("Cu", (t_Cu, Rcrit_n), xytext=(10, -15), textcoords='offset points')

# Axis
ax.set_xlim(0, t_max)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Normalised Dose Rate")
ax.set_title("Dose Rate Comparison (Same Total Dose)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Output
# -----------------------------
st.markdown("### Correct Activities (Key Point)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("177Lu")
    st.write(f"A0: {A0_Lu:.2f} MBq")
    if t_Lu:
        st.write(f"Cross time: {t_Lu:.2f} h")
        st.write(f"Activity at crossing: {A_cross_Lu:.2f} MBq")

with col2:
    st.subheader("64Cu")
    st.write(f"A0: {A0_Cu:.2f} MBq")
    if t_Cu:
        st.write(f"Cross time: {t_Cu:.2f} h")
        st.write(f"Activity at crossing: {A_cross_Cu:.2f} MBq")

# Peak ratio (VERY important for Dale)
peak_ratio = Ddot_Cu.max() / Ddot_Lu.max()
st.write(f"Peak dose rate ratio (Cu / Lu): {peak_ratio:.2f}×")
