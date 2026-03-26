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
# Functions
# -----------------------------
def trapz_manual(y, x):
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)

def activity_curve(A0, Tphys, Tbio, t):
    lambda_p = np.log(2) / Tphys
    lambda_b = np.log(2) / Tbio
    return A0 * (lambda_b / (lambda_b - lambda_p)) * (
        np.exp(-lambda_p * t) - np.exp(-lambda_b * t)
    )

def compute_dose(A0, Tphys, Tbio, S, t):
    A = activity_curve(A0, Tphys, Tbio, t)
    return S * A, A

def compute_A0_for_target(D_target, Tphys, Tbio, S):
    A_norm = activity_curve(1.0, Tphys, Tbio, t_global)
    integral = trapz_manual(S * A_norm, t_global)
    return D_target / integral

def find_crossing(t, curve, threshold):
    diff = curve - threshold
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(sign_change) == 0:
        return None, None

    i = sign_change[0]
    t_cross = t[i]
    return t_cross, i

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
# Dose + activity
# -----------------------------
Ddot_Lu, A_Lu = compute_dose(A0_Lu, Tphys_Lu, Tbio_Lu, S, t_global)
Ddot_Cu, A_Cu = compute_dose(A0_Cu, Tphys_Cu, Tbio_Cu, S, t_global)

# -----------------------------
# Rcrit
# -----------------------------
Rcrit = 0.693 / (alpha * Tav)

# -----------------------------
# Normalisation
# -----------------------------
max_val = max(Ddot_Lu.max(), Ddot_Cu.max())

Ddot_Lu_n = Ddot_Lu / max_val
Ddot_Cu_n = Ddot_Cu / max_val
Rcrit_n = Rcrit / max_val

# -----------------------------
# Crossing points
# -----------------------------
t_cross_Lu, idx_Lu = find_crossing(t_global, Ddot_Lu, Rcrit)
t_cross_Cu, idx_Cu = find_crossing(t_global, Ddot_Cu, Rcrit)

A_cross_Lu = A_Lu[idx_Lu] if idx_Lu is not None else None
A_cross_Cu = A_Cu[idx_Cu] if idx_Cu is not None else None

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Curves
ax.plot(t_global, Ddot_Lu_n, label="177Lu", linewidth=2)
ax.plot(t_global, Ddot_Cu_n, label="64Cu", linewidth=2)

# Rcrit
ax.axhline(Rcrit_n, linestyle='--', linewidth=2, label="Rcrit")

# -----------------------------
# Shading
# -----------------------------
mask_Lu = Ddot_Lu_n > Rcrit_n
mask_Cu = Ddot_Cu_n > Rcrit_n

# NOT wasted (below Rcrit)
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

# Effective (above Rcrit)
ax.fill_between(
    t_global,
    Rcrit_n,
    Ddot_Lu_n,
    where=mask_Lu,
    interpolate=True,
    alpha=0.25,
    label="Effective (Lu)"
)

ax.fill_between(
    t_global,
    Rcrit_n,
    Ddot_Cu_n,
    where=mask_Cu,
    interpolate=True,
    alpha=0.25,
    label="Effective (Cu)"
)

# -----------------------------
# Crossing markers
# -----------------------------
if idx_Lu is not None:
    ax.scatter(t_cross_Lu, Rcrit_n, s=50)
    ax.annotate("Lu cross", (t_cross_Lu, Rcrit_n),
                xytext=(10, 10), textcoords='offset points')

if idx_Cu is not None:
    ax.scatter(t_cross_Cu, Rcrit_n, s=50)
    ax.annotate("Cu cross", (t_cross_Cu, Rcrit_n),
                xytext=(10, -15), textcoords='offset points')

# -----------------------------
# Formatting
# -----------------------------
ax.set_xlim(0, t_max)
ax.set_ylim(0, 1.0)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Normalised Dose Rate")
ax.set_title("Dose Rate Comparison (Normalised, Same Total Dose)")

ax.grid(True)
ax.legend()

st.pyplot(fig)

# -----------------------------
# Output values
# -----------------------------
st.markdown("### Crossing Points")

col1, col2 = st.columns(2)

with col1:
    st.subheader("177Lu")
    if t_cross_Lu is not None:
        st.write(f"Time: {t_cross_Lu:.2f} h")
        st.write(f"Activity: {A_cross_Lu:.2f} MBq")
    else:
        st.write("No crossing with Rcrit")

with col2:
    st.subheader("64Cu")
    if t_cross_Cu is not None:
        st.write(f"Time: {t_cross_Cu:.2f} h")
        st.write(f"Activity: {A_cross_Cu:.2f} MBq")
    else:
        st.write("No crossing with Rcrit")
