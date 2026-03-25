import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(layout="wide")
st.title("Radiopharmaceutical Dose Efficiency Model")

# -----------------------------
# Time axis
# -----------------------------
t_max = 200
n_points = 2000
t_global = np.linspace(0, t_max, n_points)

# -----------------------------
# Fixed physical half-lives (hours)
# -----------------------------
Tphys_Lu = 160   # 177Lu
Tphys_Cu = 12.7  # 64Cu

# -----------------------------
# Functions
# -----------------------------
def trapz_manual(y, x):
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)

def activity_curve(A0, Tphys, Tbio, t_array):
    lambda_p = np.log(2) / Tphys
    lambda_b = np.log(2) / Tbio
    return A0 * (lambda_b / (lambda_b - lambda_p)) * (
        np.exp(-lambda_p * t_array) - np.exp(-lambda_b * t_array)
    )

def compute_dose(A0, Tphys, Tbio, S, alpha, Tav):
    A = activity_curve(A0, Tphys, Tbio, t_global)
    Ddot = S * A
    Rcrit = 0.693 / (alpha * Tav)

    mask = Ddot > Rcrit
    effective = trapz_manual(Ddot[mask], t_global[mask]) if mask.any() else 0
    wasted = trapz_manual(np.minimum(Ddot, Rcrit), t_global)
    total = trapz_manual(Ddot, t_global)
    efficiency = effective / total if total > 0 else 0

    return Ddot, Rcrit, total, effective, wasted, efficiency

def compute_A0_for_target(D_target, Tphys, Tbio, S):
    A_norm = activity_curve(1.0, Tphys, Tbio, t_global)
    integral = trapz_manual(S * A_norm, t_global)
    return D_target / integral

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Global Control")
D_target = st.sidebar.slider("Target Dose (Gy)", 10.0, 200.0, 50.0)

st.sidebar.header("177Lu Parameters")
Tbio_Lu = st.sidebar.slider("Lu Tbio (h)", 1.0, 300.0, 200.0)
alpha_Lu = st.sidebar.slider("Lu alpha (Gy⁻¹)", 0.05, 1.0, 0.3)
Tav_Lu = st.sidebar.slider("Lu Tav (h)", 10.0, 200.0, 72.0)

st.sidebar.header("64Cu Parameters")
Tbio_Cu = st.sidebar.slider("Cu Tbio (h)", 1.0, 200.0, 50.0)
alpha_Cu = st.sidebar.slider("Cu alpha (Gy⁻¹)", 0.05, 1.0, 0.3)
Tav_Cu = st.sidebar.slider("Cu Tav (h)", 10.0, 200.0, 72.0)

# Fixed S (you can expose later if needed)
S = 0.05

# -----------------------------
# Compute A0
# -----------------------------
A0_Lu = compute_A0_for_target(D_target, Tphys_Lu, Tbio_Lu, S)
A0_Cu = compute_A0_for_target(D_target, Tphys_Cu, Tbio_Cu, S)

# -----------------------------
# Compute dose
# -----------------------------
Ddot_Lu, Rcrit_Lu, total_Lu, eff_Lu, waste_Lu, eff_ratio_Lu = compute_dose(
    A0_Lu, Tphys_Lu, Tbio_Lu, S, alpha_Lu, Tav_Lu
)

Ddot_Cu, Rcrit_Cu, total_Cu, eff_Cu, waste_Cu, eff_ratio_Cu = compute_dose(
    A0_Cu, Tphys_Cu, Tbio_Cu, S, alpha_Cu, Tav_Cu
)

# -----------------------------
# Plotting
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---- Lu ----
ax = axes[0]
ax.plot(t_global, Ddot_Lu, label="Dose Rate")
ax.plot(t_global, np.full_like(t_global, Rcrit_Lu), '--', label="Rcrit")

ax.fill_between(t_global, Rcrit_Lu, Ddot_Lu,
                where=Ddot_Lu > Rcrit_Lu, alpha=0.3)
ax.fill_between(t_global, 0, np.minimum(Ddot_Lu, Rcrit_Lu), alpha=0.2)

ax.set_title(f"177Lu | Eff={eff_ratio_Lu:.2f} | Wasted={waste_Lu:.2f} Gy | A0={A0_Lu:.1f} MBq")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Dose Rate (Gy/h)")
ax.legend()

# ---- Cu ----
ax = axes[1]
ax.plot(t_global, Ddot_Cu, label="Dose Rate")
ax.plot(t_global, np.full_like(t_global, Rcrit_Cu), '--', label="Rcrit")

ax.fill_between(t_global, Rcrit_Cu, Ddot_Cu,
                where=Ddot_Cu > Rcrit_Cu, alpha=0.3)
ax.fill_between(t_global, 0, np.minimum(Ddot_Cu, Rcrit_Cu), alpha=0.2)

ax.set_title(f"64Cu | Eff={eff_ratio_Cu:.2f} | Wasted={waste_Cu:.2f} Gy | A0={A0_Cu:.1f} MBq")
ax.set_xlabel("Time (h)")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Key metrics
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("177Lu Summary")
    st.write(f"A0: {A0_Lu:.2f} MBq")
    st.write(f"Efficiency: {eff_ratio_Lu:.3f}")
    st.write(f"Wasted Dose: {waste_Lu:.2f} Gy")

with col2:
    st.subheader("64Cu Summary")
    st.write(f"A0: {A0_Cu:.2f} MBq")
    st.write(f"Efficiency: {eff_ratio_Cu:.3f}")
    st.write(f"Wasted Dose: {waste_Cu:.2f} Gy")
