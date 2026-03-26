import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Time axis
# -----------------------------
t_max = 200
n_points = 2000
t_global = np.linspace(0, t_max, n_points)

# -----------------------------
# Physical half-lives (hours)
# -----------------------------
Tphys_dict = {
    '177Lu': 160,
    '64Cu': 12.7,
    '67Cu': 61.8  # approximate physical half-life of 67Cu
}

# -----------------------------
# Default parameters
# -----------------------------
params_default = {
    '177Lu': {'Tbio': 200, 'S': 0.05, 'alpha': 0.3, 'Tav': 72},
    '64Cu': {'Tbio': 50,  'S': 0.05, 'alpha': 0.3, 'Tav': 72},
    '67Cu': {'Tbio': 50,  'S': 0.05, 'alpha': 0.3, 'Tav': 72}
}

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
    A = A0 * (lambda_b / (lambda_b - lambda_p)) * (np.exp(-lambda_p*t_array) - np.exp(-lambda_b*t_array))
    return A

# -----------------------------
# Compute dose
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

    return Ddot, Rcrit, total, effective, wasted, efficiency, t_array

# -----------------------------
# Compute required A0 for target dose
# -----------------------------
def compute_A0_for_target(D_target, Tphys, Tbio, S, t_array=None):
    if t_array is None:
        t_array = t_global
    A_norm = activity_curve(1.0, Tphys, Tbio, t_array)
    integral = trapz_manual(S*A_norm, t_array)
    A0 = D_target / integral
    return A0

# -----------------------------
# Streamlit sliders
# -----------------------------
st.title("Radionuclide Dose Simulator (177Lu, 64Cu, 67Cu)")

# Target Dose
D_target = st.slider("Target Dose (Gy)", min_value=10.0, max_value=200.0, value=50.0, step=5.0)

# Parameters for each radionuclide
params = {}
for rn in ['177Lu','64Cu','67Cu']:
    st.subheader(f"{rn} Parameters")
    Tbio = st.slider(f"{rn} Biological Half-life Tbio (h)", 1.0, 300.0, float(params_default[rn]['Tbio']))
    alpha = st.slider(f"{rn} Alpha (Gy^-1)", 0.05, 1.0, float(params_default[rn]['alpha']))
    Tav   = st.slider(f"{rn} Tav (h)", 10.0, 200.0, float(params_default[rn]['Tav']))
    params[rn] = {'Tbio': Tbio, 'S': params_default[rn]['S'], 'alpha': alpha, 'Tav': Tav}

# -----------------------------
# Plotting
# -----------------------------
fig, ax = plt.subplots(figsize=(10,6))

colors = {'177Lu':'blue', '64Cu':'orange', '67Cu':'green'}

for rn in ['177Lu','64Cu','67Cu']:
    Tphys = Tphys_dict[rn]
    Tbio = params[rn]['Tbio']
    alpha = params[rn]['alpha']
    Tav   = params[rn]['Tav']
    S     = params[rn]['S']

    # Compute A0
    A0 = compute_A0_for_target(D_target, Tphys, Tbio, S)

    # Compute dose
    Ddot, Rcrit, total, eff, wasted, efficiency, t_array = compute_dose(A0, Tphys, Tbio, S, alpha, Tav)

    # Plot dose rate
    ax.plot(t_array, Ddot/Ddot.max(), color=colors[rn], lw=2, label=f"{rn} Dose Rate (scaled)")

    # Rcrit dashed
    ax.plot(t_array, np.full_like(t_array, Rcrit/Ddot.max()), linestyle='--', color=colors[rn], alpha=0.7)

    # Shade wasted dose
    ax.fill_between(t_array, 0, np.minimum(Ddot, Rcrit)/Ddot.max(), color=colors[rn], alpha=0.2, label=f"{rn} Wasted Dose")

    # Shade effective dose
    ax.fill_between(t_array, Rcrit/Ddot.max(), Ddot/Ddot.max(), where=Ddot>Rcrit, color=colors[rn], alpha=0.3, label=f"{rn} Effective Dose")

    # Print info
    st.write(f"**{rn}:** A0={A0:.2f} MBq | Total Dose={total:.2f} Gy | Effective={eff:.2f} Gy | Wasted={wasted:.2f} Gy | Efficiency={efficiency*100:.1f}%")

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Dose Rate (normalized)")
ax.set_ylim(0, 1.0)
ax.set_title(f"Dose Rate Profiles for 3 Radionuclides (Target Dose={D_target} Gy)")
ax.legend(loc='upper right', fontsize=8)

st.pyplot(fig)
