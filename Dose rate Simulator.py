import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------
# Time axis
# -----------------------------
t_max = 200
n_points = 2000
t_global = np.linspace(0, t_max, n_points)  # global fine time axis

# -----------------------------
# Fixed physical half-lives (hours)
# -----------------------------
Tphys_Lu = 160    # 177Lu
Tphys_Cu64 = 12.7 # 64Cu
Tphys_Cu67 = 61.8 # 67Cu

# -----------------------------
# Initial parameters
# -----------------------------
params_Lu = {'Tbio': 200, 'S': 0.05, 'alpha': 0.3, 'Tav': 72}
params_Cu64 = {'Tbio': 50, 'S': 0.05, 'alpha': 0.3, 'Tav': 72}
params_Cu67 = {'Tbio': 50, 'S': 0.05, 'alpha': 0.3, 'Tav': 72}

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
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(left=0.25, bottom=0.55, right=0.95)

D_target_init = 50  # Gy

# Compute initial A0 for each radionuclide
A0_Lu = compute_A0_for_target(D_target_init, Tphys_Lu, params_Lu['Tbio'], params_Lu['S'])
A0_Cu64 = compute_A0_for_target(D_target_init, Tphys_Cu64, params_Cu64['Tbio'], params_Cu64['S'])
A0_Cu67 = compute_A0_for_target(D_target_init, Tphys_Cu67, params_Cu67['Tbio'], params_Cu67['S'])

# Compute initial doses
Ddot_Lu, Rcrit_Lu, total_Lu, eff_Lu, waste_Lu, eff_ratio_Lu, t_Lu = compute_dose(
    A0_Lu, Tphys_Lu, params_Lu['Tbio'], params_Lu['S'], params_Lu['alpha'], params_Lu['Tav'])
Ddot_Cu64, Rcrit_Cu64, total_Cu64, eff_Cu64, waste_Cu64, eff_ratio_Cu64, t_Cu64 = compute_dose(
    A0_Cu64, Tphys_Cu64, params_Cu64['Tbio'], params_Cu64['S'], params_Cu64['alpha'], params_Cu64['Tav'])
Ddot_Cu67, Rcrit_Cu67, total_Cu67, eff_Cu67, waste_Cu67, eff_ratio_Cu67, t_Cu67 = compute_dose(
    A0_Cu67, Tphys_Cu67, params_Cu67['Tbio'], params_Cu67['S'], params_Cu67['alpha'], params_Cu67['Tav'])

# -----------------------------
# Plot all three radionuclides
# -----------------------------
line_Lu, = ax.plot(t_Lu, Ddot_Lu, lw=2, label="177Lu Dose Rate", color='blue')
eff_fill_Lu = ax.fill_between(t_Lu, Rcrit_Lu, Ddot_Lu, where=Ddot_Lu>Rcrit_Lu, color='green', alpha=0.3)
waste_fill_Lu = ax.fill_between(t_Lu, 0, np.minimum(Ddot_Lu, Rcrit_Lu), color='lime', alpha=0.2)
rcrit_line_Lu, = ax.plot(t_Lu, np.full_like(t_Lu, Rcrit_Lu), '--', color='blue')

line_Cu64, = ax.plot(t_Cu64, Ddot_Cu64, lw=2, label="64Cu Dose Rate", color='orange')
eff_fill_Cu64 = ax.fill_between(t_Cu64, Rcrit_Cu64, Ddot_Cu64, where=Ddot_Cu64>Rcrit_Cu64, color='orange', alpha=0.3)
waste_fill_Cu64 = ax.fill_between(t_Cu64, 0, np.minimum(Ddot_Cu64, Rcrit_Cu64), color='gold', alpha=0.2)
rcrit_line_Cu64, = ax.plot(t_Cu64, np.full_like(t_Cu64, Rcrit_Cu64), '--', color='orange')

line_Cu67, = ax.plot(t_Cu67, Ddot_Cu67, lw=2, label="67Cu Dose Rate", color='purple')
eff_fill_Cu67 = ax.fill_between(t_Cu67, Rcrit_Cu67, Ddot_Cu67, where=Ddot_Cu67>Rcrit_Cu67, color='purple', alpha=0.3)
waste_fill_Cu67 = ax.fill_between(t_Cu67, 0, np.minimum(Ddot_Cu67, Rcrit_Cu67), color='violet', alpha=0.2)
rcrit_line_Cu67, = ax.plot(t_Cu67, np.full_like(t_Cu67, Rcrit_Cu67), '--', color='purple')

ax.set_title(f"Dose Rate Curves | Target Dose={D_target_init} Gy")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Dose Rate (Gy/h)")
ax.set_ylim(0, 1.0)  # shared y-scale
ax.legend()

# -----------------------------
# Sliders
# -----------------------------
axcolor = 'lightgoldenrodyellow'

slider_specs_Lu = [('Tbio (h)', 1, 300, params_Lu['Tbio']),
                   ('alpha (Gy^-1)', 0.05, 1.0, params_Lu['alpha']),
                   ('Tav (h)', 10, 200, params_Lu['Tav'])]

slider_specs_Cu64 = [('Tbio (h)', 1, 200, params_Cu64['Tbio']),
                     ('alpha (Gy^-1)', 0.05, 1.0, params_Cu64['alpha']),
                     ('Tav (h)', 10, 200, params_Cu64['Tav'])]

slider_specs_Cu67 = [('Tbio (h)', 1, 200, params_Cu67['Tbio']),
                     ('alpha (Gy^-1)', 0.05, 1.0, params_Cu67['alpha']),
                     ('Tav (h)', 10, 200, params_Cu67['Tav'])]

sliders_Lu = []
sliders_Cu64 = []
sliders_Cu67 = []

for i, (name, vmin, vmax, valinit) in enumerate(slider_specs_Lu):
    ax_slider = plt.axes([0.25, 0.45-0.04*i, 0.2, 0.03], facecolor=axcolor)
    sliders_Lu.append(Slider(ax_slider, name, vmin, vmax, valinit=valinit))

for i, (name, vmin, vmax, valinit) in enumerate(slider_specs_Cu64):
    ax_slider = plt.axes([0.55, 0.45-0.04*i, 0.2, 0.03], facecolor=axcolor)
    sliders_Cu64.append(Slider(ax_slider, name, vmin, vmax, valinit=valinit))

for i, (name, vmin, vmax, valinit) in enumerate(slider_specs_Cu67):
    ax_slider = plt.axes([0.75, 0.45-0.04*i, 0.2, 0.03], facecolor=axcolor)
    sliders_Cu67.append(Slider(ax_slider, name, vmin, vmax, valinit=valinit))

ax_target = plt.axes([0.25, 0.45-0.04*max(len(slider_specs_Lu), len(slider_specs_Cu64), len(slider_specs_Cu67))-0.02, 0.65, 0.03], facecolor=axcolor)
slider_target = Slider(ax_target, 'Target Dose (Gy)', 10, 200, valinit=D_target_init)

# -----------------------------
# Update function
# -----------------------------
def update(val):
    global eff_fill_Lu, waste_fill_Lu, eff_fill_Cu64, waste_fill_Cu64, eff_fill_Cu67, waste_fill_Cu67

    D_target_val = slider_target.val

    # --- 177Lu ---
    Tbio_Lu = sliders_Lu[0].val
    alpha_Lu = sliders_Lu[1].val
    Tav_Lu   = sliders_Lu[2].val
    A0_Lu = compute_A0_for_target(D_target_val, Tphys_Lu, Tbio_Lu, params_Lu['S'])
    Ddot_Lu, Rcrit_Lu, total_Lu, eff_Lu, waste_Lu, eff_ratio_Lu, t_Lu = compute_dose(
        A0_Lu, Tphys_Lu, Tbio_Lu, params_Lu['S'], alpha_Lu, Tav_Lu
    )

    try: eff_fill_Lu.remove()
    except: pass
    try: waste_fill_Lu.remove()
    except: pass
    eff_fill_Lu = ax.fill_between(t_Lu, Rcrit_Lu, Ddot_Lu, where=Ddot_Lu>Rcrit_Lu, color='green', alpha=0.3)
    waste_fill_Lu = ax.fill_between(t_Lu, 0, np.minimum(Ddot_Lu, Rcrit_Lu), color='lime', alpha=0.2)
    line_Lu.set_data(t_Lu, Ddot_Lu)
    rcrit_line_Lu.set_ydata(np.full_like(t_Lu, Rcrit_Lu))

    # --- 64Cu ---
    Tbio_Cu64 = sliders_Cu64[0].val
    alpha_Cu64 = sliders_Cu64[1].val
    Tav_Cu64   = sliders_Cu64[2].val
    A0_Cu64 = compute_A0_for_target(D_target_val, Tphys_Cu64, Tbio_Cu64, params_Cu64['S'])
    Ddot_Cu64, Rcrit_Cu64, total_Cu64, eff_Cu64, waste_Cu64, eff_ratio_Cu64, t_Cu64 = compute_dose(
        A0_Cu64, Tphys_Cu64, Tbio_Cu64, params_Cu64['S'], alpha_Cu64, Tav_Cu64
    )

    try: eff_fill_Cu64.remove()
    except: pass
    try: waste_fill_Cu64.remove()
    except: pass
    eff_fill_Cu64 = ax.fill_between(t_Cu64, Rcrit_Cu64, Ddot_Cu64, where=Ddot_Cu64>Rcrit_Cu64, color='orange', alpha=0.3)
    waste_fill_Cu64 = ax.fill_between(t_Cu64, 0, np.minimum(Ddot_Cu64, Rcrit_Cu64), color='gold', alpha=0.2)
    line_Cu64.set_data(t_Cu64, Ddot_Cu64)
    rcrit_line_Cu64.set_ydata(np.full_like(t_Cu64, Rcrit_Cu64))

    # --- 67Cu ---
    Tbio_Cu67 = sliders_Cu67[0].val
    alpha_Cu67 = sliders_Cu67[1].val
    Tav_Cu67   = sliders_Cu67[2].val
    A0_Cu67 = compute_A0_for_target(D_target_val, Tphys_Cu67, Tbio_Cu67, params_Cu67['S'])
    Ddot_Cu67, Rcrit_Cu67, total_Cu67, eff_Cu67, waste_Cu67, eff_ratio_Cu67, t_Cu67 = compute_dose(
        A0_Cu67, Tphys_Cu67, Tbio_Cu67, params_Cu67['S'], alpha_Cu67, Tav_Cu67
    )

    try: eff_fill_Cu67.remove()
    except: pass
    try: waste_fill_Cu67.remove()
    except: pass
    eff_fill_Cu67 = ax.fill_between(t_Cu67, Rcrit_Cu67, Ddot_Cu67, where=Ddot_Cu67>Rcrit_Cu67, color='purple', alpha=0.3)
    waste_fill_Cu67 = ax.fill_between(t_Cu67, 0, np.minimum(Ddot_Cu67, Rcrit_Cu67), color='violet', alpha=0.2)
    line_Cu67.set_data(t_Cu67, Ddot_Cu67)
    rcrit_line_Cu67.set_ydata(np.full_like(t_Cu67, Rcrit_Cu67))

    ax.set_ylim(0, 1.0)  # maintain shared y-scale
    fig.canvas.draw_idle()

# Connect sliders
for s in sliders_Lu + sliders_Cu64 + sliders_Cu67:
    s.on_changed(update)
slider_target.on_changed(update)

plt.show()
