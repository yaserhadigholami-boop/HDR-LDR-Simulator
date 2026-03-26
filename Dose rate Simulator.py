# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t_global, Ddot_Lu_n, label="177Lu", linewidth=2)
ax.plot(t_global, Ddot_Cu_n, label="64Cu", linewidth=2)

ax.axhline(Rcrit_n, linestyle='--', linewidth=2, label="Rcrit")

# -----------------------------
# Shading
# -----------------------------
mask_Lu = Ddot_Lu_n > Rcrit_n
mask_Cu = Ddot_Cu_n > Rcrit_n

# Not effective (below Rcrit)
ax.fill_between(t_global, 0, np.minimum(Ddot_Lu_n, Rcrit_n), alpha=0.08)
ax.fill_between(t_global, 0, np.minimum(Ddot_Cu_n, Rcrit_n), alpha=0.08)

# Effective
ax.fill_between(t_global, Rcrit_n, Ddot_Lu_n, where=mask_Lu, alpha=0.25)
ax.fill_between(t_global, Rcrit_n, Ddot_Cu_n, where=mask_Cu, alpha=0.25)

# -----------------------------
# Crossing markers
# -----------------------------
if idx_Lu is not None:
    ax.scatter(t_Lu, Rcrit_n)
    ax.annotate("Lu", (t_Lu, Rcrit_n), xytext=(10, 10), textcoords='offset points')

if idx_Cu is not None:
    ax.scatter(t_Cu, Rcrit_n)
    ax.annotate("Cu", (t_Cu, Rcrit_n), xytext=(10, -15), textcoords='offset points')

# -----------------------------
# TEXT BOXES (UPDATED)
# -----------------------------
text_Lu = (
    f"177Lu\n"
    f"A0: {A0_Lu:.2f} MBq\n"
    f"Cross: {t_Lu:.2f} h\n"
    f"Wasted: {waste_Lu:.2f} Gy"
)

text_Cu = (
    f"64Cu\n"
    f"A0: {A0_Cu:.2f} MBq\n"
    f"Cross: {t_Cu:.2f} h\n"
    f"Wasted: {waste_Cu:.2f} Gy"
)

peak_ratio = Ddot_Cu.max() / Ddot_Lu.max()

text_global = (
    f"Peak Ratio (Cu/Lu): {peak_ratio:.2f}×"
)

# Place text boxes
ax.text(0.02, 0.95, text_Lu, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round", alpha=0.2))

ax.text(0.02, 0.65, text_Cu, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round", alpha=0.2))

ax.text(0.65, 0.95, text_global, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle="round", alpha=0.3))

# -----------------------------
# Axis formatting
# -----------------------------
ax.set_xlim(0, t_max)
ax.set_ylim(0, 1.0)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Normalised Dose Rate")
ax.set_title("Dose Rate Comparison (Same Total Dose)")

ax.legend()
ax.grid(True)

st.pyplot(fig)
