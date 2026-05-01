# -*- coding: utf-8 -*-
"""
Calculate G' and G'' using:
  - GS code's EXACT column reading and time axis (col0*0.001, col1=strain, col2=stress)
  - GS code's EXACT smoothing (savgol window=501, polyorder=3)
  - GS code's EXACT plot style (gray raw, blue smooth, tomato strain)
  - v5's FFT-based frequency detection + sinusoidal fit for G'/G''

Equations: Jung et al. Carbon 230 (2024) 119599
    Eq.1: tau(t) = tau0 * sin(omega*t + phi_tau)
    Eq.2: G'  = (tau0/gamma0) * cos(delta)
          G'' = (tau0/gamma0) * sin(delta)
    Eq.3: |G*| = sqrt(G'^2 + G''^2)
    Eq.4: eta_s ~ |G*| / f_physical  [f=1 GHz from LAMMPS input]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = "/global/cfs/cdirs/m4770/Shravan/loading_simulations/SV_loading_simulations"

VISCO_FILES = [
    (f"{BASE}/AL_NNIP_loading/100k_shearxy/8x_AL_NNIP_100k_shearxy/visco_train6.9_300k.dat", "AL4_100k",     100, "original"),
    (f"{BASE}/AL_NNIP_loading/100k_shearxy/visco_train6.9_300k.dat",                          "AL4_100k_v2",  100, "original"),
    (f"{BASE}/AL_NNIP_loading/300k_shearxy/visco_AL4_300k.dat",                               "AL4_300k",     300, "original"),
    (f"{BASE}/NVT100k/shearxy/8x_NVT100k_shearxy/visco_train6.9_300k.dat",                   "NVT_100k",     100, "original"),
    (f"{BASE}/NVT100k/shearxy/visco_train6.9_300k.dat",                                       "NVT_100k_v2",  100, "original"),
    (f"{BASE}/NVT300k/shearxy/visco_train6.10_300k.dat",                                      "NVT_300k",     300, "original"),
    (f"{BASE}/NVT350k/shearxy/visco_train6.10_350k.dat",                                      "NVT_350k",     350, "original"),
    (f"{BASE}/NVT400k/shearxy/visco_train6.10_400k.dat",                                      "NVT_400k",     400, "original"),
    (f"{BASE}/2x2x1_shearxy/AL4_300k/visco_AL4_300k.dat",                                    "AL4_300k_221", 300, "221"),
    (f"{BASE}/2x2x1_shearxy/NVT_300k/visco_NVT_300k.dat",                                    "NVT_300k_221", 300, "221"),
    (f"{BASE}/2x2x1_shearxy/NVT_350k/visco_NVT_350k.dat",                                    "NVT_350k_221", 350, "221"),
    (f"{BASE}/2x2x1_shearxy/NVT_400k/visco_NVT_400k.dat",                                    "NVT_400k_221", 400, "221"),
]

OUT_DIR = os.path.join(BASE, "GpGpp_final")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fixed simulation parameters ────────────────────────────────────────────
GAMMA0  = 0.05   # strain amplitude (paper Sec 2.4, LAMMPS input)
F_PHYS  = 1.0    # physical frequency = 1 GHz (for eta only)

# ── GS smoothing function (IDENTICAL to GS code) ──────────────────────────
def smooth(y, window=501):
    if len(y) < window:
        window = max(5, len(y) // 4 * 2 - 1)
    return savgol_filter(y, window_length=window, polyorder=3)

# ── FFT frequency estimator ────────────────────────────────────────────────
def estimate_omega(t, y):
    """Dominant frequency via FFT, returned in rad/[t_unit]."""
    dt   = t[1] - t[0]
    n    = len(y)
    freq = np.fft.rfftfreq(n, d=dt)
    amp  = np.abs(np.fft.rfft(y - y.mean()))
    amp[0] = 0   # ignore DC
    return 2.0 * np.pi * freq[np.argmax(amp)]

# ── Sinusoidal models ──────────────────────────────────────────────────────
def stress_model(t, tau0, omega, phi_tau, C):
    return tau0 * np.sin(omega * t + phi_tau) + C

def strain_model(t, omega, phi_g, C):
    return GAMMA0 * np.sin(omega * t + phi_g) + C

# ── Main loop ──────────────────────────────────────────────────────────────
results       = []
metrics_lines = []

for fpath, label, temp, config in VISCO_FILES:

    if not os.path.exists(fpath):
        print(f"\nMISSING: {label}")
        continue

    print(f"\n{'='*60}")
    print(f"Processing: {label}  T={temp}K  [{config}]")

    # ── Load exactly like GS code ──────────────────────────────────────
    data = np.genfromtxt(fpath, comments='#', encoding='latin-1')

    # GS code column mapping (from working shear_GS_plots.py):
    time_ns  = data[:, 0] * 0.001    # col0 = timestep, *0.001 = GS convention
    gamma_xy = data[:, 1]            # col1 = shear strain
    pxy_GPa  = -data[:, 2] * 0.0001 # col2 = P_xy bar, converted to GPa

    print(f"  time range : {time_ns[0]:.4f} -- {time_ns[-1]:.4f}  (GS units)")
    print(f"  npts       : {len(time_ns)}")

    # ── GS smoothing (IDENTICAL to GS code) ───────────────────────────
    pxy_smooth = smooth(pxy_GPa)

    # ── GS strain scaling (IDENTICAL to GS code) ──────────────────────
    scale         = pxy_smooth.std() / gamma_xy.std()
    strain_scaled = gamma_xy * scale

    # ── Skip first 10% for fitting (equilibration) ────────────────────
    skip = max(len(time_ns) // 10, 1)
    t   = time_ns[skip:]
    tau = pxy_GPa[skip:]
    g   = gamma_xy[skip:]

    # ── Estimate omega from strain FFT ────────────────────────────────
    omega_guess = estimate_omega(t, g)
    print(f"  FFT omega  : {omega_guess:.4f} rad/[t_unit]  "
          f"-> {omega_guess/(2*np.pi):.4f} cycles/[t_unit]")

    # ── Fit stress sinusoid (Eq.1) ────────────────────────────────────
    A_guess = (tau.max() - tau.min()) / 2.0
    try:
        popt_tau, _ = curve_fit(
            stress_model, t, tau,
            p0=[A_guess, omega_guess, 0.0, tau.mean()],
            bounds=([0,            omega_guess*0.5, -np.pi, -np.inf],
                    [np.inf,       omega_guess*2.0,  np.pi,  np.inf]),
            maxfev=200000
        )
        tau0_fit, omega_fit, phi_tau, tau_C = popt_tau
        tau0_fit = abs(tau0_fit)
    except Exception as e:
        print(f"  ERROR stress fit: {e} -- skipping")
        continue

    # ── Fit strain phase at fitted omega ──────────────────────────────
    def _strain(t, phi_g, C):
        return strain_model(t, omega_fit, phi_g, C)

    try:
        popt_g, _ = curve_fit(
            _strain, t, g,
            p0=[0.0, g.mean()],
            bounds=([-np.pi, -np.inf], [np.pi, np.inf]),
            maxfev=200000
        )
        phi_g, _ = popt_g
    except Exception as e:
        print(f"  WARNING strain fit: {e}, using phi_g=0")
        phi_g = 0.0

    # ── Phase lag and G'/G'' (Eq.2 & 3) ──────────────────────────────
    delta     = np.arctan2(np.sin(phi_tau - phi_g),
                           np.cos(phi_tau - phi_g))  # wrapped [-pi, pi]
    Gprime    = (tau0_fit / GAMMA0) * np.cos(delta)
    Gpp       = (tau0_fit / GAMMA0) * np.sin(delta)
    Gstar     = np.sqrt(Gprime**2 + Gpp**2)
    tan_delta = np.tan(delta)
    eta_Pas   = Gstar / F_PHYS   # GPa/GHz = Pa*s

    print(f"  tau0       = {tau0_fit:.6f} GPa")
    print(f"  delta      = {np.degrees(delta):.2f} deg")
    print(f"  G'         = {Gprime:.4f} GPa")
    print(f"  G''        = {Gpp:.4f} GPa")
    print(f"  |G*|       = {Gstar:.4f} GPa")
    print(f"  tan(delta) = {tan_delta:.4f}")
    print(f"  eta        = {eta_Pas:.4f} Pa*s")

    results.append({
        'label': label, 'temp': temp, 'config': config,
        'Gprime': Gprime, 'Gpp': Gpp, 'Gstar': Gstar,
        'tan_delta': tan_delta, 'eta_Pas': eta_Pas,
        'delta_deg': np.degrees(delta),
    })
    metrics_lines.append(
        f"{label:<22} {config:<10} T={temp:>3}K  "
        f"G'={Gprime:+8.4f} GPa  G''={Gpp:+8.4f} GPa  "
        f"|G*|={Gstar:8.4f} GPa  tan(d)={tan_delta:+7.4f}  "
        f"d={np.degrees(delta):+6.1f}°  eta={eta_Pas:.4f} Pa*s"
    )

    # ── Plot: GS style + sinusoidal fit overlay ────────────────────────
    # Fitted sinusoids over the fit window
    t_plot      = np.linspace(t[0], t[-1], 5000)
    tau_fit_plt = stress_model(t_plot, tau0_fit, omega_fit, phi_tau, tau_C)

    fig, ax = plt.subplots(figsize=(8, 4))

    # 1. Raw noisy stress (gray, very thin) -- GS style
    ax.plot(time_ns, pxy_GPa,      color='gray',      lw=0.3, alpha=0.5)
    # 2. Smoothed stress (blue bold) -- GS style
    ax.plot(time_ns, pxy_smooth,   color='blue',       lw=2.0,
            label=r'$P_{xy}$')
    # 3. Shear strain overlay (tomato) -- GS style
    ax.plot(time_ns, strain_scaled,color='tomato',     lw=1.8,
            label=r'Shear Strain $\gamma_{xy}$')
    # 4. Sinusoidal fit (dashed green) -- for G'/G'' verification
    ax.plot(t_plot,  tau_fit_plt,  color='limegreen',  lw=1.5,
            ls='--', label=r'$\tau$ fit (Eq.1)')

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel("t (ns)", fontsize=13)   # GS label
    ax.set_ylabel(r"$P_{xy}$ (GPa)", fontsize=13)
    ax.set_title(
        f"{label} ({temp}K)\n"
        f"G' = {Gprime:.4f} GPa  |  G'' = {Gpp:.4f} GPa  |  "
        f"|G*| = {Gstar:.4f} GPa  |  δ = {np.degrees(delta):.1f}°  |  "
        f"η = {eta_Pas:.4f} Pa·s",
        fontsize=9
    )
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(-0.3, 0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"GpGpp_{label}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: GpGpp_{label}.png")

# ── Summary file ───────────────────────────────────────────────────────────
metrics_path = os.path.join(OUT_DIR, "GpGpp_summary.txt")
with open(metrics_path, "w") as f:
    f.write("=" * 115 + "\n")
    f.write("Storage (G') and Loss (G'') Moduli\n")
    f.write("Reference: Jung et al. Carbon 230 (2024) 119599\n")
    f.write("-" * 115 + "\n")
    f.write("  Eq.1: tau(t) = tau0*sin(omega*t+phi)  "
            "[omega from FFT+curvefit, unit-independent]\n")
    f.write("  Eq.2: G' = (tau0/gamma0)*cos(delta),  "
            "G'' = (tau0/gamma0)*sin(delta)\n")
    f.write("  Eq.3: |G*| = sqrt(G'^2 + G''^2)\n")
    f.write("  Eq.4: eta_s ~ |G*|/f  [f=1 GHz physical, GPa/GHz = Pa*s]\n")
    f.write(f"  Fixed: gamma0={GAMMA0}, f_physical={F_PHYS} GHz\n")
    f.write("  Plotting: identical to GS shear_GS_plots.py style\n")
    f.write("=" * 115 + "\n\n")
    for line in metrics_lines:
        f.write(line + "\n")

print(f"\n{'='*60}")
print(f"Summary : {metrics_path}")
print(f"Plots   : {OUT_DIR}/GpGpp_*.png")
print("ALL DONE!")
