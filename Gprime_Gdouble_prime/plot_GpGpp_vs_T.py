# -*- coding: utf-8 -*-
"""
Plot G' and G'' vs Temperature from GpGpp_summary results.

Plots generated:
  GROUP 1 — By config (original vs 221):
    Plot 1: G' vs T — original only
    Plot 2: G'' vs T — original only
    Plot 3: G' vs T — 221 only
    Plot 4: G'' vs T — 221 only

  GROUP 2 — By model (AL4 vs NVT):
    Plot 5: G' vs T — AL4 vs NVT — combined (original + 221 together)
    Plot 6: G'' vs T — AL4 vs NVT — combined (original + 221 together)
    Plot 7: G' vs T — AL4 vs NVT — original and 221 separate (4 lines)
    Plot 8: G'' vs T — AL4 vs NVT — original and 221 separate (4 lines)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Output directory ───────────────────────────────────────────────────────
BASE    = "/global/cfs/cdirs/m4770/Shravan/loading_simulations/SV_loading_simulations"
OUT_DIR = os.path.join(BASE, "GpGpp_final", "GvsT_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data from summary file ─────────────────────────────────────────────────
# Format: (label, model, config, T, Gprime, Gpp)
DATA = [
    # label              model   config      T     G'       G''
    ("AL4_100k",    "AL4",  "221",      100,  1.2727,  0.3536),
    ("AL4_300k",    "AL4",  "221",      300,  0.3728,  0.2727),
    ("Ref_100k",    "NVT",  "221",      100,  1.5941,  0.1811),
    ("Ref_300k",    "NVT",  "221",      300,  0.3933,  0.1362),
    ("Ref_350k",    "NVT",  "221",      350,  0.1923,  0.0932),
    ("Ref_400k",    "NVT",  "221",      400,  0.1165,  0.0718),

]

# ── Helper: filter data ────────────────────────────────────────────────────
def get(model=None, config=None):
    """Return sorted (T, Gprime, Gpp, label) for given filters."""
    out = []
    for label, m, c, T, gp, gpp in DATA:
        if model  and m != model:  continue
        if config and c != config: continue
        out.append((T, gp, gpp, label))
    out.sort(key=lambda x: x[0])
    return out

# ── Plot style ─────────────────────────────────────────────────────────────
STYLE = {
    ("AL4",  "original"): dict(color="royalblue",  marker="o", ls="-",  lw=2, ms=8),
    ("AL4",  "221"):      dict(color="royalblue",  marker="s", ls="--", lw=2, ms=8),
    ("AL4",  "441"):      dict(color="royalblue",  marker="^", ls=":",  lw=2, ms=8),
    ("NVT",  "original"): dict(color="tomato",     marker="o", ls="-",  lw=2, ms=8),
    ("NVT",  "221"):      dict(color="tomato",     marker="s", ls="--", lw=2, ms=8),
    ("NVT",  "441"):      dict(color="tomato",     marker="^", ls=":",  lw=2, ms=8),
    ("AL2",  "221"):      dict(color="seagreen",     marker="D", ls="-.",  lw=2, ms=8),
    ("AL3",  "221"):      dict(color="darkviolet",     marker="v", ls=(0, (3, 1, 1, 1)),  lw=2, ms=8),
}

def label_name(model, config):
    if model == "NVT":
        return f"Reference-NNIP ({config})"
    else:
        return f"{model}-NNIP ({config})"

# ── Annotation helper ──────────────────────────────────────────────────────
def annotate(ax, rows, ykey):
    """Add data point labels."""
    for T, gp, gpp, lbl in rows:
        y = gp if ykey == "gp" else gpp
        ax.annotate(lbl, (T, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=10, color='gray')

def finish(ax, title, xlabel="Temperature (K)", ylabel="Modulus (GPa)"):
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    #ax.axhline(0, color='k', lw=0.8, ls='--')
    plt.tight_layout()

# ══════════════════════════════════════════════════════════════════════════
# GROUP 1 — By config: original only and 221 only
# ══════════════════════════════════════════════════════════════════════════

for config in ["original", "221", "441"]:
    for ykey, ylabel, modname in [("gp", "G' (GPa)", "Gprime"),
                                   ("gpp", "G'' (GPa)", "Gpp")]:

        fig, ax = plt.subplots(figsize=(7, 5))

        for model in ["AL4", "NVT" , "AL2" , "AL3"]:
            rows = get(model=model, config=config)
            if not rows:
                continue
            T_arr  = [r[0] for r in rows]
            y_arr  = [r[1] if ykey=="gp" else r[2] for r in rows]
            labels = [r[3] for r in rows]
            sty    = STYLE[(model, config)]

            ax.plot(T_arr, y_arr,
                    label=label_name(model, config),
                    **sty)

            # label each point with its name
            for T, y, lbl in zip(T_arr, y_arr, labels):
                ax.annotate(lbl, (T, y),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=10, color=sty["color"])

        title_mod = "G'" if ykey == "gp" else "G''"
        finish(ax,
               title=f"{title_mod} vs Temperature — {config} config",
               ylabel=ylabel)

        fname = f"G{modname}_vs_T_{config}.png"
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════
# GROUP 2a — By model: AL4 vs NVT, combined (original + 221 averaged/together)
# ══════════════════════════════════════════════════════════════════════════
# Here we plot all data points for each model, using different markers for
# original vs 221, but same color per model — shows full picture

for ykey, ylabel, modname in [("gp", "G' (GPa)", "Gprime"),
                               ("gpp", "G'' (GPa)", "Gpp")]:

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in ["AL4", "NVT", "AL2" , "AL3"]:
        for config in ["original", "221", "441"]:
            rows = get(model=model, config=config)
            if not rows:
                continue
            T_arr  = [r[0] for r in rows]
            y_arr  = [r[1] if ykey=="gp" else r[2] for r in rows]
            labels = [r[3] for r in rows]
            sty    = STYLE[(model, config)]

            ax.plot(T_arr, y_arr,
                    label=label_name(model, config),
                    **sty)

            for T, y, lbl in zip(T_arr, y_arr, labels):
                ax.annotate(lbl, (T, y),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=10, color=sty["color"])

    title_mod = "G'" if ykey == "gp" else "G''"
    finish(ax,
           title=f"{title_mod} vs Temperature — AL4 vs NVT (all configs)",
           ylabel=ylabel)

    fname = f"G{modname}_vs_T_AL4vsNVT_combined.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════
# GROUP 2b — By model: AL4 vs NVT, separate subplots for original and 221
# ══════════════════════════════════════════════════════════════════════════

for ykey, ylabel, modname in [("gp", "G' (GPa)", "Gprime"),
                               ("gpp", "G'' (GPa)", "Gpp")]:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    title_mod = "G'" if ykey == "gp" else "G''"

    for ax, config in zip(axes, ["original", "221", "441"]):
        for model in ["AL4", "NVT", "AL2" , "AL3"]:
            rows = get(model=model, config=config)
            if not rows:
                continue
            T_arr  = [r[0] for r in rows]
            y_arr  = [r[1] if ykey=="gp" else r[2] for r in rows]
            labels = [r[3] for r in rows]
            sty    = STYLE[(model, config)]

            ax.plot(T_arr, y_arr,
                    label=label_name(model, config),
                    **sty)

            for T, y, lbl in zip(T_arr, y_arr, labels):
                ax.annotate(lbl, (T, y),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=10, color=sty["color"])

        ax.set_xlabel("Temperature (K)", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(f"{title_mod} vs T — {config} config", fontsize=16,
                     fontweight='bold')
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)
        #ax.axhline(0, color='k', lw=0.8, ls='--')

    fig.suptitle(f"{title_mod} vs Temperature: AL4 vs NVT",
                 fontsize=17, fontweight='bold')
    plt.tight_layout()

    fname = f"G{modname}_vs_T_AL4vsNVT_separate.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

print("\nALL DONE! Plots saved to:", OUT_DIR)
print("\nFiles generated:")
print("  GROUP 1 (by config):")
print("    GGprime_vs_T_original.png  — G' vs T, original only")
print("    GGpp_vs_T_original.png     — G'' vs T, original only")
print("    GGprime_vs_T_221.png       — G' vs T, 221 only")
print("    GGpp_vs_T_221.png          — G'' vs T, 221 only")
print("  GROUP 2a (AL4 vs NVT, combined):")
print("    GGprime_vs_T_AL4vsNVT_combined.png")
print("    GGpp_vs_T_AL4vsNVT_combined.png")
print("  GROUP 2b (AL4 vs NVT, separate subplots):")
print("    GGprime_vs_T_AL4vsNVT_separate.png")
print("    GGpp_vs_T_AL4vsNVT_separate.png")
