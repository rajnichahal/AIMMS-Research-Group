# -*- coding: utf-8 -*-
"""
Plot G' and G'' vs Temperature from GpGpp_summary results.

Plots generated:
  GROUP 1 — By config (original vs 221):
    Plot 1: G' vs T — original only
    Plot 2: G'' vs T — original only
    Plot 3: G' vs T — 221 only
    Plot 4: G'' vs T — 221 only

  GROUP 2 — By model (A vs B):
    Plot 5: G' vs T — A vs B — combined (original + 221 or 441 or both together)
    Plot 6: G'' vs T — A vs B — combined (original + 221 or 441 or both together)
    Plot 7: G' vs T — A vs B — original and 221 separate  or 441
    Plot 8: G'' vs T — A vs B — original and 221 separate  or 441
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Output directory ───────
BASE    = "Main_Location"
OUT_DIR = os.path.join(BASE, "GpGpp_final", "GvsT_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data from summary file ──────
# Format: (label, model, config, T, Gprime, Gpp)
DATA = [
    # label              model   config      T     G'       G''
    ("AL4_100k_221",    "AL4",  "221",      100,  1.2727,  0.3536),
    ("AL4_100k",        "AL4",  "original", 100,  1.4503,  0.3874),
    ("AL4_300k",        "AL4",  "original", 300,  0.1885,  0.1627),

    # Can be added more using the same format .....Can be used for 441 config as well
]

# ── Helper: filter data ───────
def get(model=None, config=None):
    """Return sorted (T, Gprime, Gpp, label) for given filters."""
    out = []
    for label, m, c, T, gp, gpp in DATA:
        if model  and m != model:  continue
        if config and c != config: continue
        out.append((T, gp, gpp, label))
    out.sort(key=lambda x: x[0])
    return out

# ── Plot style ──────
STYLE = {
    ("AL4",  "original"): dict(color="royalblue",  marker="o", ls="-",  lw=2, ms=8),
    ("AL4",  "221"):      dict(color="royalblue",  marker="s", ls="--", lw=2, ms=8),
    ("NVT",  "original"): dict(color="tomato",     marker="o", ls="-",  lw=2, ms=8),
    ("NVT",  "221"):      dict(color="tomato",     marker="s", ls="--", lw=2, ms=8),
  # Can be added more using the same format

}

def label_name(model, config):
    return f"{model}-NNIP ({config})"

# ── Annotation helper ─────
def annotate(ax, rows, ykey):
    """Add data point labels."""
    for T, gp, gpp, lbl in rows:
        y = gp if ykey == "gp" else gpp
        ax.annotate(lbl, (T, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color='gray')

def finish(ax, title, xlabel="Temperature (K)", ylabel="Modulus (GPa)"):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    plt.tight_layout()


# GROUP 1 — By config


for config in ["original", "221"]: # More config can be added using the same format and if there are more config please add here
    for ykey, ylabel, modname in [("gp", "G' (GPa)", "Gprime"),
                                   ("gpp", "G'' (GPa)", "Gpp")]:

        fig, ax = plt.subplots(figsize=(7, 5))

        for model in ["AL4", "NVT"]:     # More Model can be added using the same format and if there are more model please add here
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
                            fontsize=7.5, color=sty["color"])

        title_mod = "G'" if ykey == "gp" else "G''"
        finish(ax,
               title=f"{title_mod} vs Temperature — {config} config",
               ylabel=ylabel)

        fname = f"G{modname}_vs_T_{config}.png"
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")


# GROUP 2a — By model
# Here we plot all data points for each model, using different markers for
# original vs 221, but same color per model — shows full picture

for ykey, ylabel, modname in [("gp", "G' (GPa)", "Gprime"),
                               ("gpp", "G'' (GPa)", "Gpp")]:

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in ["AL4", "NVT"]: # if there are more model please add here
        for config in ["original", "221"]:       # if there are more config please add here

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
                            fontsize=7, color=sty["color"])

    title_mod = "G'" if ykey == "gp" else "G''"
    finish(ax,
           title=f"{title_mod} vs Temperature — AL4 vs NVT (all configs)",
           ylabel=ylabel)

    fname = f"G{modname}_vs_T_AL4vsNVT_combined.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

# GROUP 2b — By model

for ykey, ylabel, modname in [("gp", "G' (GPa)", "Gprime"),
                               ("gpp", "G'' (GPa)", "Gpp")]:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    title_mod = "G'" if ykey == "gp" else "G''"

    for ax, config in zip(axes, ["original", "221"]):       # if there are more config please add here
        for model in ["AL4", "NVT"]:                        # if there are more model please add here

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
                            fontsize=7.5, color=sty["color"])

        ax.set_xlabel("Temperature (K)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f"{title_mod} vs T — {config} config", fontsize=12,
                     fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', lw=0.8, ls='--')

    fig.suptitle(f"{title_mod} vs Temperature: AL4 vs NVT",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fname = f"G{modname}_vs_T_AL4vsNVT_separate.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

print("\nALL DONE! Plots saved to:", OUT_DIR)
print("\nFiles generated:")
print("  GROUP 1 generated")
print("  GROUP 2a generated")
print("  GROUP 2b generated")
