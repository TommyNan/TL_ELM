"""Plot the FSPL prior ablation. Run with: python plot_fspl_ablation.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Each entry contains (means, standard deviations) in Scratch, Transfer order.
DATA = {
    "RMSE": {
        "Direct": ([4.3747131217, 4.1330272544], [0.0470301356, 0.0664207524]),
        "Residual": ([4.2000227448, 3.9837123693], [0.0356453763, 0.0383187561]),
    },
    "MAE": {
        "Direct": ([2.8662947547, 2.5571358748], [0.0464488314, 0.0393110952]),
        "Residual": ([2.6560164390, 2.4182583345], [0.0361247965, 0.0238162014]),
    },
}


def main():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.65, "hatch.linewidth": 0.45,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(3.34, 2.05), sharey=True)
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.20, top=0.84,
                        wspace=0.16)
    styles = [
        ("Direct", -0.20, "#75A2D8", "#34547C", "///"),
        ("Residual", 0.20, "#F4AA62", "#80512C", "\\\\\\"),
    ]
    for ax, panel, (metric, modes) in zip(axes, ["(a)", "(b)"], DATA.items()):
        for mode, offset, fill, edge, hatch in styles:
            means, stds = modes[mode]
            centers = [offset, 1 + offset]
            ax.bar(centers, means, width=0.36, yerr=stds, label=mode,
                   color=fill, edgecolor=edge, linewidth=0.65, hatch=hatch,
                   error_kw={"ecolor": "#222222", "elinewidth": 0.7,
                             "capsize": 2, "capthick": 0.7})
            for x, mean, std in zip(centers, means, stds):
                ax.annotate(f"{mean:.2f}", (x, mean + std), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom")
        ax.set_xlabel(f"{panel} {metric}", fontsize=9, labelpad=4)
        ax.set_xticks([0, 1], ["Scratch", "Transfer"])
        ax.set(xlim=(-0.54, 1.54), ylim=(0, 5.05), yticks=range(6))
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#BDBDBD")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=0.6, pad=2)
    axes[0].set_ylabel("Error (dB)", labelpad=3)
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["Direct (no prior)", "Residual (FSPL prior)"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1),
               handlelength=1.4, handletextpad=0.4, columnspacing=1.1)
    output = Path(__file__).resolve().parent / "fspl_prior_ablation"
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()



