"""Generate the target-band sampling figure. Run: python plot_target_sampling.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Percent of eligible Rx locations used for target-band fine-tuning.
fractions = [0, 1, 2, 5, 10]
rmse = [13.691088748718505, 6.048625627509827, 5.4905754638681845,
        4.604196541957313, 3.9837123693290386]
rmse_std = [0.35810687373901345, 0.12654022701772663, 0.06471670265128367,
            0.048485158430616945, 0.0383187561308501]
mae = [11.004461020295272, 3.95406696692546, 3.479887849386433,
       2.8234114119337157, 2.4182583344999373]
mae_std = [0.31160177030147845, 0.06990195702528576, 0.035233430855456806,
           0.023248529678599528, 0.0238162014371278]


def main():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 8,
        "axes.labelsize": 8.5, "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.34, 2.12))
    fig.subplots_adjust(left=0.125, right=0.975, bottom=0.205, top=0.97)

    series = [
        ("RMSE", rmse, rmse_std, "#1F77B4", "^", "--"),
        ("MAE", mae, mae_std, "#F23526", "o", "-"),
    ]
    for label, means, stds, color, marker, style in series:
        ax.errorbar(
            fractions, means, yerr=stds, label=label, color=color,
            marker=marker, markerfacecolor=color, markeredgewidth=0.7,
            markersize=3.6, linestyle=style, linewidth=1.0,
            elinewidth=0.7, capsize=2.2,
        )

    ax.set(xlabel="Target-band Rx locations used for fine-tuning (%)",
           ylabel="Error (dB)", xlim=(-0.32, 10.4), ylim=(0, 15.6),
           xticks=fractions, yticks=range(0, 16, 3))
    ax.spines[:].set_visible(True)
    ax.tick_params(direction="in", length=2.5, width=0.65, pad=2)
    ax.set_axisbelow(True)
    ax.grid(axis="both", color="#D0D0D0", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="#CCCCCC", handlelength=2.2,
              handletextpad=0.6, borderpad=0.4)
    output = Path(__file__).resolve().parent / "target_band_sampling"
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()