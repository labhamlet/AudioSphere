import os
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt


def loglog_plot(
    freqs,
    errors: dict,
    figsize=(10, 5),
    title=None,
    save_path=None,
    show=True,
    styles: dict = None,
    ylabel="Error (dB)",
    xlim=None,
    ylim=None,
    beta=0.1,
):
    """
    Plot error curves. Automatically changes line style if a curve overlaps
    significantly with a previously plotted one.

    Parameters
    ----------
    beta : float, optional (default=0.1)
        Overlap threshold in dB. If the maximum difference between a new curve
        and any previous curve is less than 'beta', the new curve is considered
        "on top" and its style is changed.
    """
    plt.figure(figsize=figsize)

    # Store the dB values of curves we have already plotted to check for overlaps
    history_db = []

    # Cycle of styles to use ONLY when overlap is detected
    # (Dashed, Dotted, Dash-Dot)
    overlap_style_cycler = cycle(["--", ":", "-."])

    for label, err in errors.items():
        # 1. Convert to dB for plotting and comparison
        curr_db = 10 * np.log10(err)

        # 2. Determine Style
        # Priority 1: User manual styles
        if styles is not None and label in styles:
            line_style = styles[label]

        else:
            # Priority 2: Check for overlap with ANY previous curve
            is_overlapping = False
            for prev_db in history_db:
                # Check if the curves are "on top of each other" (max diff < beta)
                # You can change np.max to np.mean if you want a looser 'average' check
                if np.max(np.abs(curr_db - prev_db)) < beta:
                    is_overlapping = True
                    break

            if is_overlapping:
                line_style = next(overlap_style_cycler)
            else:
                line_style = "-"  # Default solid for unique curves

        # 3. Plot
        plt.plot(freqs, curr_db, label=label, linestyle=line_style)

        # 4. Save to history
        history_db.append(curr_db)

    if ylim is not None:
        plt.ylim(ylim)

    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xscale("log")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
