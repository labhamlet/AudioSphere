from shroom.acoustics.spatial_signal import SpatialSignal
from typing import Optional
import numpy as np
from shroom.utils.math_utils import magls
from shroom.utils.dsp_utils import (
    reconstruct_frequency_sh_spectrum_full,
)


def _validate_magls_hrtf_inputs(hrtf, sh_order, cross_over_freq):
    if not hrtf.is_space:
        raise ValueError("magls_hrtf() expects input hrtf to be in space domain.")
    return


def _create_double_ramp_alpha(n_bins, cutoff_bin, final_bin=None, factor=np.sqrt(2)):
    """
    Creates a frequency-dependent alpha parameter for MagLS blending.

    Parameters
    ----------
    n_bins : int
        Number of frequency bins.
    cutoff_bin : int
        The bin index where the transition starts.
    final_bin : int, optional
        The bin index where the transition ends (for band-pass behavior).
    factor : float, optional
        Transition width factor.

    Returns
    -------
    alpha : np.ndarray
        Array of weights in [0, 1].
    """
    idxs = np.arange(n_bins)

    # 1. Calculate points
    p_up_start = cutoff_bin
    p_up_end = cutoff_bin * factor

    if final_bin is not None:
        p_down_end = final_bin

        # --- FIX: Ensure p_down_start is logical ---
        # Option A: Stick to your formula but protect against large factors
        # p_down_start = (factor - 1) * final_bin

        # Option B (Recommended): Use division for symmetry.
        # If Up is (cutoff * factor), Down could start at (final / factor).
        # This guarantees start < end for any factor > 1.
        p_down_start = final_bin / factor

        # If you strictly want the previous formula, ensure it doesn't exceed final_bin:
        # p_down_start = min((factor - 1) * final_bin, final_bin - 1)

        # 2. Collect points
        # We use a list of tuples so we can sort them easily by x-coordinate
        points = [
            (0, 0),
            (p_up_start, 0),
            (p_up_end, 1),
            (p_down_start, 1),
            (p_down_end, 0),
            (n_bins, 0),
        ]
    else:
        points = [(0, 0), (p_up_start, 0), (p_up_end, 1), (n_bins, 1)]

    # 3. Sort by x-coordinate (crucial for np.interp)
    points.sort(key=lambda x: x[0])

    # Unzip into xp and fp
    xp, fp = zip(*points)

    # 4. Interpolate
    alpha = np.interp(idxs, xp, fp)

    return alpha


def magls_hrtf(
    hrtf: SpatialSignal,
    sh_order: Optional[int] = 1,
    cutoff_over_freq: Optional[float] = 1200,
):
    """
    Compute Magnitude Least Squares (MagLS) SH representation of an HRTF.

    This method blends standard Least Squares (LS) at low frequencies with
    Magnitude Least Squares (MagLS) at high frequencies. MagLS optimizes the
    SH coefficients to match the magnitude response, preserving timbre at the
    cost of phase accuracy.

    Parameters
    ----------
    hrtf : SpatialSignal
        Input HRTF in Space Domain.
    sh_order : int, optional
        Target SH order. Default is 1.
    cutoff_over_freq : float, optional
        Crossover frequency (Hz) between LS and MagLS. Default is 1200 Hz.

    Returns
    -------
    hrtf_magls : SpatialSignal
        Processed HRTF in SH Domain (Time).
    """

    hrtf_copy = hrtf.copy()

    _validate_magls_hrtf_inputs(hrtf_copy, sh_order, cutoff_over_freq)

    if hrtf_copy.is_time:
        hrtf_copy.toFreq()

    fs = hrtf_copy.fs
    duration = (1 / fs) * hrtf_copy.data.shape[2]

    freq_axis = np.fft.fftfreq(n=int(duration * fs), d=1 / fs)
    nfft = len(freq_axis)
    pos_freq_axis = np.fft.rfftfreq(n=int(duration * fs), d=1 / fs)
    pos_freqs_indices = freq_axis >= 0.0
    cutoff_bin = np.argmin(np.abs(pos_freq_axis - cutoff_over_freq))
    alpha = _create_double_ramp_alpha(len(pos_freq_axis), cutoff_bin)

    Y = hrtf_copy.grid.Y(sh_order)
    hrtf_space = hrtf_copy.data[..., : len(pos_freq_axis)]
    hrtf_copy.toSH(sh_order)
    hrtf_sh = hrtf_copy.data[..., : len(pos_freq_axis)]

    hnm_mag = hrtf_sh.copy()[:, : len(pos_freq_axis), :]

    for f in range(cutoff_bin, len(pos_freq_axis) - 1):
        hnm_mag[0, :, f] = (
            alpha[f]
            * magls(
                A=Y,
                b=hrtf_space[0, :, f],
                x_prev=hnm_mag[0, :, f - 1],
            )
            + (1 - alpha[f]) * hrtf_sh[0, :, f]
        )
        hnm_mag[1, :, f] = (
            alpha[f]
            * magls(
                A=Y,
                b=hrtf_space[1, :, f],
                x_prev=hnm_mag[1, :, f - 1],
            )
            + (1 - alpha[f]) * hrtf_sh[1, :, f]
        )

    hnm_mag = reconstruct_frequency_sh_spectrum_full(hnm_mag, n_fft=nfft)

    hrtf_copy.data = hnm_mag
    hrtf_copy._log_change_to_history(
        "magls",
        {
            "sh_order": sh_order,
            "cutoff_over_freq": cutoff_over_freq,
            "space hrtf grid size": hrtf_space.shape[1],
        },
    )
    hrtf_copy.toTime()

    return hrtf_copy
