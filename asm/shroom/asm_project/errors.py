import numpy as np
from shroom.utils.amb_utils import get_tilde_matrix


def asm_mse_error(cnm, sm, Y, freqs):
    """
    calculates:
            |cnm^HV - ynm^*|^2 / |ynm|^2
        Args:
            cnm: (mics, nm, F)
            sm: (mics, nm, F)
            Y: (Q, nm)
        Returns:
            error: (nm, F)
    """
    # get only positive frequencies
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    # reshape
    cnm = cnm[:, :, pos_freqs_indices].transpose(2, 1, 0)  # (F, nm, mics)
    sm = sm[:, :, pos_freqs_indices].transpose(2, 0, 1)
    Y = Y.T
    # error calculations
    nominator = cnm.conj() @ sm
    nominator = nominator - Y[np.newaxis, ...].conj()
    nominator = np.square(np.linalg.norm(nominator, ord=2, axis=2))
    denominator = np.square(np.linalg.norm(Y, ord=2, axis=1))
    error = nominator.T / denominator[..., np.newaxis]
    return error


def asm_bin_mse_error(hnm, cnm, sm, h, freqs):
    """
    calculates:
        |tilde(hnm^T) cnm^H V - h^T|^2 / |h|^2
    Args:
        hnm (ears, nm, F)
        cnm: (mics, nm, F)
        sm: (mics, Q, F)
        h: (ears, Q, F)
        freqs: all frequencies
    Returns:
        error (ears, F)
    """
    # get only positive frequencies
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    cnm = cnm[:, :, pos_freqs_indices].transpose(2, 1, 0)
    sm = sm[:, :, pos_freqs_indices].transpose(2, 0, 1)
    h = h[:, :, pos_freqs_indices].transpose(2, 0, 1)
    hnm = hnm[:, :, pos_freqs_indices].transpose(2, 0, 1)
    tilde = get_tilde_matrix(sh_order=int(np.sqrt(cnm.shape[1]) - 1))

    proj = cnm.conj() @ sm
    res_tilde = tilde @ proj
    final = hnm @ res_tilde

    nominator = final - h
    nominator = np.square(np.linalg.norm(nominator, axis=2))
    denominator = np.square(np.linalg.norm(h, axis=2))
    error = nominator / denominator
    return error.T