import numpy as np


def bsm_mse_error(cl, cr, array, hrtf, freqs):
    """
    Calculates the BSM MSE error:
        |V_f @ c_f - conj(h_f)|^2 / |h_f|^2

    Parameters
    ----------
    cl : np.ndarray, shape (F, M)
        Left ear beamformer weights (full spectrum).
    cr : np.ndarray, shape (F, M)
        Right ear beamformer weights (full spectrum).
    array : SpatialSignal
        Steering matrix, shape (M, Q, F), is_space=True, is_freq=True.
    hrtf : SpatialSignal
        HRTF in space domain, shape (2, Q, F), is_space=True, is_freq=True.
    freqs : np.ndarray
        All frequency bins (full spectrum).

    Returns
    -------
    mse_l : np.ndarray, shape (F_pos,)
    mse_r : np.ndarray, shape (F_pos,)
    """
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    V = array.data[:, :, pos_freqs_indices]   # (M, Q, F_pos)
    h = hrtf.data[:, :, pos_freqs_indices]    # (2, Q, F_pos)
    cl_pos = cl[pos_freqs_indices]             # (F_pos, M)
    cr_pos = cr[pos_freqs_indices]             # (F_pos, M)

    F_pos = V.shape[2]
    mse_l = np.zeros(F_pos)
    mse_r = np.zeros(F_pos)

    for f in range(F_pos):
        tmp = np.linalg.norm(V[:, :, f].T @ cl_pos[f, :].conj() - np.conj(h[0, :, f].conj()))
        mse_l[f] = np.square(tmp / np.linalg.norm(h[0, :, f]))
        tmp = np.linalg.norm(V[:, :, f].T @ cr_pos[f, :].conj() - np.conj(h[1, :, f].conj()))
        mse_r[f] = np.square(tmp / np.linalg.norm(h[1, :, f]))

    return mse_l, mse_r


def bsm_mag_mse_error(cl, cr, array, hrtf, freqs):
    """
    Calculates the BSM magnitude MSE error:
        ||V_f @ c_f| - |h_f||^2 / |h_f|^2

    Parameters
    ----------
    cl : np.ndarray, shape (F, M)
        Left ear beamformer weights (full spectrum).
    cr : np.ndarray, shape (F, M)
        Right ear beamformer weights (full spectrum).
    array : SpatialSignal
        Steering matrix, shape (M, Q, F), is_space=True, is_freq=True.
    hrtf : SpatialSignal
        HRTF in space domain, shape (2, Q, F), is_space=True, is_freq=True.
    freqs : np.ndarray
        All frequency bins (full spectrum).

    Returns
    -------
    mse_l : np.ndarray, shape (F_pos,)
    mse_r : np.ndarray, shape (F_pos,)
    """
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    V = array.data[:, :, pos_freqs_indices]   # (M, Q, F_pos)
    h = hrtf.data[:, :, pos_freqs_indices]    # (2, Q, F_pos)
    cl_pos = cl[pos_freqs_indices]             # (F_pos, M)
    cr_pos = cr[pos_freqs_indices]             # (F_pos, M)

    F_pos = V.shape[2]
    mse_l = np.zeros(F_pos)
    mse_r = np.zeros(F_pos)

    for f in range(F_pos):
        tmp = np.linalg.norm(np.abs(V[:, :, f].T @ cl_pos[f, :].conj()) - np.abs(h[0, :, f].conj()))
        mse_l[f] = np.square(tmp / np.linalg.norm(h[0, :, f]))
        tmp = np.linalg.norm(np.abs(V[:, :, f].T @ cr_pos[f, :].conj()) - np.abs(h[1, :, f].conj()))
        mse_r[f] = np.square(tmp / np.linalg.norm(h[1, :, f]))

    return mse_l, mse_r
