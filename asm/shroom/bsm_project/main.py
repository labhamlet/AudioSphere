import numpy as np

from shroom.geometry.sampling import sphereicalGrid
from shroom.paths import DEFAULT_HRTF_PATH
from shroom.utils.file_utils import load_file
from shroom.acoustics.spherical_array import SphericalArray
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.encoders.bsm import BSM
from bsm_project.errors import bsm_mse_error, bsm_mag_mse_error
from devutils.plot import loglog_plot


def main():
    # 1. Setup
    fs = 16000
    duration = 0.016
    freqs = np.fft.fftfreq(int(duration * fs), 1 / fs)
    pos_freqs = np.fft.rfftfreq(int(duration * fs), 1 / fs)

    hrtf = load_file(DEFAULT_HRTF_PATH)
    hrtf.resample(fs)
    hrtf.zero_pad(int(duration * fs))

    source_grid = from_fibonacci_grid(240)

    hrtf.toFreq()
    hrtf.toSH(30)
    hrtf.toSpace(source_grid)
    space_hrtf = hrtf.copy()   # (2, 240, F), is_space=True, is_freq=True

    az = np.deg2rad(np.array([-90, -45, 0, 45, 90]))
    co = np.deg2rad(np.array([90, 90 + 18, 90 - 18, 90 + 18, 90]))
    mic_grid = sphereicalGrid(
        az=az,
        co=co,
    )

    array = SphericalArray(
        fs=fs,
        duration=duration,
        r_sphere=0.08,
        r_mics=0.08 * np.ones((mic_grid.n_points,)),
        source_grid=source_grid,
        mics_grid=mic_grid,
        sphere_type="rigid",
        sh_order_for_sm_calc=14,
        convert_to_time=False,
    )
    # array: (M, Q, F), is_space=True, is_freq=True

    # 2. BSM without MagLS
    bsm_reg = BSM(
        array=array,
        hrtf=space_hrtf,
        use_magls=False,
        fs=fs,
        duration=duration,
    )
    cl_reg, cr_reg = bsm_reg.get_coefficients()

    # 3. BSM with MagLS
    bsm_mag = BSM(
        array=array,
        hrtf=space_hrtf,
        use_magls=True,
        magls_cutoff_frequency=800.0,
        fs=fs,
        duration=duration,
    )
    cl_mag, cr_mag = bsm_mag.get_coefficients()

    # 4. Compute errors
    mse_l_reg, mse_r_reg = bsm_mse_error(cl_reg, cr_reg, array, space_hrtf, freqs)
    mse_l_mag, mse_r_mag = bsm_mag_mse_error(cl_mag, cr_mag, array, space_hrtf, freqs)

    # 5. Plot
    loglog_plot(
        freqs=pos_freqs,
        title='BSM MSE Error',
        errors={
            'left (regular)': mse_l_reg,
            'right (regular)': mse_r_reg,
        },
        figsize=(6, 4),
        show=True,
    )
    loglog_plot(
        freqs=pos_freqs,
        title='BSM MagMSE Error (with MagLS)',
        errors={
            'left (MagLS)': mse_l_mag,
            'right (MagLS)': mse_r_mag,
        },
        figsize=(6, 4),
        show=True,
    )


if __name__ == "__main__":
    main()
