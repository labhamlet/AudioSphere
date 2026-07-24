
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import Resample
from pathlib import Path
from multiprocessing import Pool
from shroom.acoustics.spherical_array import SphericalArray
from shroom.encoders.asm import ASM
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.acoustics.spatial_signal import SpatialSignal

FS          = 24000.0 
OUTPUT_FS   = 32000 
DURATION    = 0.008

# Tetrahedral MIC array — TAU2020 / TAU2021 / STARSS2023
#   M1: ( 45°,  35°, 4.2 cm)
#   M2: (-45°, -35°, 4.2 cm)
#   M3: (135°, -35°, 4.2 cm)
#   M4: (-135°, 35°, 4.2 cm)
tau_az = np.array([45, -45, 135, -135]) * np.pi / 180
tau_el = np.array([35, -35, -35,  35]) * np.pi / 180
tau_co = (np.pi / 2) - tau_el
tau_r  = 0.042

# --------------------------- Complex SH -> FuMa FOA --------------------------- #
_W_SCALE  = 2.0 * np.sqrt(np.pi)
_Z_SCALE  = 2.0 * np.sqrt(np.pi / 3.0)
_XY_SCALE = 2.0 * np.sqrt(2.0 * np.pi / 3.0)


def complex_to_fuma_foa(amb_complex):
    """
    Convert shroom's complex SH output (ACN order, scipy normalization) into
    real-valued FuMa-normalized FOA channels [W, Y, Z, X], matching the
    TAU/DCASE convention (peak gain 1 on each order-1 axis).

    Uses ONLY the m=-1 channel for both X and Y to avoid relying on the
    Condon-Shortley sign of m=+1, which is the most common place for sign
    bugs to hide.

    Parameters
    ----------
    amb_complex : np.ndarray
        Complex array of shape (..., 4, T). SH index order is assumed to be
        [Y_0^0, Y_1^{-1}, Y_1^0, Y_1^{+1}] (ACN).

    Returns
    -------
    np.ndarray
        Real array of shape (..., 4, T), channels [W, Y, Z, X].
    """
    c00  = amb_complex[..., 0, :]
    c1m1 = amb_complex[..., 1, :]
    c10  = amb_complex[..., 2, :]

    W = _W_SCALE  * c00.real
    Y = _XY_SCALE * c1m1.imag
    Z = _Z_SCALE  * c10.real
    X = _XY_SCALE * c1m1.real

    return np.stack([W, Y, Z, X], axis=-2)


# --------------------------- Resampler (one per worker) --------------------------- #
_resampler = None


def _get_resampler():
    """Lazily build a Kaiser-windowed sinc resampler. One per worker process."""
    global _resampler
    if _resampler is None:
        _resampler = Resample(
            orig_freq=int(FS),
            new_freq=OUTPUT_FS,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            beta=14.769656459379492,
        )
    return _resampler


# --------------------------- Worker init + self-test --------------------------- #
def _self_test_encoder():
    """
    Encode synthetic plane waves from known directions through the array model
    and verify the FuMa FOA output. Raises if any direction is off by >15%.
    """
    test_directions_deg = [
        (  0.0,  0.0),
        ( 90.0,  0.0),
        (  0.0, 60.0),
        (180.0,  0.0),
    ]

    n_samples = int(FS * 1.0)
    t = np.arange(n_samples) / FS
    tone = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

    src_grid = _global_array.grid
    src_el = np.pi / 2 - src_grid.co
    src_vecs = np.stack([
        np.cos(src_el) * np.cos(src_grid.az),
        np.cos(src_el) * np.sin(src_grid.az),
        np.sin(src_el),
    ], axis=-1)

    max_err = 0.0
    for az_deg, el_deg in test_directions_deg:
        az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
        target = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        q = int(np.argmax(src_vecs @ target))

        ir = _global_array_time.data[:, q, :]
        mic_sigs = np.stack([
            np.convolve(tone, ir[m].real, mode="same") for m in range(ir.shape[0])
        ], axis=-1).astype(np.float32)

        encoded = _global_asm.encode_amb(mic_sigs)
        amb_c   = np.array(encoded.data).squeeze()
        foa     = complex_to_fuma_foa(amb_c[None]).squeeze()

        seg = slice(n_samples // 4, 3 * n_samples // 4)
        W = foa[0, seg]
        ratios = np.array([
            (foa[1, seg] * W).mean() / (W * W).mean(),
            (foa[2, seg] * W).mean() / (W * W).mean(),
            (foa[3, seg] * W).mean() / (W * W).mean(),
        ])
        expected = np.array([
            np.sin(az) * np.cos(el),
            np.sin(el),
            np.cos(az) * np.cos(el),
        ])
        err = np.abs(ratios - expected)
        max_err = max(max_err, err.max())

        print(f"  (az={az_deg:+6.1f}, el={el_deg:+5.1f})  "
              f"got Y/Z/X = [{ratios[0]:+.3f} {ratios[1]:+.3f} {ratios[2]:+.3f}]  "
              f"exp [{expected[0]:+.3f} {expected[1]:+.3f} {expected[2]:+.3f}]")

    if max_err > 0.15:
        raise RuntimeError(
            f"Encoder self-test failed: max ratio error {max_err:.3f} > 0.15. "
            "Likely sign/convention mismatch in complex_to_fuma_foa()."
        )


def init_worker():
    global asm, mics_grid
    global _global_array, _global_array_time, _global_asm

    mics_grid = sphereicalGrid(
        az=tau_az,
        co=tau_co,
        orientation=np.array([1, 0, 0]),
    )

    source_grid = from_fibonacci_grid(480)

    array = SphericalArray(
        source_grid=source_grid,
        mics_grid=mics_grid,
        r_mics=np.full(mics_grid.n_points, tau_r),
        fs=FS,
        duration=DURATION,
        r_sphere=tau_r,
        sh_order_for_sm_calc=7,
        sphere_type="rigid",
    )

    array_time = array.copy()
    array_time.toTime()
    array.toFreq()

    asm = ASM(sh_order=1, array=array, fs=FS, duration=DURATION)

    _global_array      = array
    _global_array_time = array_time
    _global_asm        = asm

    # Touch the resampler so it's built once per worker, not on first file.
    _get_resampler()


def process_file(args):
    """Process a single wav file — runs in a worker with pre-built asm + resampler."""
    wav_path, out_path = Path(args[0]), Path(args[1])

    if out_path.exists():
        return f"SKIP {wav_path.name}"

    try:
        mic_signals, sr = sf.read(wav_path, always_2d=True)
        if sr != int(FS):
            return f"BAD_SR {wav_path.name} (sr={sr})"

        mic_signals = np.array(mic_signals, dtype=np.float32).T[None, :, :]

        if mic_signals.shape[1] != 4:
            return f"WRONG_CH {wav_path.name}"

        mic_signals = SpatialSignal(
            data=mic_signals,
            fs=FS,
            grid=mics_grid,
            is_time=True,
            is_space=True,
        )

        # --- Encode ASM-FOA at 24 kHz ---
        raw_mics   = mic_signals.data.squeeze().T
        ambisonics = asm.encode_amb(raw_mics)
        amb_c      = np.array(ambisonics.data).squeeze()
        foa_24k    = complex_to_fuma_foa(amb_c[None]).squeeze()  # (4, T) float

        # --- Resample 24 kHz -> 32 kHz ---
        # Torch shape (C, T); keep as float32 throughout.
        foa_t  = torch.from_numpy(foa_24k.astype(np.float32))
        foa_rs = _get_resampler()(foa_t).numpy()                 # (4, T') float32

        amb_data = foa_rs.T                                      # (T', 4)

        # --- Headroom check, then int16 ---
        peak = float(np.abs(amb_data).max())
        if not np.isfinite(peak):
            return f"FAIL {wav_path.name}: non-finite output"
        note = ""
        if peak >= 1.0:
            amb_data = amb_data * (0.999 / peak)
            note = f" (peak={peak:.3f}, scaled)"
        amb_i16 = np.clip(amb_data * 32767.0, -32768, 32767).astype(np.int16)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), amb_i16, OUTPUT_FS, subtype="PCM_16")
        return f"OK {wav_path.name}{note}"

    except Exception as e:
        return f"FAIL {wav_path.name}: {e}"


def collect_tasks(mic_root: Path, foa_root: Path):
    if not mic_root.exists():
        print(f"  [missing] {mic_root}")
        return []
    wavs = sorted(mic_root.rglob("*.wav"))
    print(f"  {mic_root}: {len(wavs)} files")
    return [(str(w), str(foa_root / w.relative_to(mic_root))) for w in wavs]


if __name__ == "__main__":
    print("Running encoder self-test...")
    init_worker()
    _self_test_encoder()
    print("Self-test passed.\n")

    datasets = [
        # (Path("tau2020/mic_dev"),    Path("tau2020/foa_dev_asm_new")),
        (Path("tau2021/mic_dev"),      Path("tau2021/foa_dev_asm")),
        # (Path("starss2023/mic_dev"), Path("starss2023/foa_dev_asm")),
    ]

    all_tasks = []
    for mic_root, foa_root in datasets:
        all_tasks.extend(collect_tasks(mic_root, foa_root))

    print(f"\nTotal files: {len(all_tasks)}")

    num_workers = 16
    print(f"Launching {num_workers} workers...\n")

    with Pool(num_workers, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, all_tasks), 1):
            tag, *rest = result.split(" ", 1)
            label = {
                "OK": "✓", "SKIP": "⟳", "FAIL": "✗",
                "WRONG_CH": "⚠", "BAD_SR": "⚠",
            }.get(tag, "?")
            print(f"[{i:>5}/{len(all_tasks)}] {label} {rest[0] if rest else ''}")

    print("\nDone.")