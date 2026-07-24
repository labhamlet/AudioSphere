import numpy as np
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool

from shroom.acoustics.spherical_array import SphericalArray
from shroom.encoders.asm import ASM
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid

FS = 48000.0
DURATION = 0.008

respeaker_az = np.array([-135, 135, 45, -45]) * np.pi / 180   # ch0..ch3
respeaker_co = np.array([  90,  90, 90,  90]) * np.pi / 180   # planar board
respeaker_r  = 0.0325

SM_ORDER = 14
SPHERE_TYPE = "open"

_W_SCALE = 2.0 * np.sqrt(np.pi)
_Z_SCALE = 2.0 * np.sqrt(np.pi / 3.0)
_XY_SCALE = 2.0 * np.sqrt(2.0 * np.pi / 3.0)


def complex_to_sn3d_foa(amb_complex):
    """
    (..., 4, T) complex SH in ACN order -> (..., 4, T) real [W, Y, Z, X],
    ACN/SN3D. Uses only the m=-1 channel for X and Y to sidestep the
    Condon-Shortley sign of m=+1.
    """
    c00  = amb_complex[..., 0, :]
    c1m1 = amb_complex[..., 1, :]
    c10  = amb_complex[..., 2, :]

    W = _W_SCALE  * c00.real
    Y = _XY_SCALE * c1m1.imag
    Z = _Z_SCALE  * c10.real
    X = _XY_SCALE * c1m1.real

    return np.stack([W, Y, Z, X], axis=-2)


def init_worker():
    global asm, _global_array, _global_array_time

    mics_grid = sphereicalGrid(
        az=respeaker_az,
        co=respeaker_co,
        orientation=np.array([1, 0, 0]),
    )

    source_grid = from_fibonacci_grid(480)

    array = SphericalArray(
        source_grid=source_grid,
        mics_grid=mics_grid,
        r_mics=np.full(mics_grid.n_points, respeaker_r),
        fs=FS,
        duration=DURATION,
        r_sphere=respeaker_r,
        sh_order_for_sm_calc=SM_ORDER,
        sphere_type=SPHERE_TYPE,
    )

    array_time = array.copy()
    array_time.toTime()
    array.toFreq()

    asm = ASM(sh_order=1, array=array, fs=FS, duration=DURATION)

    _global_array      = array
    _global_array_time = array_time


def _self_test_encoder():
    """
    Encode synthetic plane waves from known horizontal directions through the
    array model and verify the SN3D FOA output via E[ch*W]/E[W*W] ratios.
    Azimuth-only (planar array => Z is unobservable and is NOT checked).
    """
    test_az_deg = [0.0, 90.0, -45.0, 180.0]

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
    for az_deg in test_az_deg:
        az = np.deg2rad(az_deg)
        target = np.array([np.cos(az), np.sin(az), 0.0])
        q = int(np.argmax(src_vecs @ target))

        ir = _global_array_time.data[:, q, :]
        mic_sigs = np.stack([
            np.convolve(tone, ir[m].real, mode="same") for m in range(ir.shape[0])
        ], axis=-1).astype(np.float32)

        encoded = asm.encode_amb(mic_sigs)
        amb_c   = np.array(encoded.data).squeeze()
        foa     = complex_to_sn3d_foa(amb_c[None]).squeeze()

        seg = slice(n_samples // 4, 3 * n_samples // 4)
        W = foa[0, seg]
        ratio_y = (foa[1, seg] * W).mean() / (W * W).mean()
        ratio_x = (foa[3, seg] * W).mean() / (W * W).mean()
        exp_y, exp_x = np.sin(az), np.cos(az)

        err = max(abs(ratio_y - exp_y), abs(ratio_x - exp_x))
        max_err = max(max_err, err)
        print(f"  az={az_deg:+6.1f}  got Y/X = [{ratio_y:+.3f} {ratio_x:+.3f}]"
              f"  exp [{exp_y:+.3f} {exp_x:+.3f}]")

    if max_err > 0.15:
        raise RuntimeError(
            f"Encoder self-test failed: max ratio error {max_err:.3f} > 0.15. "
            "Likely sign/convention mismatch in complex_to_sn3d_foa() or the "
            "mic grid.")


def process_file(args):
    """Process a single wav file — runs in a worker with pre-built asm."""
    wav_path, out_path = Path(args[0]), Path(args[1])

    if out_path.exists():
        return f"SKIP {wav_path.name}"

    try:
        mic_signals, sr = sf.read(wav_path, always_2d=True)   # (T, M)
        # FIX 4: don't silently assume the sample rate.
        if sr != int(FS):
            return f"BAD_SR {wav_path.name} (sr={sr})"
        if mic_signals.shape[1] != 4:
            return f"WRONG_CH {wav_path.name}"

        ambisonics = asm.encode_amb(mic_signals.astype(np.float32))
        amb_c = np.array(ambisonics.data).squeeze()          # (4, T) complex
        foa = complex_to_sn3d_foa(amb_c[None]).squeeze()   # (4, T) real

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), foa.T.astype(np.float32), int(FS), subtype='FLOAT')
        return f"OK {wav_path.name}"

    except Exception as e:
        return f"FAIL {wav_path.name}: {e}"


def collect_tasks(out_dir: Path):
    all_tasks = []
    for dist in (100, 150):
        for m in range(0, 360, 5):
            input_dir = Path(f"RSL2019/{dist}cm/RSL_{dist}_{m}")
            wav_files = sorted(input_dir.glob("*.wav"))
            print(f"  {input_dir}: {len(wav_files)} files")
            for wav_path in wav_files:
                all_tasks.append((str(wav_path), str(out_dir / wav_path.name)))
    return all_tasks


if __name__ == "__main__":
    OUTPUT_DIR = Path("RSL2019/asm_foa")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    init_worker()
    _self_test_encoder()

    all_tasks = collect_tasks(OUTPUT_DIR)
    print(f"\nTotal files: {len(all_tasks)}")

    num_workers = 16
    print(f"Launching {num_workers} workers...\n")

    with Pool(num_workers, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, all_tasks), 1):
            tag, *rest = result.split(" ", 1)
            label = {"OK": "✓", "SKIP": "⟳", "FAIL": "✗",
                     "WRONG_CH": "⚠", "BAD_SR": "⚠"}.get(tag, "?")
            print(f"[{i:>5}/{len(all_tasks)}] {label} {rest[0] if rest else ''}")

    print("\nDone.")