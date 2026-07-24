import numpy as np
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool, cpu_count
from shroom.acoustics.spherical_array import SphericalArray
from shroom.encoders.asm import ASM
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.acoustics.spatial_signal import SpatialSignal

FS       = 48000.0
DURATION = 0.008

respeaker_az = np.array([135, -135, -45, 45]) * np.pi / 180
respeaker_co = np.array([90,  90,   90,  90]) * np.pi / 180
respeaker_r  = 0.0325

def init_worker():
    global asm, mics_grid

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
        sh_order_for_sm_calc=7,
    )

    array.toFreq()
    array_time_sh = array.copy()
    array_time_sh.toTime()
    array_time_sh.toSH(1)

    asm = ASM(sh_order=1, array=array, fs=FS, duration=DURATION)


def process_file(args):
    """Process a single wav file — runs in a worker with pre-built asm."""
    wav_path, out_path = Path(args[0]), Path(args[1])

    if out_path.exists():
        return f"SKIP {wav_path.name}"

    try:
        mic_signals, _ = sf.read(wav_path, always_2d=True)
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

        raw_mics  = mic_signals.data.squeeze().T
        ambisonics = asm.encode_amb(raw_mics)
        amb_data   = np.array(ambisonics.data).squeeze().T.real

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), amb_data.astype(np.float32), int(FS), subtype='FLOAT')
        return f"OK {wav_path.name}"

    except Exception as e:
        return f"FAIL {wav_path.name}: {e}"


if __name__ == "__main__":
    OUTPUT_DIR = Path("RSL2019/bsm")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build all (wav_path, out_path) pairs across all degrees up front
    all_tasks = []
    for m in range(0, 360, 5):
        input_dir = Path(f"RSL2019/100cm/RSL_100_{m}")
        wav_files = sorted(input_dir.glob("*.wav"))
        print(f"  {input_dir}: {len(wav_files)} files")
        for wav_path in wav_files:
            out_path = OUTPUT_DIR / wav_path.name
            all_tasks.append((str(wav_path), str(out_path)))

    for m in range(0, 360, 5):
        input_dir = Path(f"RSL2019/150cm/RSL_150_{m}")
        wav_files = sorted(input_dir.glob("*.wav"))
        print(f"  {input_dir}: {len(wav_files)} files")
        for wav_path in wav_files:
            out_path = OUTPUT_DIR / wav_path.name
            all_tasks.append((str(wav_path), str(out_path)))

    print(f"\nTotal files: {len(all_tasks)}")

    num_workers = 16
    print(f"Launching {num_workers} workers...\n")

    with Pool(num_workers, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, all_tasks), 1):
            tag, *rest = result.split(" ", 1)
            label = {"OK": "✓", "SKIP": "⟳", "FAIL": "✗", "WRONG_CH": "⚠"}.get(tag, "?")
            print(f"[{i:>5}/{len(all_tasks)}] {label} {rest[0] if rest else ''}")

    print("\nDone.")