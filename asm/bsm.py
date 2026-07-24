import numpy as np
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool, cpu_count

from shroom.acoustics.spherical_array import SphericalArray
from shroom.encoders.bsm import BSM
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.utils.file_utils import load_file
from shroom.acoustics.spatial_signal import SpatialSignal

FS       = 48000.0
DURATION = 0.008

HRTF_SH_ORDER    = 30
MAGLS_CUTOFF_FREQ = 1200.0

respeaker_az = np.array([135, -135, -45, 45]) * np.pi / 180
respeaker_co = np.array([90,  90,   90,  90]) * np.pi / 180
respeaker_r  = 0.0325


def init_worker():
    global bsm

    mics_grid = sphereicalGrid(
        az=respeaker_az,
        co=respeaker_co,
        orientation=np.array([1, 0, 0]),
    )

    source_grid = from_fibonacci_grid(480)
    r_mics = np.full(mics_grid.n_points, respeaker_r)

    array = SphericalArray(
        source_grid=source_grid,
        mics_grid=mics_grid,
        r_mics=r_mics,
        fs=FS,
        duration=DURATION,
        r_sphere=respeaker_r,
        sh_order_for_sm_calc=20,
        sphere_type='rigid',
        convert_to_time=False,
    )

    array.resample(desired_fs=FS)
    array_freq = array.copy()
    array_freq.toFreq()

    # Load HRTF
    hrtf = load_file("/projects/0/prjs1338/locata/shroom/src/shroom/data/default_hrtf.sofa")
    hrtf.resample(desired_fs=FS)
    hrtf.toSH(HRTF_SH_ORDER)
    hrtf.toSpace(array_freq.grid)
    hrtf.zero_pad(int(DURATION * FS))
    hrtf.toFreq()

    bsm = BSM(
        array=array_freq,
        hrtf=hrtf,
        fs=FS,
        duration=DURATION,
        use_magls=True,
        magls_cutoff_frequency=MAGLS_CUTOFF_FREQ,
    )


def process_file(args):
    wav_path, out_path = Path(args[0]), Path(args[1])

    if out_path.exists():
        return f"SKIP {wav_path.name}"

    try:
        mic_signals, _ = sf.read(wav_path, always_2d=True)
        mic_signals = np.array(mic_signals, dtype=np.float32)

        if mic_signals.shape[1] != 4:
            return f"WRONG_CH {wav_path.name}"

        # (n_samples, n_mics) -> (n_mics, 1, n_samples)
        mic_data = mic_signals.T[:, None, :]

        mic_spatial = SpatialSignal(
            data=mic_data,
            fs=FS,
            is_time=True,
            is_space=False,
        )

        binaural = bsm.process(mic_spatial)
        # (2, 1, T) -> (T, 2)
        binaural_data = binaural.data[:, 0, :].real.T.astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), binaural_data, int(FS), subtype='FLOAT')
        return f"OK {wav_path.name}"

    except Exception as e:
        return f"FAIL {wav_path.name}: {e}"


if __name__ == "__main__":
    OUTPUT_DIR = Path("RSL2019/binaural")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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