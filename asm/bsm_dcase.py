import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import Resample
from pathlib import Path
from multiprocessing import Pool

from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.hrtf_processing import array_aware_magls_hrtf
from shroom.encoders.bsm import BSM
from shroom.encoders.asm import ASM
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.utils.file_utils import load_file
from shroom.acoustics.spatial_signal import SpatialSignal

# --- Config ---
FS                = 24000.0   # MIC files' native sample rate (input to encoder)
OUTPUT_FS         = 32000     # target sample rate for downstream pipeline
DURATION          = 0.008
SM_SH_ORDER       = 20        # SH order for steering matrix expansion (was 7)
HRTF_SH_ORDER     = 30        # SH order for HRTF interpolation onto array grid
AA_MAGLS_SH_ORDER = 1         # SH order used by the AA-MagLS correction
AA_MAGLS_CUTOFF   = 1200.0    # Hz — LS below, AA-MagLS magnitude-only above
HRTF_PATH         = "/projects/0/prjs1338/locata/shroom/src/shroom/data/default_hrtf.sofa"

# Tetrahedral MIC array — TAU2020 / TAU2021 / STARSS2023
#   M1: ( 45°,  35°, 4.2 cm)
#   M2: (-45°, -35°, 4.2 cm)
#   M3: (135°, -35°, 4.2 cm)
#   M4: (-135°, 35°, 4.2 cm)
tau_az = np.array([45, -45, 135, -135]) * np.pi / 180
tau_el = np.array([35, -35, -35,  35]) * np.pi / 180
tau_co = (np.pi / 2) - tau_el          # elevation -> co-latitude
tau_r  = 0.042                         # 4.2 cm spherical baffle


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


# --------------------------- Worker init + smoke test --------------------------- #
def _smoke_test_encoder():
    """
    Encode a synthetic plane wave from a lateral direction through the array
    model and verify the BSM output is sane:
      - shape is (2, T)
      - finite
      - non-silent
      - shows some lateralization (non-trivial ILD)
    """
    n_samples = int(FS * 1.0)
    t = np.arange(n_samples) / FS

    # Use:
    rng = np.random.default_rng(0)
    tone = rng.standard_normal(n_samples).astype(np.float32) * 0.1  # broadband noise

    src_grid = _global_array_time.grid
    src_el = np.pi / 2 - src_grid.co
    src_vecs = np.stack([
        np.cos(src_el) * np.cos(src_grid.az),
        np.cos(src_el) * np.sin(src_grid.az),
        np.sin(src_el),
    ], axis=-1)

    # Lateral source at az=+90°, el=0° — far from median plane, expect strong ILD
    target = np.array([0.0, 1.0, 0.0])
    q = int(np.argmax(src_vecs @ target))

    ir = _global_array_time.data[:, q, :]
    mic_sigs = np.stack([
        np.convolve(tone, ir[m].real, mode="same") for m in range(ir.shape[0])
    ], axis=0).astype(np.float32)  # (4, T)

    sig = SpatialSignal(
        data=mic_sigs[:, None, :],
        fs=FS,
        is_time=True,
        is_space=False,
    )
    out = bsm.process(sig)
    binaural = out.data[:, 0, :].real  # (2, T)

    if binaural.shape[0] != 2:
        raise RuntimeError(f"Expected 2 binaural channels, got shape {binaural.shape}")
    if not np.isfinite(binaural).all():
        raise RuntimeError("BSM output has non-finite values")

    seg = slice(n_samples // 4, 3 * n_samples // 4)
    rms_l = float(np.sqrt(np.mean(binaural[0, seg] ** 2)))
    rms_r = float(np.sqrt(np.mean(binaural[1, seg] ** 2)))
    if rms_l < 1e-6 and rms_r < 1e-6:
        raise RuntimeError("BSM output is silent")
    ild_db = 20.0 * np.log10((rms_l + 1e-12) / (rms_r + 1e-12))

    print(f"  source at az=+90°, el=0°:  RMS L={rms_l:.4f}  R={rms_r:.4f}  ILD={ild_db:+.1f} dB")
    if abs(ild_db) < 1.0:
        print(f"  [warn] ILD magnitude < 1 dB at lateral source — check HRTF convention")


def init_worker():
    global bsm, mics_grid, _global_array_time

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
        sh_order_for_sm_calc=SM_SH_ORDER,
        sphere_type="rigid",
    )

    # Time-domain copy for the smoke test (mic impulse responses per direction).
    # Take this BEFORE converting `array` to freq so we don't bounce domains.
    array_time = array.copy()

    # Convert original to frequency domain for BSM
    array.toFreq()
    array_freq = array

    # ASM encoder built only to feed into array_aware_magls_hrtf — it
    # characterizes how the array maps to SH space, which AA-MagLS needs to
    # compute the magnitude correction. Not used for actual encoding (BSM
    # does that).
    asm_for_aa_magls = ASM(
        sh_order=AA_MAGLS_SH_ORDER,
        array=array_freq,
        fs=FS,
        duration=DURATION,
    )

    # HRTF: project onto the array's source grid in freq + space domain.
    # Order matches the AA-MagLS example: resample -> zero_pad -> toFreq -> toSH -> toSpace.
    hrtf = load_file(HRTF_PATH)
    hrtf.resample(desired_fs=FS)
    hrtf.zero_pad(int(DURATION * FS))
    hrtf.toFreq()
    hrtf.toSH(HRTF_SH_ORDER)
    hrtf.toSpace(array_freq.grid)

    # AA-MagLS HRTF correction
    hrtf_aa_magls = array_aware_magls_hrtf(
        hrtf=hrtf,
        asm=asm_for_aa_magls,
        array=array_freq,
        sh_order=AA_MAGLS_SH_ORDER,
        cutoff_over_freq=AA_MAGLS_CUTOFF,
    )
    # Make sure we end up in space + freq domain for BSM. AA-MagLS / toSpace
    # may leave the HRTF in SH or time domain; these calls are no-ops if
    # already in the target state.
    hrtf_aa_magls.toSpace(array_freq.grid)
    if not hrtf_aa_magls.is_freq:
        hrtf_aa_magls.toFreq()

    bsm = BSM(
        array=array_freq,
        hrtf=hrtf_aa_magls,
        fs=FS,
        duration=DURATION,
        use_magls=False,    # AA-MagLS correction already applied to the HRTF
    )

    _global_array_time = array_time

    # Touch the resampler so it's built once per worker, not on first file.
    _get_resampler()


def process_file(args):
    """Process a single wav file — runs in a worker with pre-built bsm + resampler."""
    wav_path, out_path = Path(args[0]), Path(args[1])

    if out_path.exists():
        return f"SKIP {wav_path.name}"

    try:
        mic_signals, sr = sf.read(wav_path, always_2d=True)
        if sr != int(FS):
            return f"BAD_SR {wav_path.name} (sr={sr})"
        if mic_signals.ndim != 2 or mic_signals.shape[1] != 4:
            return f"WRONG_CH {wav_path.name} (shape={mic_signals.shape})"

        sig = SpatialSignal(
            data=np.ascontiguousarray(mic_signals.T, dtype=np.float32)[:, None, :],
            fs=FS,
            is_time=True,
            is_space=False,
        )

        # --- Encode BSM binaural at 24 kHz ---
        binaural = bsm.process(sig)
        bin_24k = binaural.data[:, 0, :].real.astype(np.float32)   # (2, T)

        # --- Resample 24 kHz -> 32 kHz ---
        bin_t  = torch.from_numpy(bin_24k)
        bin_rs = _get_resampler()(bin_t).numpy()                   # (2, T')
        bin_out = bin_rs.T                                          # (T', 2)

        # --- Headroom check, then int16 ---
        peak = float(np.abs(bin_out).max())
        if not np.isfinite(peak):
            return f"FAIL {wav_path.name}: non-finite output"
        note = ""
        if peak >= 1.0:
            bin_out = bin_out * (0.999 / peak)
            note = f" (peak={peak:.3f}, scaled)"
        bin_i16 = np.clip(bin_out * 32767.0, -32768, 32767).astype(np.int16)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), bin_i16, OUTPUT_FS, subtype="PCM_16")
        return f"OK {wav_path.name}{note}"

    except Exception as e:
        return f"FAIL {wav_path.name}: {e}"


def collect_tasks(mic_root: Path, out_root: Path):
    """Mirror every .wav under mic_root into out_root, preserving sub-paths."""
    if not mic_root.exists():
        print(f"  [missing] {mic_root}")
        return []
    wavs = sorted(mic_root.rglob("*.wav"))
    print(f"  {mic_root}: {len(wavs)} files")
    return [(str(w), str(out_root / w.relative_to(mic_root))) for w in wavs]


if __name__ == "__main__":
    print("Running encoder smoke test...")
    init_worker()
    _smoke_test_encoder()
    print("Smoke test passed.\n")

    datasets = [
        # (Path("tau2020/mic_dev"),    Path("tau2020/bsm_dev")),
        # (Path("tau2021/mic_dev"),      Path("tau2021/bsm_dev")),
        (Path("starss2023/mic_dev"), Path("starss2023/binaural_dev")),
    ]

    all_tasks = []
    for mic_root, out_root in datasets:
        all_tasks.extend(collect_tasks(mic_root, out_root))

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