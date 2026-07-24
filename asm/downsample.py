import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly
from multiprocessing import Pool, cpu_count

FS_IN        = 48000
FS_OUT       = 32000
UP, DOWN     = 2, 3
KAISER_BETA  = 14


def process_file(args):
    wav_path, out_path = Path(args[0]), Path(args[1])

    if out_path.exists():
        return "SKIP", wav_path.name, ""

    try:
        audio, fs = sf.read(wav_path, always_2d=True)
        assert fs == FS_IN, f"Expected {FS_IN} Hz, got {fs} Hz"

        resampled = resample_poly(
            audio, up=UP, down=DOWN, axis=0,
            window=("kaiser", KAISER_BETA),
        ).astype(np.float32)

        sf.write(str(out_path), resampled, FS_OUT, subtype="FLOAT")
        return "OK", wav_path.name, f"({audio.shape[0]} → {resampled.shape[0]} samples, {audio.shape[1]} ch)"

    except Exception as e:
        return "FAIL", wav_path.name, str(e)


if __name__ == "__main__":
    LABELS = {"OK": "✓", "SKIP": "⟳", "FAIL": "✗"}

    for split in ["train", "valid", "test"]:
        input_dir  = Path(f"./{split}")
        output_dir = input_dir / "32000"
        output_dir.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(input_dir.glob("*.wav"))
        print(f"\n── {split} ── {len(wav_files)} files  ({FS_IN} Hz → {FS_OUT} Hz)")

        tasks = [(str(p), str(output_dir / p.name)) for p in wav_files]
        ok, skipped, failed = 0, 0, 0

        with Pool(cpu_count()) as pool:
            for i, (status, name, info) in enumerate(
                pool.imap_unordered(process_file, tasks), 1
            ):
                label = LABELS.get(status, "?")
                print(f"  [{i:>5}/{len(tasks)}] {label} {name}  {info}")
                if status == "OK":   ok      += 1
                if status == "SKIP": skipped += 1
                if status == "FAIL": failed  += 1

        print(f"\n  ✓ {ok}  ⟳ {skipped}  ✗ {failed}")
        print(f"  Output: {output_dir.resolve()}")