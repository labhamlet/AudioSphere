import json
import math
import numpy as np
from scipy.signal import stft, find_peaks
import soundfile as sf
import os
from itertools import permutations
from multiprocessing import Pool, cpu_count

def az_to_cartesian(az_deg: float, el_deg: float = 0.0, r: float = 1.0) -> tuple[float, float, float]:
    az_rad = math.radians(az_deg)
    el_rad = math.radians(el_deg)
    x = -r * math.cos(el_rad) * math.cos(az_rad)
    y =  r * math.cos(el_rad) * math.sin(az_rad)
    z =  r * math.sin(el_rad)
    return x, y, z

def unit_vector_to_azimuth(uv):
    return np.arctan2(uv[1], -uv[0])

respeaker_az = np.array([-45, 45, 135, -135]) 
respeaker_r = 0.0325
mic_pos = np.array([az_to_cartesian(az, 0, respeaker_r) for az in respeaker_az])

CANDIDATE_ANGLES = np.linspace(-np.pi, np.pi, 360, endpoint=False)

def resolve_wav_path(filename):
    parts = filename.split("_")
    dist, az = parts[1], parts[2]
    return os.path.join("RSL2019", f"{dist}cm", f"RSL_{dist}_{az}", filename)

def music_spectrum(X, num_sources, angles, fs=48000, c=343):
    freq_bins, _, Z = stft(X, fs=fs, nperseg=1024, noverlap=768)
    num_mics, num_freqs, num_frames = Z.shape
    spectrum = np.zeros(len(angles))

    for f_idx in range(len(freq_bins)):
        freq = freq_bins[f_idx]
        Zf = Z[:, f_idx, :]
        Rxx = (Zf @ Zf.conj().T) / num_frames
        Rxx += 1e-6 * np.eye(num_mics)

        eigenvalues, eigenvectors = np.linalg.eigh(Rxx)
        noise_subspace = eigenvectors[:, :num_mics - num_sources]

        for i, theta in enumerate(angles):
            d_vec = np.array([-np.cos(theta), np.sin(theta), 0.0])
            delays = mic_pos @ d_vec / c
            a = np.exp(-1j * 2 * np.pi * freq * delays)[:, np.newaxis]
            denom = a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a
            spectrum[i] += 1.0 / (np.abs(denom.item()) + 1e-15)

    return spectrum

def cartesian_to_spherical(x, y, z):
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x**2 + y**2))
    return az, el

def doe_single(pred_xyz, target_xyz):
    pred_az, pred_el = cartesian_to_spherical(pred_xyz[0], pred_xyz[1], pred_xyz[2])
    tgt_az, tgt_el = cartesian_to_spherical(target_xyz[0], target_xyz[1], target_xyz[2])
    cos_dist = (np.sin(tgt_el) * np.sin(pred_el) +
                np.cos(tgt_el) * np.cos(pred_el) * np.cos(tgt_az - pred_az))
    return float(np.rad2deg(np.arccos(np.clip(cos_dist, -1.0, 1.0))))

def angle_to_unit_xyz(theta):
    """Convert MUSIC azimuth back to unit vector in the dataset's coordinate system."""
    return np.array([-np.cos(theta), np.sin(theta), 0.0])

def process_one(args):
    wav_file, sources = args
    path = resolve_wav_path(wav_file)
    if not os.path.exists(path):
        return None

    audio, fs = sf.read(path)
    audio = audio.T[:4, :]

    num_sources = len(sources)
    targets = [np.array(s) for s in sources]

    spec = music_spectrum(audio, num_sources, CANDIDATE_ANGLES, fs=fs)

    peaks, _ = find_peaks(spec, distance=20)
    if len(peaks) < num_sources:
        top_indices = np.argsort(spec)[-num_sources:]
    else:
        top_indices = peaks[np.argsort(spec[peaks])[-num_sources:]]

    est_xyzs = [angle_to_unit_xyz(CANDIDATE_ANGLES[idx]) for idx in top_indices]

    # Best permutation by DOE
    best_mean_err = float('inf')
    best_perm = est_xyzs
    for p in permutations(est_xyzs):
        errs = [doe_single(e, t) for e, t in zip(p, targets)]
        m = np.mean(errs)
        if m < best_mean_err:
            best_mean_err = m
            best_perm = list(p)

    print(f"{wav_file}: DOE={best_mean_err:.1f}°", flush=True)
    return best_mean_err, [xyz.tolist() for xyz in best_perm]

if __name__ == "__main__":
    with open("test.json", "r") as f:
        test_data = json.load(f)

    items = list(test_data.items())

    with Pool(processes=16) as pool:
        results = pool.map(process_one, items)

    predictions = {}
    errors = []
    for (wav_file, _), result in zip(items, results):
        if result is None:
            continue
        err_deg, pred_xyzs = result
        errors.append(err_deg)
        predictions[wav_file] = pred_xyzs

    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nFinal Test DOE (median): {np.median(errors):.2f}°")
    print(f"Final Test DOE (mean):   {np.mean(errors):.2f}°")
    print(f"Final Test DOE (std):    {np.std(errors):.2f}°")