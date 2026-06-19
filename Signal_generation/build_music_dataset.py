"""
Build an ML-ready music dataset:
  - 50 tracks from GTZAN (via HuggingFace datasets/marsyas/gtzan)
  - 17 jazz + 17 metal + 16 pop (balanced, total 50)
  - Each track saved both clean and noisy (same composite EMC+measurement
    noise model as signal_contamination.py, independent realisation per
    track, target SNR set by NOISY_SNR_DB below)
  - Labels written to out/Model/labels.csv for downstream training

Output layout:
  out/Model/
    clean/<genre>_<id>.wav
    noisy/<genre>_<id>.wav
    labels.csv
"""

import os
import csv
import hashlib
import numpy as np
import soundfile as sf
from datasets import load_dataset

from signal_contamination import make_noise

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "out", "Model")
CLEAN_DIR = os.path.join(OUT_ROOT, "clean")
NOISY_DIR = os.path.join(OUT_ROOT, "noisy")
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)

GENRE_COUNTS = {"jazz": 84, "metal": 83, "pop": 83}   # total 250 (as equal as possible)
NOISY_SNR_DB = -10.0                                  # heavy contamination baseline
GLOBAL_SEED = 42


def stable_seed(text):
    """Deterministic 32-bit seed from a string (for per-track noise)."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def main():
    print("Loading GTZAN (marsyas/gtzan)...")
    ds = load_dataset("marsyas/gtzan", split="train", trust_remote_code=True)
    genre_names = ds.features["genre"].names

    # Build index of row positions per genre we care about.
    # Use whole-column accessor so we don't decode audio for indexing.
    wanted_ids = {g: genre_names.index(g) for g in GENRE_COUNTS}
    id_to_genre = {gid: g for g, gid in wanted_ids.items()}
    all_genre_ids = ds["genre"]
    per_genre_indices = {g: [] for g in GENRE_COUNTS}
    for i, g_id in enumerate(all_genre_ids):
        if g_id in id_to_genre:
            per_genre_indices[id_to_genre[g_id]].append(i)

    rng = np.random.default_rng(GLOBAL_SEED)
    rows = []
    for genre, count in GENRE_COUNTS.items():
        pool = per_genre_indices[genre]
        if len(pool) < count:
            raise RuntimeError(f"only {len(pool)} {genre} tracks found, need {count}")
        picks = sorted(rng.choice(len(pool), size=count, replace=False).tolist())
        print(f"  {genre}: picking {count} / {len(pool)}")
        for seq, k in enumerate(picks, start=1):
            row_idx = pool[k]
            item = ds[row_idx]
            arr = np.asarray(item["audio"]["array"], dtype=np.float32)
            fs = int(item["audio"]["sampling_rate"])
            source_file = os.path.basename(item["file"])
            file_id = f"{genre}_{seq:02d}"

            # make noise with a per-track seed so every file has an
            # independent realisation of the same statistical model.
            emc, meas, total = make_noise(arr.astype(np.float64), fs,
                                          seed=stable_seed(file_id))
            # pipeline in signal_contamination already scales to SNR_TARGET_DB
            # which is set globally in that module. Re-scale here if we want
            # a local override:
            from signal_contamination import SNR_TARGET_DB as DEFAULT_SNR
            if NOISY_SNR_DB != DEFAULT_SNR:
                p_sig = float(np.mean(arr ** 2))
                p_noise_cur = float(np.mean(total ** 2))
                target_p = p_sig / (10 ** (NOISY_SNR_DB / 10))
                k_scale = np.sqrt(target_p / p_noise_cur) if p_noise_cur > 0 else 1.0
                emc *= k_scale
                meas *= k_scale
                total *= k_scale

            noisy = arr.astype(np.float64) + total
            # ADC clip at 3x clean peak
            fs_full = 3.0 * (float(np.max(np.abs(arr))) + 1e-12)
            pre_peak = float(np.max(np.abs(noisy)))
            was_clipped = pre_peak > fs_full
            noisy = np.clip(noisy, -fs_full, fs_full)

            clean_path = os.path.join(CLEAN_DIR, f"{file_id}.wav")
            noisy_path = os.path.join(NOISY_DIR, f"{file_id}.wav")
            sf.write(clean_path, arr, fs, subtype="PCM_16")
            # normalise to int16 range preserving sign
            peak = max(float(np.max(np.abs(noisy))), 1e-12)
            scale_int = 0.98 / peak
            noisy_i16 = np.clip(noisy * scale_int * 32767, -32768, 32767).astype(np.int16)
            sf.write(noisy_path, noisy_i16, fs, subtype="PCM_16")

            p_sig = float(np.mean(arr ** 2))
            p_noise = float(np.mean(total ** 2))
            snr_achieved = 10 * np.log10(p_sig / p_noise) if p_noise > 0 else float("nan")

            rows.append({
                "file_id": file_id,
                "genre": genre,
                "source_file": source_file,
                "fs": fs,
                "duration_s": round(len(arr) / fs, 3),
                "samples": len(arr),
                "target_snr_db": NOISY_SNR_DB,
                "achieved_snr_db": round(snr_achieved, 3),
                "clean_path": os.path.relpath(clean_path, OUT_ROOT).replace("\\", "/"),
                "noisy_path": os.path.relpath(noisy_path, OUT_ROOT).replace("\\", "/"),
                "clipped": was_clipped,
            })
            print(f"    {file_id}: {len(arr)/fs:.1f}s @ {fs} Hz  "
                  f"(SNR {snr_achieved:+.2f} dB, clipped={was_clipped})")

    labels_path = os.path.join(OUT_ROOT, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} tracks under {OUT_ROOT}")
    print(f"Labels CSV: {labels_path}")
    # small summary
    by_genre = {}
    for r in rows:
        by_genre.setdefault(r["genre"], []).append(r["achieved_snr_db"])
    for g, snrs in by_genre.items():
        print(f"  {g}: n={len(snrs)}, mean SNR {np.mean(snrs):+.2f} dB, "
              f"std {np.std(snrs):.2f} dB")


if __name__ == "__main__":
    main()
