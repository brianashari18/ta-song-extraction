"""
extract_features.py
====================
Pipeline ekstraksi fitur audio (MFCC & Chromagram) dari SEMUA subfolder
instrumen di input/ berdasarkan metadata CSV.

Subfolder yang di‑scan otomatis:
  input/vocals, input/guitar, input/piano, input/bass,
  input/guitar_piano, input/guitar_piano_bass, input/no_vocals

Output di‑kelompokkan per subfolder:
  output/bass/extracted_features_mfcc.pkl         + .csv
  output/bass/extracted_features_chromagram.pkl    + .csv
  output/bass/extracted_features_mfcc_chroma.pkl   + .csv
  output/guitar/...
  output/vocals/...
  dst.

Cara pakai:
  pip install librosa pandas numpy tqdm
  python extract_features.py
"""

import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── KONFIGURASI ────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "master_dataset_lstm_ready.csv")
INPUT_DIR  = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

SR          = 22050      # sample‑rate librosa
N_MFCC      = 13         # koefisien MFCC
N_CHROMA    = 12         # pitch classes chroma
MIN_SAMPLES = 2048       # minimum samples agar STFT tidak error

# Kolom metadata asli yang ikut di‑carry
META_COLS = [
    "title", "artist", "part_song",
    "chord_absolute", "roman_numeral",
    "start_time", "end_time",
]


# ─────────────────────── HELPER FUNCTIONS ─────────────────────────────────
def _build_track_id(title: str, artist: str) -> str:
    """(artist, title) → track_id slug, e.g. 'vierratale-rasa_ini'."""
    artist_slug = re.sub(r"\s+", "_", artist.strip().lower())
    title_slug  = re.sub(r"\s+", "_", title.strip().lower())
    return f"{artist_slug}-{title_slug}"


def _wav_path(track_id: str, instrument: str) -> str:
    """track_id + instrument → full path WAV."""
    return os.path.join(INPUT_DIR, instrument,
                        f"{track_id}_{instrument}.wav")


def _load_and_cache(track_id: str, instrument: str,
                    cache: dict) -> np.ndarray | None:
    """Load audio sekali per lagu, simpan ke dict cache."""
    key = f"{track_id}_{instrument}"
    if key in cache:
        return cache[key]
    path = _wav_path(track_id, instrument)
    if not os.path.isfile(path):
        return None
    y, _ = librosa.load(path, sr=SR)
    cache[key] = y
    return y


def _slice_audio(y_full: np.ndarray, start: float, end: float) -> np.ndarray:
    """Potong sinyal berdasarkan detik."""
    s = int(start * SR)
    e = int(end * SR) if end else len(y_full)
    e = min(e, len(y_full))
    return y_full[s:e]


def _extract_mfcc(y_seg: np.ndarray) -> dict:
    """Hitung MFCC → mean & std (26 fitur)."""
    mfcc = librosa.feature.mfcc(y=y_seg, sr=SR, n_mfcc=N_MFCC)
    return {
        **{f"mfcc_mean_{i}": v for i, v in enumerate(np.mean(mfcc, axis=1))},
        **{f"mfcc_std_{i}":  v for i, v in enumerate(np.std(mfcc,  axis=1))},
    }


def _extract_chroma(y_seg: np.ndarray) -> dict:
    """Hitung Chroma STFT → mean & std (24 fitur)."""
    chroma = librosa.feature.chroma_stft(y=y_seg, sr=SR, n_chroma=N_CHROMA)
    return {
        **{f"chroma_mean_{i}": v for i, v in enumerate(np.mean(chroma, axis=1))},
        **{f"chroma_std_{i}":  v for i, v in enumerate(np.std(chroma,  axis=1))},
    }


def _parse_end_time(val) -> float:
    """Konversi end_time; kembalikan 0.0 jika kosong/NaN."""
    if pd.notna(val) and str(val).strip() != "":
        return float(val)
    return 0.0


def _process_row(row, instrument: str, cache: dict):
    """Proses 1 baris CSV → (y_seg, meta) atau None jika skip."""
    track_id = _build_track_id(row["title"], row["artist"])
    y_full   = _load_and_cache(track_id, instrument, cache)
    if y_full is None:
        return None

    end = _parse_end_time(row["end_time"])
    y_seg = _slice_audio(y_full, float(row["start_time"]), end)

    if len(y_seg) < MIN_SAMPLES:
        return None

    meta = {c: row[c] for c in META_COLS}
    meta["end_time"] = end if end > 0 else len(y_full) / SR
    return y_seg, meta


# ─────────────────── CORE EXTRACTION FUNCTIONS ────────────────────────────
def extract_mfcc_only(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Ekstrak fitur MFCC saja (7 meta + 26 MFCC = 33 kolom)."""
    cache: dict[str, np.ndarray] = {}
    rows, skipped = [], 0

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  [MFCC|{instrument}]"):
        result = _process_row(row, instrument, cache)
        if result is None:
            skipped += 1
            continue
        y_seg, meta = result
        rows.append({**meta, **_extract_mfcc(y_seg)})

    if skipped:
        print(f"    ⚠  {skipped} segmen di‑skip")
    return pd.DataFrame(rows)


def extract_chroma_only(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Ekstrak fitur Chromagram saja (7 meta + 24 chroma = 31 kolom)."""
    cache: dict[str, np.ndarray] = {}
    rows, skipped = [], 0

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  [Chroma|{instrument}]"):
        result = _process_row(row, instrument, cache)
        if result is None:
            skipped += 1
            continue
        y_seg, meta = result
        rows.append({**meta, **_extract_chroma(y_seg)})

    if skipped:
        print(f"    ⚠  {skipped} segmen di‑skip")
    return pd.DataFrame(rows)


def extract_mfcc_chroma(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Ekstrak MFCC + Chromagram gabungan (7 meta + 50 fitur = 57 kolom)."""
    cache: dict[str, np.ndarray] = {}
    rows, skipped = [], 0

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  [MFCC+Chroma|{instrument}]"):
        result = _process_row(row, instrument, cache)
        if result is None:
            skipped += 1
            continue
        y_seg, meta = result
        rows.append({**meta, **_extract_mfcc(y_seg), **_extract_chroma(y_seg)})

    if skipped:
        print(f"    ⚠  {skipped} segmen di‑skip")
    return pd.DataFrame(rows)


# ─────────────────────── SAVE HELPERS ─────────────────────────────────────
def _save(df: pd.DataFrame, name: str, out_dir: str) -> None:
    """Simpan DataFrame ke .pkl dan .csv."""
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, f"{name}.pkl")
    csv_path = os.path.join(out_dir, f"{name}.csv")

    with open(pkl_path, "wb") as f:
        pickle.dump(df, f)
    df.to_csv(csv_path, index=False)

    print(f"    ✅ {name}  →  shape {df.shape}")


def _get_instrument_folders() -> list[str]:
    """Auto‑detect semua subfolder di input/."""
    folders = sorted([
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
        and not d.startswith(".")
    ])
    return folders


# ─────────────────────── ENTRY POINT ──────────────────────────────────────
def main() -> None:
    instruments = _get_instrument_folders()

    print("=" * 60)
    print("  Feature Extraction Pipeline (Multi‑Instrument)")
    print(f"  CSV        : {CSV_PATH}")
    print(f"  Input dir  : {INPUT_DIR}")
    print(f"  Instruments: {', '.join(instruments)}")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    print(f"\n📄 Loaded {len(df)} baris dari CSV\n")

    for inst in instruments:
        inst_out = os.path.join(OUTPUT_DIR, inst)
        print(f"\n{'─' * 50}")
        print(f"🎸 Instrument: {inst}")
        print(f"   Audio dir : input/{inst}/")
        print(f"   Output dir: output/{inst}/")
        print(f"{'─' * 50}")

        # 1 ─ MFCC only
        df_mfcc = extract_mfcc_only(df, inst)
        _save(df_mfcc, "extracted_features_mfcc", inst_out)

        # 2 ─ Chromagram only
        df_chroma = extract_chroma_only(df, inst)
        _save(df_chroma, "extracted_features_chromagram", inst_out)

        # 3 ─ MFCC + Chromagram combined
        df_combined = extract_mfcc_chroma(df, inst)
        _save(df_combined, "extracted_features_mfcc_chroma", inst_out)

    print(f"\n🎉 Selesai! Semua fitur untuk {len(instruments)} instrumen berhasil diekstrak.\n")


if __name__ == "__main__":
    main()
