"""
extract_features.py
====================
Pipeline ekstraksi fitur audio (MFCC & Chromagram) dari struktur folder:
  input/{instrument}/{artist}/{artist_name}-{song_title}_{instrument}.wav

Output di‑kelompokkan per lagu:
  output/{artist}/{track_id}/{instrument}/extracted_features_mfcc.pkl
  output/{artist}/{track_id}/{instrument}/extracted_features_chromagram.pkl
  output/{artist}/{track_id}/{instrument}/extracted_features_mfcc_chroma.pkl

Subfolder instrumen yang di‑scan otomatis:
  input/vocals, input/guitar, input/piano, input/bass, dll.

Cara pakai:
  pip install librosa pandas numpy tqdm
  python extract_features.py
"""

import os
import re
import pickle
import shutil
import warnings

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────────── KONFIGURASI ────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "input/base_dataset.csv")
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
    # Bersihkan artis: hapus semua simbol dan spasi (misal: ST 12 -> st12, D'Masiv -> dmasiv)
    artist_slug = re.sub(r"[^a-zA-Z0-9]+", "", artist.lower())
    # Bersihkan judul: spasi jadi underscore
    title_slug  = re.sub(r"\s+", "_", title.strip().lower())
    return f"{artist_slug}-{title_slug}"


def _wav_path(artist_raw: str, track_id: str, instrument: str) -> str:
    """artist + track_id + instrument → full path WAV atau MP3."""
    # 1. Coba folder asli dari CSV
    # 2. Coba folder lowercase (misal: ST 12 -> st 12)
    # 3. Coba folder lowercase tanpa spasi (misal: ST 12 -> st12)
    
    artist_variations = [
        artist_raw,
        artist_raw.lower(),
        artist_raw.lower().replace(" ", ""),
        artist_raw.lower().replace("'", ""),
        artist_raw.lower().replace(" ", "").replace("'", "")
    ]
    
    # Hapus duplikat sambil menjaga urutan
    seen = set()
    artist_variations = [v for v in artist_variations if not (v in seen or seen.add(v))]

    for art_folder in artist_variations:
        base_folder = os.path.join(INPUT_DIR, instrument, art_folder)
        if not os.path.isdir(base_folder):
            continue

        # Tentukan pola nama file yang akan dicoba
        filename_patterns = [f"{track_id}_{instrument}"]
        if instrument == "raw_audio":
            filename_patterns.append(track_id) # Coba tanpa suffix _raw_audio

        for pattern in filename_patterns:
            for ext in [".wav", ".mp3"]:
                path = os.path.join(base_folder, f"{pattern}{ext}")
                if os.path.isfile(path):
                    return path
    return None


def _load_and_cache(artist_raw: str, track_id: str, instrument: str,
                    cache: dict) -> np.ndarray | None:
    """Load audio sekali per lagu, simpan ke dict cache."""
    key = f"{track_id}_{instrument}"
    if key in cache:
        return cache[key]
    
    path = _wav_path(artist_raw, track_id, instrument)
    if path is None:
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
    """Hitung Chroma CQT (Constant-Q Transform) → mean & std (24 fitur)."""
    # CQT lebih akurat secara musikal dibanding STFT
    chroma = librosa.feature.chroma_cqt(y=y_seg, sr=SR, n_chroma=N_CHROMA)
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
    y_full   = _load_and_cache(row["artist"], track_id, instrument, cache)
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
def _save(df: pd.DataFrame, feature_method: str, instrument: str) -> None:
    """
    Simpan DataFrame ke struktur nested:
    output/artist/artist_name-song_title/instrument/extracted_features_extracted_methods.pkl
    """
    if df.empty:
        return

    # Group by artist & title untuk memisahkan per lagu
    grouped = df.groupby(["artist", "title"])
    
    for (artist_raw, title), group_df in grouped:
        track_id = _build_track_id(title, artist_raw)
        # Gunakan pembersihan yang sama dengan track_id agar folder artist rapi (st12, dmasiv)
        artist_folder = re.sub(r"[^a-zA-Z0-9]+", "", artist_raw.lower())
        
        # Bangun path: output/artist/track_id/instrument/
        song_out_dir = os.path.join(OUTPUT_DIR, artist_folder, track_id, instrument)
        os.makedirs(song_out_dir, exist_ok=True)
        
        pkl_path = os.path.join(song_out_dir, f"{feature_method}.pkl")
        csv_path = os.path.join(song_out_dir, f"{feature_method}.csv")

        with open(pkl_path, "wb") as f:
            pickle.dump(group_df, f)
        group_df.to_csv(csv_path, index=False)

        # Copy audio source jika belum ada di output (agar satu paket)
        src_path = _wav_path(artist_raw, track_id, instrument)
        if src_path:
            dst_path = os.path.join(song_out_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

    print(f"    ✅ {feature_method} saved for {len(grouped)} songs.")


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
    # instruments = ["raw_audio"]

    print("=" * 60)
    print("  Feature Extraction Pipeline (Multi‑Instrument)")
    print(f"  CSV        : {CSV_PATH}")
    print(f"  Input dir  : {INPUT_DIR}")
    print(f"  Instruments: {', '.join(instruments)}")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    print(f"\n📄 Loaded {len(df)} baris dari CSV\n")

    for inst in instruments:
        print(f"\n{'─' * 50}")
        print(f"🎸 Instrument: {inst}")
        print(f"   Audio path: input/{inst}/" + "{artist}/{track_id}_" + f"{inst}.wav")
        print(f"   Output dir: output/" + "{artist_lower}/{track_id}/" + f"{inst}/")
        print(f"{'─' * 50}")

        # 1 ─ MFCC only
        df_mfcc = extract_mfcc_only(df, inst)
        _save(df_mfcc, "extracted_features_mfcc", inst)

        # 2 ─ Chromagram only
        df_chroma = extract_chroma_only(df, inst)
        _save(df_chroma, "extracted_features_chromagram", inst)

        # 3 ─ MFCC + Chromagram combined
        df_combined = extract_mfcc_chroma(df, inst)
        _save(df_combined, "extracted_features_mfcc_chroma", inst)

    print(f"\n🎉 Selesai! Semua fitur untuk {len(instruments)} instrumen berhasil diekstrak.\n")


if __name__ == "__main__":
    main()
