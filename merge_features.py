import os
import pandas as pd
from tqdm import tqdm

# ───────────────────────────── KONFIGURASI ────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "output")
FINAL_DIR  = os.path.join(BASE_DIR, "final")

def merge_features():
    print("=" * 60)
    print("  Merging Extracted Features into Global Dataset")
    print(f"  Source dir: {SOURCE_DIR}")
    print(f"  Target dir: {FINAL_DIR}")
    print("=" * 60)

    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Folder '{SOURCE_DIR}' tidak ditemukan.")
        return

    # Kolektor data: {(instrument, feature_name): [list_of_dataframes]}
    collections = {}

    # 1. Scanning folder output
    print("\n🔍 Scanning files...")
    all_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith(".csv") and file.startswith("extracted_features_"):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("⚠ Tidak ditemukan file fitur untuk digabungkan.")
        return

    print(f"📦 Ditemukan {len(all_files)} file CSV.")

    # 2. Reading and grouping
    for file_path in tqdm(all_files, desc="📖 Reading CSVs"):
        # Path structure: output/{artist}/{track_id}/{instrument}/{feature_name}.csv
        parts = file_path.replace(SOURCE_DIR + os.sep, "").split(os.sep)
        
        if len(parts) < 4:
            continue
            
        instrument = parts[2]
        feature_name = os.path.splitext(parts[3])[0] # e.g. extracted_features_mfcc
        
        # Ambil nama fitur pendeknya saja (misal: mfcc)
        short_feature = feature_name.replace("extracted_features_", "")
        
        key = (instrument, short_feature)
        if key not in collections:
            collections[key] = []
            
        try:
            df = pd.read_csv(file_path)
            
            # Standarisasi nama kolom ke lowercase agar mudah dideteksi
            df.columns = [c.lower() for c in df.columns]
            
            # Buat kolom track_id: artist_slug-title_slug
            if "artist" in df.columns and "title" in df.columns:
                def clean_track_id(row):
                    art = str(row["artist"]).lower().strip()
                    art_slug = "".join(filter(str.isalnum, art)) # Hapus simbol & spasi
                    
                    tit = str(row["title"]).lower().strip()
                    tit_slug = tit.replace(" ", "_") # Spasi jadi underscore
                    return f"{art_slug}-{tit_slug}"

                df["track_id"] = df.apply(clean_track_id, axis=1)
                
                # Hapus kolom artist dan title yang lama
                df = df.drop(columns=["artist", "title"])
                
                # Pindahkan track_id ke kolom pertama
                cols = ["track_id"] + [c for c in df.columns if c != "track_id"]
                df = df[cols]

            collections[key].append(df)
        except Exception as e:
            print(f"    ⚠ Gagal membaca {file_path}: {e}")

    # 3. Concatenating and Saving
    print("\n💾 Saving merged files...")
    for (inst, feat), df_list in collections.items():
        if not df_list:
            continue
            
        # Gabungkan semua baris
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # Tentukan folder output: final/{instrument}/{feature}/
        out_dir = os.path.join(FINAL_DIR, inst, feat)
        os.makedirs(out_dir, exist_ok=True)
        
        out_file_csv = os.path.join(out_dir, f"extracted_features_{feat}.csv")
        out_file_pkl = os.path.join(out_dir, f"extracted_features_{feat}.pkl")
        
        # Simpan CSV dan Pickle (Pickle lebih cepat untuk loading nantinya)
        merged_df.to_csv(out_file_csv, index=False)
        merged_df.to_pickle(out_file_pkl)
        
        print(f"    ✅ [{inst} | {feat}] -> {merged_df.shape[0]} rows saved to {out_dir}")

    print("\n🎉 Semua fitur berhasil digabungkan ke folder 'final/'!\n")

if __name__ == "__main__":
    merge_features()
