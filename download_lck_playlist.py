"""
LCK Summer Playlist Downloader
==============================
Automatizált YouTube lejátszási lista letöltő szkript League of Legends esport mérkőzésekhez.
- 1080p MP4 formátum (a YOLOv8 és OCR pipeline-hoz optimalizálva)
- Megszakításbiztos: '.downloaded_archive.txt' nyilvántartja a már letöltött videókat, így hiba vagy leállítás esetén onnan folytatja, ahol abbahagyta.
- Opcionálisan levágja a kezdeti Draft fázist (pl. az első 120 másodpercet).
"""

import os
import sys
import argparse
import yt_dlp
from yt_dlp.utils import download_range_func

# ==============================================================================
# ALAPÉRTELMEZETT BEÁLLÍTÁSOK
# ==============================================================================
DEFAULT_OUTPUT_DIR = "LCK Summer"
# Ide másolhatod be a lejátszási lista linkjét, ha nem parancssori argumentumként adod meg:
DEFAULT_PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLXCkXhBN4Qdo"

# Ha szeretnéd levágni a meccs előtti draft fázist, állítsd be a másodpercek számát (pl. 120 mp = 2 perc).
# Ha a teljes videót le szeretnéd tölteni az elejétől, állítsd 0-ra!
SKIP_DRAFT_SECONDS = 0  # Ajánlott: 0 (a teljes videó letöltése a legbiztonságosabb a meccskezdet elcsúszása ellen)


def download_playlist(playlist_url: str, output_dir: str = DEFAULT_OUTPUT_DIR, skip_seconds: int = 0, max_resolution: int = 1080):
    if not playlist_url:
        print("[Hiba] Nem adtál meg YouTube lejátszási lista linket!")
        print("Használat:")
        print("  python download_lck_playlist.py \"https://www.youtube.com/playlist?list=...\"")
        print("Vagy írd be a szkript DEFAULT_PLAYLIST_URL változójába.")
        return

    os.makedirs(output_dir, exist_ok=True)
    archive_file = os.path.join(output_dir, ".downloaded_archive.txt")

    print("=" * 70)
    print(" LCK SUMMER YOUTUBE PLAYLIST LETÖLTŐ")
    print("=" * 70)
    print(f" Célkönyvtár       : {os.path.abspath(output_dir)}")
    print(f" Lejátszási lista  : {playlist_url}")
    print(f" Maximális felbontás: {max_resolution}p (1080p a számítógépes látáshoz)")
    print(f" Draft levágása    : {f'Első {skip_seconds} mp átugrása' if skip_seconds > 0 else 'Teljes videó letöltése (nincs vágás)'}")
    print(f" Folytatási napló  : {archive_file}")
    print("=" * 70)

    # yt-dlp konfiguráció
    ydl_opts = {
        # 1080p MP4 + m4a audio, FFmpeg-gel egyesítve
        'format': f'bestvideo[height<={max_resolution}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={max_resolution}]+bestaudio/best[height<={max_resolution}]/best',
        'merge_output_format': 'mp4',
        # Sorszámozott, tiszta fájlnév (pl. 01 - T1 vs GenG Game 1.mp4)
        'outtmpl': os.path.join(output_dir, '%(playlist_index)02d - %(title)s.%(ext)s'),
        # Archívum: a sikeresen befejezett videók azonosítóit eltárolja, így újraindításkor nem tölti le őket újra
        'download_archive': archive_file,
        # Hálózati stabilitás és újrapróbálkozás
        'retries': 10,
        'fragment_retries': 10,
        'file_access_retries': 5,
        'continuedl': True,
        'ignoreerrors': True,  # Ha egy privát vagy törölt videó van a listában, átugorja és megy tovább
        'no_warnings': False,
    }

    # Ha a draft levágása be van kapcsolva
    if skip_seconds > 0:
        print(f"[Info] Idősáv vágás bekapcsolva: {skip_seconds}s-tól a videó végéig.")
        ydl_opts['download_ranges'] = download_range_func(None, [(skip_seconds, None)])
        ydl_opts['force_keyframes_at_cuts'] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([playlist_url])
            print("\n[Kész] A lejátszási lista feldolgozása befejeződött!")
        except Exception as e:
            print(f"\n[Hiba történt a letöltés során]: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCK Summer YouTube lejátszási lista letöltő")
    parser.add_argument("url", nargs="?", default=DEFAULT_PLAYLIST_URL, help="A YouTube lejátszási lista URL-je")
    parser.add_argument("--dir", default=DEFAULT_OUTPUT_DIR, help="Célkönyvtár (alapértelmezett: 'LCK Summer')")
    parser.add_argument("--skip-draft", type=int, default=SKIP_DRAFT_SECONDS, help="Draft időtartam levágása másodpercben (pl. 120)")
    parser.add_argument("--resolution", type=int, default=1080, help="Maximális felbontás (alapértelmezett: 1080)")

    args = parser.parse_args()
    download_playlist(
        playlist_url=args.url,
        output_dir=args.dir,
        skip_seconds=args.skip_draft,
        max_resolution=args.resolution
    )
