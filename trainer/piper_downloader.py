# trainer/piper_downloader.py
from __future__ import annotations
from pathlib import Path
import httpx

VOICES_INDEX_URL = "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json"
HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"


def fetch_voices_index(lang_filter: str | None = None) -> dict[str, dict]:
    resp = httpx.get(VOICES_INDEX_URL, timeout=15, follow_redirects=True)
    resp.raise_for_status()
    voices = resp.json()
    if lang_filter:
        voices = {
            k: v for k, v in voices.items()
            if v["language"]["code"].startswith(lang_filter)
        }
    return voices


def download_voice(file_paths: list[str], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for file_path in file_paths:
        filename = Path(file_path).name
        url = HF_BASE_URL + file_path
        dest = dest_dir / filename
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        downloaded.append(dest)
    return downloaded
