"""Download and extract Backblaze Drive Stats archives.

Archives are published at
https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data as
``data_Q<n>_<year>.zip`` (2016 onward) or ``data_<year>.zip`` (2013-2015). Each
archive holds one CSV per day named ``YYYY-MM-DD.csv``.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)


def archive_url(base_url: str, period: str) -> str:
    """URL of the archive for a period such as ``Q1_2026`` or ``2015``."""
    return f"{base_url.rstrip('/')}/data_{period}.zip"


def download_archive(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("Already downloaded: %s", dest)
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def extract_daily_csvs(archive: Path, raw_dir: Path) -> list[Path]:
    """Extract the daily CSVs into a flat directory, skipping OS metadata entries."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = Path(info.filename)
            if info.is_dir() or name.suffix != ".csv" or "__MACOSX" in name.parts:
                continue
            target = raw_dir / name.name
            with zf.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            written.append(target)
    return written


def download(cfg: dict) -> list[Path]:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    zip_dir = raw_dir.parent / "archives"
    files: list[Path] = []
    for period in cfg["download"]["periods"]:
        url = archive_url(cfg["download"]["base_url"], period)
        log.info("Downloading %s", url)
        archive = download_archive(url, zip_dir / f"data_{period}.zip")
        files += extract_daily_csvs(archive, raw_dir)
    return files
