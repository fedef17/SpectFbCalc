#!/usr/bin/env python
"""
Download datasets (Huang kernels and spectral TOA kernels).

Usage:
    python download_data.py

Recommended: run from a terminal in the background 
    nohup python download_data.py > download_data.log 2>&1 &

Notes:
    - Re-running the script after an interruption skips files that are
      already present with a matching checksum (automatic resume).
    - To add a new Zenodo dataset later, just add a
      new entry to the DOWNLOADS list.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Zenodo datasets: filtered, per-file download with checksum verification.
# Each entry: (record_id, output_dir, exclude_substrings)
# - record_id: numeric Zenodo record ID (last part of the DOI, e.g.
#   "21245639" from "10.5281/zenodo.21245639")
# - output_dir: local directory the files are downloaded into
# - exclude_substrings: files whose name contains any of these are skipped
#   (pass an empty list to download everything)
ZENODO_DOWNLOADS = [
    ("21245639", "../kernels/spectral/toa_clear", []),
    ("21246421", "../kernels/spectral/toa_all", [])
    # ("<record_id>", "../data/sample", []), 
]

# Mendeley Data datasets: no per-file API, download the "Download All" zip
# and unpack it. Each entry: (zip_url, output_dir, strip_top_level_dir)
# - zip_url: the "Download All" link shown on the dataset page
# - output_dir: local directory the unpacked files land in
# - strip_top_level_dir: if the zip wraps everything in one folder,
# set True to flatten it away; False to keep it as-is
MENDELEY_DOWNLOADS = [
    (
        "https://data.mendeley.com/public-api/zip/3drx8fmmz9/download/1",
        "../kernels/huang",
        True,
    ),
    # ("<zip_url>", "../data/other", True), 
]

WGET_BASE = ["wget", "-4", "-q"] 


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check_connectivity() -> None:
    """Check Zenodo is reachable before starting, with a clear error message
    instead of a cryptic timeout."""
    try:
        subprocess.run(
            WGET_BASE + ["--spider", "--timeout=15", "https://zenodo.org"],
            check=True,
        )
    except subprocess.CalledProcessError:
        sys.exit(
            "\nERROR: could not reach https://zenodo.org.\n"
        )


def _md5(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_ok(path: str, checksum: str | None) -> bool:
    """Expected checksum in Zenodo's format: 'md5:<hex>'. If missing, assume ok."""
    if not checksum or not checksum.startswith("md5:"):
        return True
    return _md5(path) == checksum.split("md5:", 1)[1]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_from_zenodo(record_id: str, output_dir: str, exclude: list[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta_file = f"record_{record_id}.json"

    print(f"Fetching metadata for record {record_id}...")
    subprocess.run(
        WGET_BASE + ["-O", meta_file, f"https://zenodo.org/api/records/{record_id}"],
        check=True,
    )

    with open(meta_file) as f:
        record = json.load(f)
    os.remove(meta_file)

    files = [f for f in record["files"] if not any(x in f["key"] for x in exclude)]
    print(f"Found {len(files)} relevant files. Downloading to {output_dir}...")

    for f in files:
        name = f["key"]
        url = f["links"]["self"]
        checksum = f.get("checksum")
        dest = os.path.join(output_dir, name)

        if os.path.exists(dest) and _hash_ok(dest, checksum):
            print(f"  [skip] {name} (already present, hash ok)")
            continue

        print(f"  [get ] {name}")
        subprocess.run(
            WGET_BASE + ["--show-progress", "-O", dest, url],
            check=True,
        )

        if not _hash_ok(dest, checksum):
            print(f"  [WARN] hash mismatch for {name} — file may be corrupted, retry later")

    print(f"Done: {output_dir}\n")


def download_mendeley_zip(zip_url: str, output_dir: str, strip_top_level: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)
    marker = os.path.join(output_dir, ".download_complete")

    if os.path.exists(marker):
        print(f"Skipping {output_dir} (already downloaded — delete {marker} to force a re-download)\n")
        return

    zip_path = os.path.join(output_dir, "_download.zip")

    print(f"Downloading {zip_url} ...")
    subprocess.run(
        WGET_BASE + ["--show-progress", "-O", zip_path, zip_url],
        check=True,
    )

    print(f"Extracting to {output_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            rel = member.filename
            if strip_top_level and "/" in rel:
                rel = rel.split("/", 1)[1]
            if not rel:
                continue
            dest = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    os.remove(zip_path)
    open(marker, "w").close()
    print(f"Done: {output_dir}\n")


if __name__ == "__main__":
    check_connectivity()
    for record_id, output_dir, exclude in ZENODO_DOWNLOADS:
        download_from_zenodo(record_id, output_dir, exclude)
    for zip_url, output_dir, strip_top_level in MENDELEY_DOWNLOADS:
        download_mendeley_zip(zip_url, output_dir, strip_top_level)
    print("All downloads completed.")