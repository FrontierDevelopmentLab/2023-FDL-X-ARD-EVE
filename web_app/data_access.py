"""Data access layer for reading AIA images from Zarr stores (local filesystem or S3)."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

AIA_WAVELENGTHS = ["131A", "1600A", "1700A", "171A", "193A", "211A", "304A", "335A", "94A"]

# ── Backend configuration ───────────────────────────────────────────────────
# Set DATA_BACKEND=s3 to fall back to reading from AWS S3.
# Default: local filesystem at LOCAL_DATA_ROOT.

DATA_BACKEND = os.environ.get("DATA_BACKEND", "local")
LOCAL_DATA_ROOT = Path(os.environ.get("LOCAL_DATA_ROOT", "/data"))

S3_BUCKET = os.environ.get("S3_BUCKET", "nasa-radiant-data")
AIA_ZARR_PREFIX = os.environ.get("AIA_ZARR_PREFIX", "helioai-datasets/us-fdlx-ard/sdomlv2a/AIA.zarr")
HMI_ZARR_PREFIX = os.environ.get("HMI_ZARR_PREFIX", "helioai-datasets/us-fdlx-ard/sdomlv2a/HMI.zarr")

CACHE_DIR = Path(__file__).parent / "cache"
INDEX_CACHE = CACHE_DIR / "aia_time_index.csv"

# ── S3 helpers (only used when DATA_BACKEND=s3) ────────────────────────────

_fs = None


def get_s3fs():
    global _fs
    if _fs is None:
        import s3fs
        _fs = s3fs.S3FileSystem(anon=True)
    return _fs


# ── Zarr root accessors ────────────────────────────────────────────────────

def get_aia_root():
    """Open AIA Zarr group (local or S3)."""
    if DATA_BACKEND == "s3":
        import s3fs
        fs = get_s3fs()
        store = s3fs.S3Map(root=f"{S3_BUCKET}/{AIA_ZARR_PREFIX}", s3=fs)
        return zarr.open(store, mode="r")
    else:
        return zarr.open(str(LOCAL_DATA_ROOT / "AIA.zarr"), mode="r")


# ── Time index construction ────────────────────────────────────────────────

def _read_t_obs_local(year: str, wl: str) -> tuple[str, str, list[str]]:
    """Read T_OBS attr from local .zattrs file."""
    path = LOCAL_DATA_ROOT / "AIA.zarr" / year / wl / ".zattrs"
    attrs = json.loads(path.read_text())
    return year, wl, attrs["T_OBS"]


def _read_t_obs_s3(year: str, wl: str) -> tuple[str, str, list[str]]:
    """Read T_OBS attr from S3 .zattrs file (with retries)."""
    import s3fs
    import time as _time
    fs = s3fs.S3FileSystem(anon=True)
    path = f"{S3_BUCKET}/{AIA_ZARR_PREFIX}/{year}/{wl}/.zattrs"
    for attempt in range(3):
        try:
            raw = fs.cat(path)
            attrs = json.loads(raw)
            return year, wl, attrs["T_OBS"]
        except Exception:
            if attempt == 2:
                raise
            _time.sleep(2 ** attempt)


def _discover_years() -> list[str]:
    """Return sorted list of year directories in the AIA Zarr store."""
    if DATA_BACKEND == "s3":
        fs = get_s3fs()
        entries = fs.ls(f"{S3_BUCKET}/{AIA_ZARR_PREFIX}/")
        return sorted(e.split("/")[-1] for e in entries if e.split("/")[-1].isdigit())
    else:
        aia_zarr = LOCAL_DATA_ROOT / "AIA.zarr"
        return sorted(
            d.name for d in aia_zarr.iterdir()
            if d.is_dir() and d.name.isdigit()
        )


def build_time_index(aia_root, progress_callback=None) -> pd.DataFrame:
    """Build a mapping from timestamp -> (year, index) for each AIA wavelength.

    Uses T_OBS attrs from the Zarr store. Caches result to CSV.
    First run reads metadata (parallel), subsequent runs use cache.
    """
    if INDEX_CACHE.exists():
        df = pd.read_csv(INDEX_CACHE, parse_dates=["Time"])
        df.set_index("Time", inplace=True)
        return df

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    years = _discover_years()
    read_fn = _read_t_obs_s3 if DATA_BACKEND == "s3" else _read_t_obs_local

    jobs = [(year, wl) for wl in AIA_WAVELENGTHS for year in years]
    total = len(jobs)
    results = {}  # (year, wl) -> list[str]

    max_workers = 8 if DATA_BACKEND == "s3" else 16
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(read_fn, year, wl): (year, wl) for year, wl in jobs}
        done = 0
        for future in as_completed(futures):
            year, wl, t_obs = future.result()
            results[(year, wl)] = t_obs
            done += 1
            if progress_callback:
                progress_callback(done, total, f"{year}/{wl} ({len(t_obs)} timestamps)")
            else:
                print(f"  [{done}/{total}] {year}/{wl} ({len(t_obs)} timestamps)")

    # Build aligned index per wavelength, then join
    join_series = None
    for wl in AIA_WAVELENGTHS:
        frames = []
        for year in years:
            t_obs = results[(year, wl)]
            df_wl = pd.DataFrame({
                "Time": pd.to_datetime(t_obs, format="mixed", utc=True),
                f"idx_{wl}": np.arange(len(t_obs)),
                "year": int(year),
            })
            frames.append(df_wl)

        df_wl = pd.concat(frames, ignore_index=True)
        df_wl["Time"] = df_wl["Time"].dt.tz_localize(None)
        df_wl["Time"] = df_wl["Time"].dt.round("36min")
        df_wl = df_wl.drop_duplicates(subset="Time", keep="first")
        df_wl.set_index("Time", inplace=True)

        if join_series is None:
            join_series = df_wl
        else:
            join_series = join_series.join(df_wl[[f"idx_{wl}"]], how="inner")

    join_series.sort_index(inplace=True)
    join_series.to_csv(INDEX_CACHE)
    return join_series


# ── Query helpers ───────────────────────────────────────────────────────────

def get_available_dates(time_index: pd.DataFrame):
    """Return the min and max dates available in the index."""
    return time_index.index.min(), time_index.index.max()


def get_timestamps_in_range(time_index: pd.DataFrame, start, end) -> pd.DataFrame:
    """Return index rows within a date range."""
    mask = (time_index.index >= pd.to_datetime(start)) & (time_index.index <= pd.to_datetime(end))
    return time_index[mask]


def get_aia_image(aia_root, time_index: pd.DataFrame, timestamp) -> dict | None:
    """Load AIA images for all wavelengths at a given timestamp.

    Returns dict mapping wavelength -> 512x512 numpy array, or None if not found.
    """
    timestamp = pd.to_datetime(timestamp)
    rounded = timestamp.round("36min")

    idx_loc = time_index.index.get_indexer([rounded], method="nearest")
    if idx_loc[0] == -1:
        return None

    row = time_index.iloc[idx_loc[0]]
    year = str(int(row["year"]))

    aia_image = {}
    for wl in AIA_WAVELENGTHS:
        idx = int(row[f"idx_{wl}"])
        aia_image[wl] = aia_root[year][wl][idx, :, :]

    return aia_image
