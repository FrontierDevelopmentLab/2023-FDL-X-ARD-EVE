"""Data access layer for reading AIA images from S3 Zarr stores."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import zarr

AIA_WAVELENGTHS = ["131A", "1600A", "1700A", "171A", "193A", "211A", "304A", "335A", "94A"]

S3_BUCKET = "nasa-radiant-data"
AIA_ZARR_PREFIX = "helioai-datasets/us-fdlx-ard/sdomlv2a/AIA.zarr"
HMI_ZARR_PREFIX = "helioai-datasets/us-fdlx-ard/sdomlv2a/HMI.zarr"

CACHE_DIR = Path(__file__).parent / "cache"
INDEX_CACHE = CACHE_DIR / "aia_time_index.csv"

# Shared filesystem instance
_fs = None


def get_s3fs():
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(anon=True)
    return _fs


def get_aia_root():
    """Open AIA Zarr group from S3."""
    fs = get_s3fs()
    store = s3fs.S3Map(root=f"{S3_BUCKET}/{AIA_ZARR_PREFIX}", s3=fs)
    return zarr.open(store, mode="r")


def _read_t_obs(year: str, wl: str) -> tuple[str, str, list[str]]:
    """Read T_OBS attr directly via s3fs.cat — returns (year, wl, timestamps)."""
    # Use a fresh filesystem per thread to avoid shared session issues
    fs = s3fs.S3FileSystem(anon=True)
    path = f"{S3_BUCKET}/{AIA_ZARR_PREFIX}/{year}/{wl}/.zattrs"
    for attempt in range(3):
        try:
            raw = fs.cat(path)
            attrs = json.loads(raw)
            return year, wl, attrs["T_OBS"]
        except Exception as e:
            if attempt == 2:
                raise
            import time
            time.sleep(2 ** attempt)


def build_time_index(aia_root, progress_callback=None) -> pd.DataFrame:
    """Build a mapping from timestamp -> (year, index) for each AIA wavelength.

    Uses T_OBS attrs from the Zarr store. Caches result to CSV.
    First run reads from S3 (parallel, ~3-5 minutes), subsequent runs use cache.
    """
    if INDEX_CACHE.exists():
        df = pd.read_csv(INDEX_CACHE, parse_dates=["Time"])
        df.set_index("Time", inplace=True)
        return df

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Discover available years
    fs = get_s3fs()
    entries = fs.ls(f"{S3_BUCKET}/{AIA_ZARR_PREFIX}/")
    years = sorted(e.split("/")[-1] for e in entries if e.split("/")[-1].isdigit())

    # Read all T_OBS attrs in parallel (14 years x 9 wavelengths = 126 requests)
    jobs = [(year, wl) for wl in AIA_WAVELENGTHS for year in years]
    total = len(jobs)
    results = {}  # (year, wl) -> list[str]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_read_t_obs, year, wl): (year, wl) for year, wl in jobs}
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
