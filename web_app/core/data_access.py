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
LOCAL_DATA_ROOT = Path(os.environ.get(
    "LOCAL_DATA_ROOT",
    "/mnt/volume4/isidore-volume-4/projects/us-fdl-x/us-fdlx-ard-sdomlv2a",
))

S3_BUCKET = "nasa-radiant-data"
AIA_ZARR_PREFIX = "helioai-datasets/us-fdlx-ard/sdomlv2a/AIA.zarr"
HMI_ZARR_PREFIX = "helioai-datasets/us-fdlx-ard/sdomlv2a/HMI.zarr"

CACHE_DIR = Path(__file__).parent.parent / "cache"
INDEX_CACHE = CACHE_DIR / "aia_time_index.parquet"

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

    Uses T_OBS attrs from the Zarr store. Caches result to parquet; the first
    run reads metadata (parallel within each wavelength), subsequent runs use
    the cache.

    Processed one wavelength at a time: each wavelength's raw timestamp lists
    are parsed, folded into the running inner-join, and then dropped before the
    next wavelength is read. Peak memory therefore stays roughly proportional
    to a single wavelength's metadata rather than the whole store at once —
    which matters on small deployment hosts (this used to hold every
    wavelength's timestamps simultaneously and could OOM a ~3 GiB box).
    """
    if INDEX_CACHE.exists():
        return pd.read_parquet(INDEX_CACHE)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    years = _discover_years()
    if not years:
        raise RuntimeError(
            f"No year directories found in the AIA Zarr store "
            f"({'s3://' + S3_BUCKET + '/' + AIA_ZARR_PREFIX if DATA_BACKEND == 's3' else LOCAL_DATA_ROOT / 'AIA.zarr'})."
        )
    read_fn = _read_t_obs_s3 if DATA_BACKEND == "s3" else _read_t_obs_local
    max_workers = 8 if DATA_BACKEND == "s3" else 16
    total = len(AIA_WAVELENGTHS) * len(years)
    done = 0

    join_index = None  # running inner-join of the per-wavelength frames
    for wl in AIA_WAVELENGTHS:
        frames = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(read_fn, year, wl): year for year in years}
            for future in as_completed(futures):
                year, _wl, t_obs = future.result()
                frames.append(pd.DataFrame({
                    # T_OBS is ISO 8601 (e.g. "2014-09-22T13:18:14.84Z");
                    # parse with the fast ISO path, coercing the odd bad
                    # value to NaT rather than crashing the whole startup.
                    "Time": pd.to_datetime(t_obs, format="ISO8601", utc=True, errors="coerce"),
                    f"idx_{wl}": np.arange(len(t_obs)),
                    "year": int(year),
                }))
                done += 1
                msg = f"{year}/{wl} ({len(t_obs)} timestamps)"
                if progress_callback:
                    progress_callback(done, total, msg)
                else:
                    print(f"  [{done}/{total}] {msg}")

        df_wl = pd.concat(frames, ignore_index=True)
        del frames
        df_wl = df_wl.dropna(subset=["Time"]).copy()
        df_wl["Time"] = df_wl["Time"].dt.tz_localize(None).dt.round("36min")
        df_wl = df_wl.drop_duplicates(subset="Time", keep="first").set_index("Time")

        if join_index is None:
            join_index = df_wl
        else:
            # 'year' is wavelength-independent for a given timestamp, so only
            # carry the new index column in from each subsequent wavelength.
            join_index = join_index.join(df_wl[[f"idx_{wl}"]], how="inner")
        del df_wl

    if join_index is None or join_index.empty:
        raise RuntimeError(
            "Built an empty AIA time index — check that the Zarr .zattrs files "
            "contain ISO 8601 T_OBS values and that the wavelengths share timestamps."
        )

    join_index.sort_index(inplace=True)
    join_index.to_parquet(INDEX_CACHE)
    return join_index


# ── Query helpers ───────────────────────────────────────────────────────────

def get_available_dates(time_index: pd.DataFrame):
    """Return the min and max dates available in the index."""
    return time_index.index.min(), time_index.index.max()


def get_timestamps_in_range(time_index: pd.DataFrame, start, end) -> pd.DataFrame:
    """Return index rows within a date range."""
    mask = (time_index.index >= pd.to_datetime(start)) & (time_index.index <= pd.to_datetime(end))
    return time_index[mask]


def find_nearest_indexed_timestamp(
    time_index: pd.DataFrame, ts
) -> pd.Timestamp | None:
    """Round ts to 36 minutes and snap to the nearest indexed timestamp.

    Returns None if the snapped value would fall outside [index.min, index.max].
    """
    ts = pd.to_datetime(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    rounded = ts.round("36min")

    if rounded < time_index.index.min() or rounded > time_index.index.max():
        return None

    idx_loc = time_index.index.get_indexer([rounded], method="nearest")
    if idx_loc[0] == -1:
        return None
    return time_index.index[idx_loc[0]]


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
