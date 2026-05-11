"""Unit tests for core.data_access cache I/O."""

import json
from pathlib import Path

import pandas as pd
import pytest

from core import data_access


def test_build_time_index_reads_parquet_cache(tmp_path, monkeypatch):
    """If the parquet cache exists, build_time_index returns it without rebuilding."""
    cache_path = tmp_path / "aia_time_index.parquet"
    monkeypatch.setattr(data_access, "INDEX_CACHE", cache_path)

    expected = pd.DataFrame(
        {
            "year": [2017, 2017],
            "idx_94A": [0, 1],
            "idx_131A": [0, 1],
            "idx_171A": [0, 1],
            "idx_193A": [0, 1],
            "idx_211A": [0, 1],
            "idx_304A": [0, 1],
            "idx_335A": [0, 1],
            "idx_1600A": [0, 1],
            "idx_1700A": [0, 1],
        },
        index=pd.DatetimeIndex(
            ["2017-09-06 00:00:00", "2017-09-06 00:36:00"], name="Time"
        ),
    )
    expected.to_parquet(cache_path)

    result = data_access.build_time_index(aia_root=None)

    pd.testing.assert_frame_equal(result, expected)


def test_index_cache_filename_is_parquet():
    """The cache constant should point at a .parquet file."""
    assert data_access.INDEX_CACHE.suffix == ".parquet"


import datetime as _dt


def _build_fake_index() -> pd.DataFrame:
    return pd.DataFrame(
        {"year": [2017] * 3, **{f"idx_{wl}": [0, 1, 2] for wl in data_access.AIA_WAVELENGTHS}},
        index=pd.DatetimeIndex(
            [
                "2017-09-06 00:00:00",
                "2017-09-06 00:36:00",
                "2017-09-06 01:12:00",
            ],
            name="Time",
        ),
    )


def test_find_nearest_indexed_timestamp_snaps_within_range():
    idx = _build_fake_index()
    snapped = data_access.find_nearest_indexed_timestamp(
        idx, _dt.datetime(2017, 9, 6, 0, 30, 0)
    )
    assert snapped == pd.Timestamp("2017-09-06 00:36:00")


def test_find_nearest_indexed_timestamp_returns_none_when_out_of_range():
    idx = _build_fake_index()
    snapped = data_access.find_nearest_indexed_timestamp(
        idx, _dt.datetime(2010, 1, 1, 0, 0, 0)
    )
    assert snapped is None


def test_find_nearest_indexed_timestamp_strips_timezone():
    idx = _build_fake_index()
    aware = _dt.datetime(2017, 9, 6, 0, 30, 0, tzinfo=_dt.timezone.utc)
    snapped = data_access.find_nearest_indexed_timestamp(idx, aware)
    assert snapped == pd.Timestamp("2017-09-06 00:36:00")


# ── build_time_index: the from-scratch metadata-scan path ───────────────────


def _make_fake_aia_store(root: Path, t_obs_by_year: dict[str, list[str]]) -> None:
    """Lay out a minimal local AIA.zarr tree with one .zattrs per year/wavelength."""
    for year, t_obs in t_obs_by_year.items():
        for wl in data_access.AIA_WAVELENGTHS:
            wl_dir = root / "AIA.zarr" / year / wl
            wl_dir.mkdir(parents=True, exist_ok=True)
            (wl_dir / ".zattrs").write_text(json.dumps({"T_OBS": t_obs}))


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    """Point data_access at a tmp store + tmp cache, local backend."""
    monkeypatch.setattr(data_access, "DATA_BACKEND", "local")
    monkeypatch.setattr(data_access, "LOCAL_DATA_ROOT", tmp_path)
    monkeypatch.setattr(data_access, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(data_access, "INDEX_CACHE", tmp_path / "cache" / "aia_time_index.parquet")
    return tmp_path


def test_build_time_index_from_scratch_writes_parquet_and_returns_frame(fake_store):
    _make_fake_aia_store(fake_store, {"2017": [
        "2017-09-06T00:00:01.0Z",
        "2017-09-06T00:36:01.0Z",
        "2017-09-06T01:12:01.0Z",
    ]})

    result = data_access.build_time_index(aia_root=None)

    assert list(result.index) == [
        pd.Timestamp("2017-09-06 00:00:00"),
        pd.Timestamp("2017-09-06 00:36:00"),
        pd.Timestamp("2017-09-06 01:12:00"),
    ]
    assert (result["year"] == 2017).all()
    for wl in data_access.AIA_WAVELENGTHS:
        assert list(result[f"idx_{wl}"]) == [0, 1, 2]
    # Cache was written and is reused on the next call.
    assert data_access.INDEX_CACHE.exists()
    pd.testing.assert_frame_equal(data_access.build_time_index(aia_root=None), result)


def test_build_time_index_coerces_bad_timestamps(fake_store):
    _make_fake_aia_store(fake_store, {"2017": [
        "2017-09-06T00:00:01.0Z",
        "not-a-real-timestamp",
        "2017-09-06T01:12:01.0Z",
    ]})

    result = data_access.build_time_index(aia_root=None)

    # The garbage entry is dropped; the surviving rows keep their original idx.
    assert list(result.index) == [
        pd.Timestamp("2017-09-06 00:00:00"),
        pd.Timestamp("2017-09-06 01:12:00"),
    ]
    for wl in data_access.AIA_WAVELENGTHS:
        assert list(result[f"idx_{wl}"]) == [0, 2]


def test_build_time_index_raises_when_no_year_dirs(fake_store):
    (fake_store / "AIA.zarr").mkdir()
    with pytest.raises(RuntimeError, match="No year directories"):
        data_access.build_time_index(aia_root=None)


def test_build_time_index_raises_on_empty_join(fake_store):
    # All entries unparseable -> every wavelength frame is empty -> empty index.
    _make_fake_aia_store(fake_store, {"2017": ["nope", "still-nope"]})
    with pytest.raises(RuntimeError, match="empty AIA time index"):
        data_access.build_time_index(aia_root=None)
