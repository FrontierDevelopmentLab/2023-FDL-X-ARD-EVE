"""Unit tests for core.data_access cache I/O."""

from pathlib import Path

import pandas as pd

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
