"""Shared fixtures for API tests."""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def _check_data_backend():
    """Sanity check: tests that hit the lifespan need a configured backend.

    Tests that mock the lifespan or exercise pure logic don't need this; they
    should still run. We only warn — we don't fail — so unit tests can run
    on a dev machine without data access.
    """
    backend = os.environ.get("DATA_BACKEND", "local")
    if backend == "local":
        root = os.environ.get("LOCAL_DATA_ROOT", "")
        if not root or not os.path.isdir(root):
            print(
                f"\n[pytest] DATA_BACKEND=local but LOCAL_DATA_ROOT={root!r} "
                "is not a directory. Tests that exercise the FastAPI "
                "lifespan will fail. Set DATA_BACKEND=s3 or point "
                "LOCAL_DATA_ROOT at a real Zarr root to enable them."
            )
