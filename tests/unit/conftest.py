"""Pytest fixtures shared across unit tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_registry_isolation() -> None:
    """No-op placeholder. Registry mutations happen at import time and are
    intentionally process-global; tests that mutate the registry are
    responsible for cleaning up via ``unregister``.
    """
    yield
