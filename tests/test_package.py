from __future__ import annotations

import importlib.metadata

import interest_rate_derivatives as m


def test_version() -> None:
    assert importlib.metadata.version("interest_rate_derivatives") == m.__version__
