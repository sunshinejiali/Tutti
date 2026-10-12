"""Human-readable connector capacity is normalized without CUDA imports."""

import pytest

from tutti.common.utils import parse_capacity


@pytest.mark.parametrize("value, expected", [
    ("64 GiB", 64 * 1024**3),
    ("1 TB", 1000**4),
    ("1.5GB", 1500000000),
    (" 0.5 gib ", 512 * 1024**2),
    ("8 MiB", 8 * 1024**2),
    ("1 MB", 1000000),
    ("4096 B", 4096),
    ("0.001 KB", 1),
    ("12345678901234567890 B", 12345678901234567890),
])
def test_capacity_units(value, expected):
    assert parse_capacity(value) == expected


@pytest.mark.parametrize("value", [None, True, 64, 1.5, "64", "0 GB", "-1 GB",
                                   "1 XB", "nan GB", "1e3 GB", "0.1 B", ""])
def test_capacity_rejects_ambiguous_or_invalid_units(value):
    with pytest.raises(ValueError, match="capacity"):
        parse_capacity(value)


def test_capacity_parser_has_no_cuda_imports():
    import os
    from pathlib import Path
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    code = ("import sys; from tutti.common.utils import parse_capacity; "
            "assert parse_capacity('64 GiB') == 64 * 1024**3; "
            "assert not any(k in sys.modules for k in ('torch', 'tutti_runtime', 'tutti_kv_transfer'))")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
