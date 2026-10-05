import pytest

from ai_asic.hardware.miner_profiles import (
    Protocol,
    all_profiles,
    default_profile,
    detect_profile,
)
from ai_asic.cgminer.client import _extract_type


@pytest.mark.parametrize("hint,model,chip", [
    ("Antminer S9", "Antminer S9", "BM1387"),
    ("antminer s9i", "Antminer S9i", "BM1387"),
    ("S9j", "Antminer S9j", "BM1387"),
    ("Antminer T9+", "Antminer T9+", "BM1387"),
    ("Antminer S19 Pro", "Antminer S19 Pro", "BM1398"),
    ("antminer s19j pro", "Antminer S19j Pro", "BM1398"),
    ("Antminer S19", "Antminer S19", "BM1398"),
    ("Antminer S17", "Antminer S17", "BM1397"),
    ("Antminer S3", "Antminer S3", "BM1382"),
    ("bm1387", "Antminer S9", "BM1387"),
    ("Some Antminer S21 Hyd", "Antminer S21", "BM1370"),
])
def test_detect_profile(hint, model, chip):
    p = detect_profile(hint)
    assert p is not None
    assert p.model == model
    assert p.chip == chip


def test_s19_not_s9():
    assert detect_profile("Antminer S19").model == "Antminer S19"


def test_unknown_and_empty():
    assert detect_profile("Whatsminer M30S") is None
    assert detect_profile("") is None


def test_default_is_s3():
    p = default_profile()
    assert p.model == "Antminer S3"
    assert p.protocol == Protocol.BM1382_USB
    assert p.default_device_path == "/dev/bitmain-asic"


def test_capabilities():
    p = detect_profile("Antminer S9")
    avail = p.capabilities(True)
    assert avail["is_hardware"] and avail["production_ready"]
    assert avail["hash_rate"] == int(13.5e12)
    assert avail["hardware_info"]["chip_count"] == 189
    assert avail["hardware_info"]["connection_type"] == "Network"
    assert avail["hardware_info"]["metadata"]["protocol"] == "cgminer-api"
    unavail = p.capabilities(False)
    assert not unavail["is_hardware"]


def test_registry_integrity():
    seen = set()
    for p in all_profiles():
        assert p.model and p.model not in seen
        seen.add(p.model)
        assert p.aliases and p.chip and p.protocol


def test_extract_type():
    resp = {"STATS": [{"Type": "Antminer S9"}]}
    assert _extract_type(resp) == "Antminer S9"
    assert _extract_type({}) is None
