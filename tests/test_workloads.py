import pytest

from ai_asic.hardware.bitcoin_header import prepare_asic_job
from ai_asic.server import HasherServer, VirtualAsicDevice
from ai_asic.workloads.split_model import (
    DEVICE_ASIC,
    DEVICE_HOST,
    MiningBackend,
    SplitModel,
    _software_mine,
)


@pytest.fixture
def running_server():
    srv = HasherServer(device=VirtualAsicDevice("Antminer S9"), port=0).start()
    try:
        yield srv
    finally:
        srv.stop()


def _model():
    return SplitModel(input_size=64, mining_neurons=3, output_size=10, difficulty_bits=10,
                      nonce_range=1 << 18)


# --- host-side execution -----------------------------------------------------

def test_infer_host_only_trace():
    model = _model()
    res = model.infer(b"hello world", MiningBackend(prefer_asic=False, difficulty_bits=10))
    stages = {s.name: s for s in res.trace.stages}
    assert set(stages) == {"encoder", "mining", "head"}
    assert stages["encoder"].device == DEVICE_HOST
    assert stages["head"].device == DEVICE_HOST
    assert stages["mining"].device == DEVICE_HOST  # no server -> host fallback
    assert stages["mining"].ops == 3
    assert 0 <= res.prediction < 10
    assert res.trace.asic_is_hardware is False
    assert res.trace.host_ops > 0 and res.trace.asic_ops == 0


def test_mining_backend_falls_back_without_server():
    # Point at a port with nothing listening -> software path, still works.
    be = MiningBackend("127.0.0.1", 1, difficulty_bits=8)
    assert be.is_hardware is False
    assert be.device == DEVICE_HOST
    header = prepare_asic_job([1] * 12, 0, timestamp=0)
    outcome = be.mine(header)
    assert outcome.found


# --- the split: same result, different device --------------------------------

def test_offload_matches_host(running_server):
    model = _model()
    data = b"optical alignment with AI"

    host_res = model.infer(data, MiningBackend(prefer_asic=False, difficulty_bits=10))
    asic_be = MiningBackend("127.0.0.1", running_server.port, difficulty_bits=10)
    assert asic_be.is_hardware is True
    asic_res = model.infer(data, asic_be)

    mining = {s.name: s for s in asic_res.trace.stages}["mining"]
    assert mining.device == DEVICE_ASIC
    assert asic_res.trace.asic_is_hardware is True
    assert asic_res.trace.asic_ops == 3

    # Identical maths; only the device that mined changed.
    assert asic_res.nonces == host_res.nonces
    assert asic_res.prediction == host_res.prediction


# --- persistence -------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    model = _model()
    data = b"serialize me"
    before = model.infer(data, MiningBackend(prefer_asic=False, difficulty_bits=10))

    path = tmp_path / "model.json"
    model.save(str(path))
    loaded = SplitModel.load(str(path))
    after = loaded.infer(data, MiningBackend(prefer_asic=False, difficulty_bits=10))

    assert loaded.mining_neurons == model.mining_neurons
    assert after.nonces == before.nonces
    assert after.prediction == before.prediction


def test_load_rejects_foreign_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"format": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError):
        SplitModel.load(str(path))


def test_software_mine_meets_target():
    header = prepare_asic_job([2] * 12, 0, timestamp=0)
    out = _software_mine(header, difficulty_bits=10, max_nonces=1 << 18)
    assert out.found
    assert out.leading_zeros >= 10
